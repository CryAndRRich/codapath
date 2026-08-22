from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.runtime import clear_memory
from .. import register_sampler


def _ece(logits: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    """Expected Calibration Error (Naeini et al., AAAI 2015), matching the
    official ECELoss (repos/uherding/deep-al/pycls/calibration/metrics.py):
    equal-width confidence bins over [0,1], weighted avg |confidence-accuracy|."""
    probs = F.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(torch.as_tensor(labels, dtype=torch.long))

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(lo.item()) & confidences.le(hi.item())
        prop_in_bin = in_bin.float().mean().item()
        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].float().mean().item()
            conf_in_bin = confidences[in_bin].mean().item()
            ece += abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return ece


def _calibrate_temperature(
    features_np: np.ndarray,
    labels: np.ndarray,
    selected_indices: List[int],
    num_classes: int,
    probe_epochs: int,
    probe_lr: float,
    device,
) -> float:
    """Paper Sec. 3.2 / Proposition 3: split L into train/val, train a
    THROWAWAY probe on train only, search temperatures minimizing ECE on val,
    return the best one. Matches official obtain_temperature()
    (repos/uherding/.../temperature_scaling.py): 70/30 split, grid
    np.arange(1.0, 20, 0.1), n_bins=num_classes. This calibration probe is
    separate from the "real" probe trained on the FULL selected set below —
    only the resulting temperature is reused, to rescale the real probe's
    logits before computing margin.
    """
    from training.probe import train_linear

    n_val = max(1, int(len(selected_indices) * 0.3))
    if len(selected_indices) - n_val < 2:
        return 1.0
    temp_train = selected_indices[:-n_val]
    temp_val = selected_indices[-n_val:]
    train_labels = labels[temp_train]
    val_labels = labels[temp_val]
    if len(np.unique(train_labels)) < 2 or len(temp_val) < 2:
        return 1.0

    calib_probe = train_linear(
        features_np[temp_train], train_labels, num_classes, probe_epochs, probe_lr, device,
    )
    val_logits = calib_probe.predict_logits(features_np[temp_val], device)
    del calib_probe

    best_temp, best_ece = 1.0, float("inf")
    for temp in np.arange(1.0, 20.0, 0.1):
        ece = _ece(val_logits / temp, val_labels, n_bins=num_classes)
        if ece < best_ece:
            best_ece = ece
            best_temp = float(temp)
    return best_temp


@register_sampler("uncertainty_herding")
def uncertainty_herding_sampling(**kwargs) -> List[int]:
    """Uncertainty Herding (Bae, Oliveira, Sutherland — arXiv:2412.20644).

    The official implementation (repos/uherding/deep-al/pycls/al/uherding.py)
    re-instantiates UHerding at the START OF EVERY AL round: the classifier is
    retrained on the current labeled set, sigma is recomputed as the min
    pairwise distance on that same labeled set, and only THEN does the round's
    batch get selected via the greedy weighted-coverage rule. The official
    config default is 5 rounds (MAX_ITER=5). This function reproduces that:
    round 0 has no labels yet (U=1, bootstrap sigma -> pure MaxHerding, exactly
    Proposition 3's limit), rounds 1..num_rounds-1 each retrain the probe and
    sigma from scratch on ALL labels revealed so far before greedily picking
    that round's share of the budget.

    Verified 2026-08-17 against the paper PDF directly (Definition 5,
    Algorithm 1, Propositions 3/4) and `uherding.py::UHerding.select_samples`
    line-by-line: the greedy update here (`k_running = torch.maximum(k_running,
    best_k_col)`) and the official code's (`max_embedding = updated_max_embedding
    [selected_index] + max_embedding`) are algebraically IDENTICAL
    (`max(a-b,0)+b == max(a,b)`), just written differently. One REAL,
    deliberate difference found and confirmed with the user: the official
    code subsamples the unlabeled pool to `compute_cand_size(...) <= 35000`
    candidates per round (`deep-al/pycls/utils/io.py`) — a GPU-memory
    optimization for their ImageNet-scale experiments (up to ~1.2M images),
    NOT part of Algorithm 1 itself (which defines the argmax over the FULL
    U_t, no subsampling). This function considers the FULL pool every round
    (chunked via `chunk_size` for memory, not subsampled) — arguably MORE
    faithful to Algorithm 1's literal definition than the official code's own
    scalability compromise, at the cost of being slower per round on pools
    larger than ~35k (e.g. PathMNIST). Kept as full-pool by explicit user
    decision (2026-08-17) rather than adding a matching subsample, to avoid
    introducing a new source of randomness for no fidelity gain at this
    project's pool sizes. This also applies to `refine.py`'s stage-2, which
    reuses `_calibrate_temperature` from this file and mirrors this same
    per-point loop.
    """
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 50)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]
    chunk_size = kwargs.get("chunk_size", 2000)
    num_rounds = kwargs.get("num_rounds", 5)

    from training.probe import train_linear

    num_samples = image_embeddings.shape[0]
    rounds = max(1, min(num_rounds, max_budget))
    base, remainder = divmod(max_budget, rounds)
    round_sizes = [base + (1 if r < remainder else 0) for r in range(rounds)]

    features = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features = F.normalize(features, p=2, dim=1)

    # Round-0 bootstrap sigma: no labeled set exists yet, so there is no
    # min-pairwise-distance to compute (paper doesn't define this case either).
    n_ref = min(1000, num_samples)
    ref_idx = np.random.choice(num_samples, n_ref, replace=False)
    ref = features[ref_idx]
    sim_ref = torch.matmul(ref, features.T)
    for i, gi in enumerate(ref_idx):
        sim_ref[i, gi] = -2.0
    nn_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sim_ref.max(dim=1).values, min=0.0))
    sigma = max(nn_dist.mean().item(), 1e-3)
    del ref, sim_ref, nn_dist
    clear_memory()

    U = torch.ones(num_samples, device=device, dtype=torch.float32)
    k_running = torch.zeros(num_samples, device=device, dtype=torch.float32)

    selected_indices: List[int] = []
    selected_set: set = set()

    for round_idx in range(rounds):
        n_select = round_sizes[round_idx]
        if n_select <= 0:
            continue

        if round_idx > 0 and len(selected_indices) >= 2:
            sel_feats = features[selected_indices]
            sel_sim = torch.matmul(sel_feats, sel_feats.T)
            sel_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sel_sim, min=0.0))
            sel_dist.fill_diagonal_(float("inf"))
            # Guard against near-duplicate selected points: if the closest PAIR
            # happens to be (near-)identical, flooring sigma to 1e-3 would collapse
            # the kernel to an indicator-of-duplicates for every other pair. Only
            # take a genuinely positive distance as the new bandwidth; otherwise
            # keep the previous round's sigma.
            valid_dist = sel_dist[sel_dist > 1e-6]
            if valid_dist.numel() > 0:
                sigma = max(valid_dist.min().item(), 1e-3)
            del sel_feats, sel_sim, sel_dist
            clear_memory()

            # Sigma changed -> the running-max coverage k_n must be rebuilt from
            # scratch against every previously selected point (not just carried
            # forward), since it was computed with the OLD bandwidth.
            k_running.zero_()
            for si in selected_indices:
                si_feat = features[si].unsqueeze(0)
                for cs in range(0, num_samples, chunk_size):
                    ce = min(cs + chunk_size, num_samples)
                    chunk = features[cs:ce]
                    sim_c = torch.matmul(chunk, si_feat.T).squeeze(1)
                    dist_sq_c = torch.clamp(2.0 - 2.0 * sim_c, min=0.0)
                    k_c = torch.exp(-dist_sq_c / (sigma ** 2))
                    k_running[cs:ce] = torch.maximum(k_running[cs:ce], k_c)
                    del chunk, sim_c, dist_sq_c, k_c
            clear_memory()

            norm_embeddings = features.cpu().numpy()
            tau = _calibrate_temperature(
                norm_embeddings, oracle_labels, selected_indices, num_classes,
                probe_epochs, probe_lr, device,
            )
            probe = train_linear(
                norm_embeddings[selected_indices],
                oracle_labels[selected_indices],
                num_classes, probe_epochs, probe_lr, device,
            )
            logits = probe.predict_logits(norm_embeddings, device)
            probs = F.softmax(torch.as_tensor(logits / tau, dtype=torch.float32), dim=1).numpy()
            s_probs = np.sort(probs, axis=1)
            margin = s_probs[:, -1] - s_probs[:, -2]
            U = torch.tensor(1.0 - margin, device=device, dtype=torch.float32)
            del probe
            clear_memory()

        for _ in tqdm(range(n_select), desc=f"UHerding Round {round_idx + 1}/{rounds}"):
            best_idx = -1
            best_score = -float("inf")
            best_k_col = None

            for cs in range(0, num_samples, chunk_size):
                ce = min(cs + chunk_size, num_samples)
                cand = features[cs:ce]

                sim = torch.matmul(features, cand.T)
                dist_sq = torch.clamp(2.0 - 2.0 * sim, min=0.0)
                k_vals = torch.exp(-dist_sq / (sigma ** 2))
                gain = torch.clamp(k_vals - k_running.unsqueeze(1), min=0.0)
                scores = (U.unsqueeze(1) * gain).sum(dim=0)

                for si in selected_set:
                    if cs <= si < ce:
                        scores[si - cs] = -float("inf")

                local_best = torch.argmax(scores).item()
                if scores[local_best].item() > best_score:
                    best_score = scores[local_best].item()
                    best_idx = cs + local_best
                    best_k_col = k_vals[:, local_best].clone()

                del cand, sim, dist_sq, k_vals, gain, scores
                clear_memory()

            if best_idx >= 0 and best_idx not in selected_set:
                selected_indices.append(best_idx)
                selected_set.add(best_idx)
                k_running = torch.maximum(k_running, best_k_col)
                del best_k_col
                clear_memory()
            else:
                break

    del features, U, k_running
    clear_memory()
    return selected_indices
