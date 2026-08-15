from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from . import register_sampler


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

    from trainer import train_linear

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
            probe = train_linear(
                norm_embeddings[selected_indices],
                oracle_labels[selected_indices],
                num_classes, probe_epochs, probe_lr, device,
            )
            probs = probe.predict_proba(norm_embeddings, device)
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
