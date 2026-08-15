"""Sequential two-branch UHerding sampler (Task 14/8 — Minh Hai).

Branch 1 (CellViT cell embeddings):
    Score all unlabelled samples via UHerding -> keep top 50%.
Branch 2 (DINO visual embeddings):
    From the filtered candidates, greedily select A samples via UHerding.

Both branches maintain independent sigma / k_running / probe, but share the
same selected set: once Branch 2 picks a sample, coverage is updated in
BOTH spaces for the next round.
"""

from typing import List, Set

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from . import register_sampler


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------

def _bootstrap_sigma(features: torch.Tensor, num_samples: int) -> float:
    """Estimate initial bandwidth from nearest-neighbour distances."""
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
    return sigma


def _update_sigma(features: torch.Tensor, selected: List[int], old_sigma: float) -> float:
    """Recompute sigma as min pairwise distance among selected points."""
    sel_feats = features[selected]
    sel_sim = torch.matmul(sel_feats, sel_feats.T)
    sel_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sel_sim, min=0.0))
    sel_dist.fill_diagonal_(float("inf"))
    valid = sel_dist[sel_dist > 1e-6]
    sigma = max(valid.min().item(), 1e-3) if valid.numel() > 0 else old_sigma
    del sel_feats, sel_sim, sel_dist
    clear_memory()
    return sigma


def _rebuild_k_running(
    features: torch.Tensor,
    selected: List[int],
    sigma: float,
    num_samples: int,
    chunk_size: int,
    device,
) -> torch.Tensor:
    """Rebuild coverage vector from scratch for given selected set and sigma."""
    k_running = torch.zeros(num_samples, device=device, dtype=torch.float32)
    for si in selected:
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
    return k_running


def _update_k_running_single(
    k_running: torch.Tensor,
    features: torch.Tensor,
    idx: int,
    sigma: float,
    num_samples: int,
    chunk_size: int,
) -> torch.Tensor:
    """Incrementally update k_running after selecting a single new point."""
    si_feat = features[idx].unsqueeze(0)
    for cs in range(0, num_samples, chunk_size):
        ce = min(cs + chunk_size, num_samples)
        chunk = features[cs:ce]
        sim_c = torch.matmul(chunk, si_feat.T).squeeze(1)
        dist_sq_c = torch.clamp(2.0 - 2.0 * sim_c, min=0.0)
        k_c = torch.exp(-dist_sq_c / (sigma ** 2))
        k_running[cs:ce] = torch.maximum(k_running[cs:ce], k_c)
        del chunk, sim_c, dist_sq_c, k_c
    return k_running


def _compute_uncertainty(
    features_np: np.ndarray,
    oracle_labels: np.ndarray,
    selected: List[int],
    num_classes: int,
    probe_epochs: int,
    probe_lr: float,
    device,
    num_samples: int,
) -> torch.Tensor:
    """Train a probe, calibrate temperature, compute margin uncertainty."""
    from trainer import train_linear
    from .uncertainty_herding import _calibrate_temperature

    tau = _calibrate_temperature(
        features_np, oracle_labels, selected,
        num_classes, probe_epochs, probe_lr, device,
    )
    probe = train_linear(
        features_np[selected], oracle_labels[selected],
        num_classes, probe_epochs, probe_lr, device,
    )
    logits = probe.predict_logits(features_np, device)
    probs = F.softmax(
        torch.as_tensor(logits / tau, dtype=torch.float32), dim=1
    ).numpy()
    s_probs = np.sort(probs, axis=1)
    margin = s_probs[:, -1] - s_probs[:, -2]
    U = torch.tensor(1.0 - margin, device=device, dtype=torch.float32)
    del probe
    clear_memory()
    return U


def _compute_scores_all(
    features: torch.Tensor,
    U: torch.Tensor,
    k_running: torch.Tensor,
    sigma: float,
    num_samples: int,
    chunk_size: int,
    exclude: Set[int],
) -> torch.Tensor:
    """Compute UHerding score for every sample (for Branch 1 filtering).

    score(c) = sum_i [ U(i) * max(0, k(x_i, x_c) - k_running(i)) ]
    """
    all_scores = torch.full((num_samples,), -float("inf"), device=features.device)

    for cs in range(0, num_samples, chunk_size):
        ce = min(cs + chunk_size, num_samples)
        cand = features[cs:ce]
        sim = torch.matmul(features, cand.T)
        dist_sq = torch.clamp(2.0 - 2.0 * sim, min=0.0)
        k_vals = torch.exp(-dist_sq / (sigma ** 2))
        gain = torch.clamp(k_vals - k_running.unsqueeze(1), min=0.0)
        scores = (U.unsqueeze(1) * gain).sum(dim=0)

        for si in exclude:
            if cs <= si < ce:
                scores[si - cs] = -float("inf")

        all_scores[cs:ce] = scores
        del cand, sim, dist_sq, k_vals, gain, scores
        clear_memory()

    return all_scores


def _greedy_select(
    features: torch.Tensor,
    U: torch.Tensor,
    k_running: torch.Tensor,
    sigma: float,
    num_samples: int,
    chunk_size: int,
    candidates: Set[int],
    exclude: Set[int],
    n_select: int,
    round_label: str,
) -> List[int]:
    """Greedy UHerding selection restricted to candidate set."""
    newly_selected: List[int] = []
    local_exclude = set(exclude)

    for _ in tqdm(range(n_select), desc=round_label):
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

            # Mask: exclude already-selected AND non-candidates
            for j in range(cs, ce):
                if j in local_exclude or j not in candidates:
                    scores[j - cs] = -float("inf")

            local_best = torch.argmax(scores).item()
            if scores[local_best].item() > best_score:
                best_score = scores[local_best].item()
                best_idx = cs + local_best
                best_k_col = k_vals[:, local_best].clone()

            del cand, sim, dist_sq, k_vals, gain, scores
            clear_memory()

        if best_idx >= 0 and best_idx not in local_exclude:
            newly_selected.append(best_idx)
            local_exclude.add(best_idx)
            k_running = torch.maximum(k_running, best_k_col)
            del best_k_col
            clear_memory()
        else:
            break

    return newly_selected


# ---------------------------------------------------------------------------
#  Main sequential sampler
# ---------------------------------------------------------------------------

def _sequential_cell_dino(**kwargs) -> List[int]:
    dino_embeddings = kwargs["image_embeddings"]      # (N, D_dino)
    cell_embeddings = kwargs["cell_embeddings"]        # (N, D_cell)
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 50)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]
    chunk_size = kwargs.get("chunk_size", 2000)
    num_rounds = kwargs.get("num_rounds", 5)

    num_samples = dino_embeddings.shape[0]
    rounds = max(1, min(num_rounds, max_budget))
    base, remainder = divmod(max_budget, rounds)
    round_sizes = [base + (1 if r < remainder else 0) for r in range(rounds)]

    # Normalise features
    dino_feat = F.normalize(
        torch.tensor(dino_embeddings, device=device, dtype=torch.float32), p=2, dim=1
    )
    cell_feat = F.normalize(
        torch.tensor(cell_embeddings, device=device, dtype=torch.float32), p=2, dim=1
    )

    # Numpy copies for probe training
    dino_np = dino_feat.cpu().numpy()
    cell_np = cell_feat.cpu().numpy()

    # Initial sigma
    sigma_cell = _bootstrap_sigma(cell_feat, num_samples)
    sigma_dino = _bootstrap_sigma(dino_feat, num_samples)

    # Coverage tracking — both spaces
    k_running_cell = torch.zeros(num_samples, device=device, dtype=torch.float32)
    k_running_dino = torch.zeros(num_samples, device=device, dtype=torch.float32)

    # Uncertainty — both branches
    U_cell = torch.ones(num_samples, device=device, dtype=torch.float32)
    U_dino = torch.ones(num_samples, device=device, dtype=torch.float32)

    selected_indices: List[int] = []
    selected_set: Set[int] = set()

    for round_idx in range(rounds):
        A = round_sizes[round_idx]
        if A <= 0:
            continue

        # ---- Update probes, sigma, k_running if round > 0 ----
        if round_idx > 0 and len(selected_indices) >= 2:
            # Cell space
            sigma_cell = _update_sigma(cell_feat, selected_indices, sigma_cell)
            k_running_cell = _rebuild_k_running(
                cell_feat, selected_indices, sigma_cell,
                num_samples, chunk_size, device,
            )
            U_cell = _compute_uncertainty(
                cell_np, oracle_labels, selected_indices,
                num_classes, probe_epochs, probe_lr, device, num_samples,
            )

            # DINO space
            sigma_dino = _update_sigma(dino_feat, selected_indices, sigma_dino)
            k_running_dino = _rebuild_k_running(
                dino_feat, selected_indices, sigma_dino,
                num_samples, chunk_size, device,
            )
            U_dino = _compute_uncertainty(
                dino_np, oracle_labels, selected_indices,
                num_classes, probe_epochs, probe_lr, device, num_samples,
            )

        # ---- Branch 1: Filter using cell embeddings ----
        cell_scores = _compute_scores_all(
            cell_feat, U_cell, k_running_cell, sigma_cell,
            num_samples, chunk_size, selected_set,
        )

        # Keep top 50% of unlabelled samples
        unlabelled_mask = torch.ones(num_samples, dtype=torch.bool, device=device)
        for si in selected_set:
            unlabelled_mask[si] = False
        unlabelled_indices = torch.where(unlabelled_mask)[0]
        unlabelled_scores = cell_scores[unlabelled_indices]

        n_candidates = max(A, len(unlabelled_indices) // 2)
        _, top_local = torch.topk(unlabelled_scores, min(n_candidates, len(unlabelled_indices)))
        candidates = set(unlabelled_indices[top_local].cpu().tolist())

        print(f"  [Round {round_idx+1}/{rounds}] Branch 1 filtered "
              f"{len(unlabelled_indices)} -> {len(candidates)} candidates")

        # ---- Branch 2: Select from candidates using DINO ----
        newly_selected = _greedy_select(
            dino_feat, U_dino, k_running_dino, sigma_dino,
            num_samples, chunk_size,
            candidates, selected_set, A,
            f"Round {round_idx+1}/{rounds} Branch2-DINO",
        )

        # Update both coverage spaces
        for idx in newly_selected:
            selected_indices.append(idx)
            selected_set.add(idx)
            k_running_cell = _update_k_running_single(
                k_running_cell, cell_feat, idx, sigma_cell, num_samples, chunk_size,
            )
            # k_running_dino already updated inside _greedy_select

        print(f"  [Round {round_idx+1}/{rounds}] Selected {len(newly_selected)} samples "
              f"(total: {len(selected_indices)})")

    del dino_feat, cell_feat, k_running_cell, k_running_dino, U_cell, U_dino
    clear_memory()
    return selected_indices


@register_sampler("seq_cell_dino_mean")
def seq_cell_dino_mean_sampling(**kwargs) -> List[int]:
    return _sequential_cell_dino(**kwargs)


@register_sampler("seq_cell_dino_kde")
def seq_cell_dino_kde_sampling(**kwargs) -> List[int]:
    return _sequential_cell_dino(**kwargs)
