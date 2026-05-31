"""
UHerding: Uncertainty Herding — greedy maximisation of Uncertainty Coverage.

UCoverage(S) = (1/N) Σ_n U(x_n) · max_{x'∈S} k_σ(x_n, x')

where k_σ(x, x') = exp(−‖x − x'‖² / σ²) (Gaussian kernel on L2-normalised features).

Adaptive parameters (from the paper):
  - σ* = min pairwise distance in selected set (→ 0 at high budget = pure uncertainty)
  - U   = margin uncertainty from LinearProbe (= 1 for cold-start = MaxHerding)

Reference: Bae et al. arXiv:2412.20644 (2024)
"""
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from . import register_sampler


@register_sampler("uncertainty_herding")
def uncertainty_herding_sampling(**kwargs) -> List[int]:
    """
    Greedy UHerding sampler — sliceable (run once at max_budget).

    Cold-start behaviour: σ is initialised to the mean NN distance of the full
    dataset (→ MaxHerding / pure coverage).  After `step_budget` selections a
    LinearProbe is fitted and σ is tightened to min pairwise dist within the
    selected set, shifting the objective toward uncertainty sampling.
    """
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 50)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]
    chunk_size = kwargs.get("chunk_size", 2000)

    from trainer import train_linear

    num_samples = image_embeddings.shape[0]
    # warm-up = first 20% of budget (at least num_classes samples)
    step_budget = max(num_classes, int(0.2 * max_budget))

    features = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features = F.normalize(features, p=2, dim=1)

    # ── Initial σ: mean nearest-neighbour distance (subsample for speed) ─────
    n_ref = min(1000, num_samples)
    ref_idx = np.random.choice(num_samples, n_ref, replace=False)
    ref = features[ref_idx]                              # (n_ref, D)
    # ||x - y||^2 = 2 - 2·cos(x, y) for L2-normalised vectors
    sim_ref = torch.matmul(ref, features.T)              # (n_ref, N)
    # exclude each point's self-similarity
    for i, gi in enumerate(ref_idx):
        sim_ref[i, gi] = -2.0
    nn_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sim_ref.max(dim=1).values, min=0.0))
    sigma = nn_dist.mean().item()
    sigma = max(sigma, 1e-3)
    del ref, sim_ref, nn_dist
    clear_memory()

    # ── State ─────────────────────────────────────────────────────────────────
    # U_n: uncertainty weight per unlabeled point (starts at 1 = MaxHerding)
    U = torch.ones(num_samples, device=device, dtype=torch.float32)
    # k_running[n] = max_{x'∈S} k_σ(x_n, x')
    k_running = torch.zeros(num_samples, device=device, dtype=torch.float32)

    selected_indices: List[int] = []
    selected_set: set = set()

    for step in tqdm(range(max_budget), desc="UHerding Selection"):

        # ── After warm-up: update σ* and uncertainty ─────────────────────────
        if step == step_budget and len(selected_indices) >= 2:
            # σ* = min pairwise distance in selected set
            sel_feats = features[selected_indices]       # (K, D)
            sel_sim = torch.matmul(sel_feats, sel_feats.T)
            sel_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sel_sim, min=0.0))
            sel_dist.fill_diagonal_(float("inf"))
            new_sigma = sel_dist.min().item()
            sigma = max(new_sigma, 1e-6)
            del sel_feats, sel_sim, sel_dist
            clear_memory()

            # Recompute k_running under new σ
            k_running.zero_()
            for si in selected_indices:
                si_feat = features[si].unsqueeze(0)      # (1, D)
                for cs in range(0, num_samples, chunk_size):
                    ce = min(cs + chunk_size, num_samples)
                    chunk = features[cs:ce]
                    sim_c = torch.matmul(chunk, si_feat.T).squeeze(1)
                    dist_sq_c = torch.clamp(2.0 - 2.0 * sim_c, min=0.0)
                    k_c = torch.exp(-dist_sq_c / (sigma ** 2))
                    k_running[cs:ce] = torch.maximum(k_running[cs:ce], k_c)
                    del chunk, sim_c, dist_sq_c, k_c
            clear_memory()

            # Margin uncertainty: U(x) = 1 − (p_1 − p_2)
            probe = train_linear(
                image_embeddings[selected_indices],
                oracle_labels[selected_indices],
                num_classes, probe_epochs, probe_lr, device,
            )
            probs = probe.predict_proba(image_embeddings, device)   # (N, C)
            s_probs = np.sort(probs, axis=1)
            margin = s_probs[:, -1] - s_probs[:, -2]
            U = torch.tensor(1.0 - margin, device=device, dtype=torch.float32)
            del probe
            clear_memory()

        # ── Greedy step: argmax_x̄ Σ_n U_n · max(k(x_n, x̄) − k_n, 0) ───────
        best_idx = -1
        best_score = -float("inf")
        best_k_col = None

        for cs in range(0, num_samples, chunk_size):
            ce = min(cs + chunk_size, num_samples)
            cand = features[cs:ce]                       # (C, D)

            sim = torch.matmul(features, cand.T)         # (N, C)
            dist_sq = torch.clamp(2.0 - 2.0 * sim, min=0.0)
            k_vals = torch.exp(-dist_sq / (sigma ** 2))  # (N, C)
            gain = torch.clamp(k_vals - k_running.unsqueeze(1), min=0.0)  # (N, C)
            scores = (U.unsqueeze(1) * gain).sum(dim=0)  # (C,)

            # Zero out already selected candidates
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
