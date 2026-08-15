from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from set_up import clear_memory
from . import register_sampler, get_sampler
from .uncertainty_herding import _calibrate_temperature


def _stage2_coverage_select(
    features: torch.Tensor,
    candidate_indices: List[int],
    labeled_indices: List[int],
    oracle_labels: np.ndarray,
    num_classes: int,
    n_select: int,
    probe_epochs: int,
    probe_lr: float,
    device,
) -> List[int]:
    """REFINE Eq. (2): B* = argmax_{B⊂CR,|B|=b} E_x[max_{x'∈(Lt∪B)} k(x,x')].

    Coverage target x ranges over the refined pool CR (`candidate_indices`),
    the "already covered" reference set is Lt (all REAL labels accumulated
    across every previous AL cycle, `labeled_indices` — may include points
    outside CR) union the batch B built up during this greedy pass. Paper
    adopts UHerding (uncertainty-weighted coverage) for this step, so the
    per-x weight is margin-uncertainty from a probe trained on Lt whenever Lt
    has at least 2 points; the very first cycle (Lt empty) falls back to pure
    coverage (MaxHerding), matching UHerding's own U->1 cold-start behavior.
    """
    from trainer import train_linear

    cand_t = torch.as_tensor(candidate_indices, device=device, dtype=torch.long)
    cand_feats = features[cand_t]

    if len(labeled_indices) >= 2:
        sel_feats = features[labeled_indices]
        sel_sim = torch.matmul(sel_feats, sel_feats.T)
        sel_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sel_sim, min=0.0))
        sel_dist.fill_diagonal_(float("inf"))
        valid_dist = sel_dist[sel_dist > 1e-6]
        sigma = max(valid_dist.min().item(), 1e-3) if valid_dist.numel() > 0 else 1e-3
        del sel_feats, sel_sim, sel_dist

        norm_embeddings = features.cpu().numpy()
        tau = _calibrate_temperature(
            norm_embeddings, oracle_labels, labeled_indices, num_classes,
            probe_epochs, probe_lr, device,
        )
        probe = train_linear(
            norm_embeddings[labeled_indices],
            oracle_labels[labeled_indices],
            num_classes, probe_epochs, probe_lr, device,
        )
        logits = probe.predict_logits(norm_embeddings[candidate_indices], device)
        probs = F.softmax(torch.as_tensor(logits / tau, dtype=torch.float32), dim=1).numpy()
        s_probs = np.sort(probs, axis=1)
        margin = s_probs[:, -1] - s_probs[:, -2]
        U = torch.tensor(1.0 - margin, device=device, dtype=torch.float32)
        del probe

        labeled_t = torch.as_tensor(labeled_indices, device=device, dtype=torch.long)
        labeled_feats = features[labeled_t]
        sim_lc = torch.matmul(labeled_feats, cand_feats.T)  # (|Lt|, |CR|)
        dist_sq_lc = torch.clamp(2.0 - 2.0 * sim_lc, min=0.0)
        k_lc = torch.exp(-dist_sq_lc / (sigma ** 2))
        k_running = k_lc.max(dim=0).values if len(labeled_indices) > 0 else torch.zeros(
            len(candidate_indices), device=device
        )
        del labeled_feats, sim_lc, dist_sq_lc, k_lc
    else:
        # Round 0 of the very first cycle: no real labels exist yet -> bootstrap
        # sigma from the candidate pool itself, U≡1 (pure MaxHerding — exactly
        # UHerding's Proposition 3 limit).
        n_ref = min(1000, len(candidate_indices))
        ref_idx = np.random.choice(len(candidate_indices), n_ref, replace=False)
        ref = cand_feats[ref_idx]
        sim_ref = torch.matmul(ref, cand_feats.T)
        for i, gi in enumerate(ref_idx):
            sim_ref[i, gi] = -2.0
        nn_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sim_ref.max(dim=1).values, min=0.0))
        sigma = max(nn_dist.mean().item(), 1e-3)
        del ref, sim_ref, nn_dist
        U = torch.ones(len(candidate_indices), device=device, dtype=torch.float32)
        k_running = torch.zeros(len(candidate_indices), device=device, dtype=torch.float32)

    n_cand = len(candidate_indices)
    selected_local: List[int] = []
    selected_set: set = set()

    for _ in range(min(n_select, n_cand)):
        sim = torch.matmul(cand_feats, cand_feats.T)  # (|CR|,|CR|)
        dist_sq = torch.clamp(2.0 - 2.0 * sim, min=0.0)
        k_vals = torch.exp(-dist_sq / (sigma ** 2))
        gain = torch.clamp(k_vals - k_running.unsqueeze(1), min=0.0)
        scores = (U.unsqueeze(1) * gain).sum(dim=0)
        for si in selected_set:
            scores[si] = -float("inf")

        best_local = int(torch.argmax(scores).item())
        selected_local.append(best_local)
        selected_set.add(best_local)
        k_running = torch.maximum(k_running, k_vals[:, best_local])
        del sim, dist_sq, k_vals, gain, scores
        clear_memory()

    return [candidate_indices[i] for i in selected_local]


@register_sampler("refine")
def refine_sampling(**kwargs) -> List[int]:
    """REFINE (CVPR 2026): progressive ensemble pool-filtering -> UHerding coverage.

    Paper (references/REFINE.md, Sec. 3): the AL loop runs A CYCLES, each
    labeling b instances (B = b + A*b). Each cycle has two stages: (1)
    progressive filtering — R rounds where every base strategy in a DIVERSE
    ensemble (coreset = representative, typiclust = density, margin =
    uncertainty, badge = gradient diversity) queries J random subsamples of
    the CURRENT unlabeled pool; the union of picks becomes the refined pool
    CR (round 1 draws a fixed-size subsample, `init_subset_size`); (2)
    coverage-based selection — UHerding picks b points from CR (Eq. 2),
    weighted by uncertainty from a probe trained on ALL real labels
    accumulated across every previous cycle.

    This mirrors that with `num_rounds` outer cycles (default 5, matching the
    rest of this codebase's iterative samplers) instead of the earlier
    single-shot approximation, which ran progressive filtering once over the
    WHOLE pool and picked the entire budget in one coverage pass — collapsing
    away the paper's core "informed by growing Lt across cycles" mechanism.
    """
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels    = kwargs["oracle_labels"]
    max_budget       = kwargs["max_budget"]
    num_classes      = kwargs["num_classes"]
    device           = kwargs["device"]
    R                = kwargs.get("filter_rounds", 5)
    alpha            = kwargs.get("filter_alpha", 0.4)
    J                = kwargs.get("filter_batches", 5)
    init_subset      = kwargs.get("init_subset_size", 5000)
    probe_epochs     = kwargs.get("probe_epochs", 30)
    probe_lr         = kwargs.get("probe_lr", 1e-3)
    chunk_size       = kwargs.get("chunk_size", 2000)
    num_rounds       = kwargs.get("num_rounds", 5)

    num_samples = image_embeddings.shape[0]
    strategies = ["coreset", "typiclust", "margin", "badge"]

    features = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features = F.normalize(features, p=2, dim=1)

    cycles = max(1, min(num_rounds, max_budget))
    base, remainder = divmod(max_budget, cycles)
    cycle_sizes = [base + (1 if r < remainder else 0) for r in range(cycles)]

    selected_indices: List[int] = []
    selected_set: set = set()

    for cycle_idx in range(cycles):
        b = cycle_sizes[cycle_idx]
        if b <= 0:
            continue

        # --- Stage 1: progressive filtering over the CURRENT unlabeled pool ---
        pool_indices = [i for i in range(num_samples) if i not in selected_set]
        if len(pool_indices) <= b:
            selected_indices.extend(pool_indices)
            selected_set.update(pool_indices)
            continue

        for r in range(R):
            if len(pool_indices) <= b:
                break

            if r == 0:
                sample_size = min(init_subset, len(pool_indices))
            else:
                sample_size = min(max(b + 1, int(alpha * len(pool_indices))), len(pool_indices))

            next_pool: set = set()
            for s_name in strategies:
                for _ in range(J):
                    sub_local = np.random.choice(len(pool_indices), sample_size, replace=False)
                    sub_global = [pool_indices[i] for i in sub_local]

                    local_sel = get_sampler(
                        s_name,
                        image_embeddings=image_embeddings[sub_global],
                        oracle_labels=oracle_labels[sub_global],
                        num_classes=num_classes,
                        max_budget=min(b, sample_size),
                        device=device,
                        chunk_size=chunk_size,
                        probe_epochs=probe_epochs,
                        probe_lr=probe_lr,
                    )
                    for li in local_sel:
                        next_pool.add(sub_global[li])

            if len(next_pool) < b:
                extras = [i for i in pool_indices if i not in next_pool]
                need = b - len(next_pool)
                next_pool.update(np.random.choice(extras, min(need, len(extras)), replace=False).tolist())

            pool_indices = list(next_pool)

        # --- Stage 2: uncertainty-weighted coverage select over the refined pool ---
        cycle_picks = _stage2_coverage_select(
            features, pool_indices, selected_indices, oracle_labels, num_classes,
            b, probe_epochs, probe_lr, device,
        )

        if len(cycle_picks) < b:
            used = selected_set | set(cycle_picks)
            extras = [i for i in range(num_samples) if i not in used]
            need = b - len(cycle_picks)
            cycle_picks.extend(np.random.choice(extras, min(need, len(extras)), replace=False).tolist())

        selected_indices.extend(cycle_picks)
        selected_set.update(cycle_picks)

    del features
    clear_memory()

    return selected_indices
