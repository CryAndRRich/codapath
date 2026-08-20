"""Reliability-aware dual-view Uncertainty Herding for pathology patches.

The primary sampler builds one UCoverage objective in full-image DINO space
and another in a pooled CellViT space.  Each branch greedily proposes ``2A``
items per AL round, then the final batch of ``A`` is selected from their union
by joint marginal gain (default) or an explicit rank/score-fusion ablation.

``disagreement_uherding`` is a strict control: it keeps only DINO coverage and
replaces UHerding's calibrated margin weight by calibrated DINO/CellViT JSD.

For fixed non-negative target weights within a round, each branch is weighted
facility location and is monotone submodular.  Their fixed positive sum is
also monotone submodular.  The usual (1 - 1/e) guarantee applies to full-pool
joint greedy; shortlist fusion only inherits it relative to the restricted
union, and RRF/Borda/static-score fusion has no such guarantee.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nucleus.uncertainty import (
    js_disagreement_from_logits,
    margin_uncertainty_from_logits,
)
from set_up import clear_memory
from trainer import train_dual_linear, train_linear
from . import register_sampler
from .nucleus_coverage import _build_coverage_features, _running_max_coverage
from .scalpel import _adaptive_sigma, _k_col, _k_gaussian
from .uncertainty_herding import _calibrate_temperature


VALID_UNCERTAINTIES = {"branch_margin", "disagreement"}
VALID_FUSIONS = {
    "joint", "rrf", "borda", "score", "sum", "mean",
    "visual_then_cell", "cell_then_visual",
}
VALID_COVERAGE_MODES = {"dual", "dino"}


def _normalize_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.clip(norms, 1e-12, None)).astype(np.float32)


def _can_train(labels: np.ndarray) -> bool:
    return len(labels) >= 2 and len(np.unique(labels)) >= 2


def _labeled_positive_sigma(
    features: torch.Tensor,
    selected: Sequence[int],
    previous_sigma: float,
) -> float:
    """Minimum genuinely-positive labeled distance, matching updated UH."""
    if len(selected) < 2:
        return previous_sigma
    idx = torch.as_tensor(selected, device=features.device, dtype=torch.long)
    labeled = features[idx]
    similarity = torch.matmul(labeled, labeled.T)
    distance = torch.sqrt(torch.clamp(2.0 - 2.0 * similarity, min=0.0))
    distance.fill_diagonal_(float("inf"))
    positive = distance[(distance > 1e-6) & torch.isfinite(distance)]
    if positive.numel() == 0:
        return previous_sigma
    return max(float(positive.min().item()), 1e-3)


def _weighted_marginal_scores(
    features: torch.Tensor,
    target_weights: torch.Tensor,
    running_coverage: torch.Tensor,
    sigma: float,
    candidate_indices: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Exact UCoverage marginal gains for an explicit candidate set.

    Both target and candidate axes are chunked.  This has the same objective
    as the original ``N x candidate_chunk`` implementation but caps temporary
    kernel memory at roughly ``chunk_size ** 2`` floats.
    """
    candidates_np = np.asarray(candidate_indices, dtype=np.int64)
    if len(candidates_np) == 0:
        return np.empty(0, dtype=np.float32)
    n = len(features)
    result = np.zeros(len(candidates_np), dtype=np.float32)
    for cs in range(0, len(candidates_np), chunk_size):
        ce = min(cs + chunk_size, len(candidates_np))
        idx = torch.as_tensor(
            candidates_np[cs:ce], device=features.device, dtype=torch.long
        )
        candidate_features = features[idx]
        score = torch.zeros(ce - cs, device=features.device, dtype=torch.float32)
        for ns in range(0, n, chunk_size):
            ne = min(ns + chunk_size, n)
            kernel = _k_gaussian(features[ns:ne], candidate_features, sigma)
            gain = torch.clamp(
                kernel - running_coverage[ns:ne].unsqueeze(1), min=0.0
            )
            score += (target_weights[ns:ne].unsqueeze(1) * gain).sum(dim=0)
            del kernel, gain
        result[cs:ce] = score.detach().cpu().numpy().astype(np.float32)
        del idx, candidate_features, score
    return result


def _greedy_branch_order(
    features: torch.Tensor,
    target_weights: torch.Tensor,
    running_coverage: torch.Tensor,
    sigma: float,
    candidate_indices: np.ndarray,
    n_select: int,
    chunk_size: int,
) -> Tuple[List[int], List[float]]:
    """Return a branch's greedy order and its conditional selected gains."""
    remaining = np.unique(np.asarray(candidate_indices, dtype=np.int64))
    running = running_coverage.clone()
    order: List[int] = []
    gains: List[float] = []
    for _ in range(min(n_select, len(remaining))):
        scores = _weighted_marginal_scores(
            features, target_weights, running, sigma, remaining, chunk_size
        )
        best_position = int(np.argmax(scores))
        best_index = int(remaining[best_position])
        order.append(best_index)
        gains.append(float(scores[best_position]))
        best_column = _k_col(
            features, features[best_index].unsqueeze(0), sigma, chunk_size
        )
        running.copy_(torch.maximum(running, best_column))
        remaining = np.delete(remaining, best_position)
        del scores, best_column
    del running
    clear_memory()
    return order, gains


def _precompute_relevant_kernel(
    features: torch.Tensor,
    relevant_indices: np.ndarray,
    sigma: float,
    chunk_size: int,
) -> torch.Tensor:
    """Materialize one round's relevant-set kernel in row chunks.

    Official UHerding constructs its kernel over labeled points plus a sampled
    candidate set and reuses it throughout the greedy batch.  Doing the same
    is essential here: recomputing an ``N x candidates`` kernel for every one
    of the two ``2A`` proposal sequences is prohibitively expensive.
    """
    indices = torch.as_tensor(
        relevant_indices, device=features.device, dtype=torch.long
    )
    relevant_features = features[indices]
    size = len(relevant_indices)
    kernel = torch.empty(
        (size, size), device=features.device, dtype=torch.float32
    )
    for start in range(0, size, chunk_size):
        end = min(start + chunk_size, size)
        kernel[start:end] = _k_gaussian(
            relevant_features[start:end], relevant_features, sigma
        )
    del indices, relevant_features
    return kernel


def _initial_kernel_coverage(
    kernel: torch.Tensor,
    selected_positions: Sequence[int],
) -> torch.Tensor:
    if not selected_positions:
        return torch.zeros(
            len(kernel), device=kernel.device, dtype=torch.float32
        )
    idx = torch.as_tensor(
        selected_positions, device=kernel.device, dtype=torch.long
    )
    return kernel[:, idx].max(dim=1).values


def _kernel_greedy_order(
    kernel: torch.Tensor,
    target_weights: torch.Tensor,
    running_coverage: torch.Tensor,
    candidate_positions: np.ndarray,
    relevant_indices: np.ndarray,
    n_select: int,
) -> Tuple[List[int], List[float]]:
    """UH greedy over a precomputed relevant-set kernel."""
    remaining = np.unique(np.asarray(candidate_positions, dtype=np.int64))
    running = running_coverage.clone()
    order: List[int] = []
    gains: List[float] = []
    for _ in range(min(n_select, len(remaining))):
        positions = torch.as_tensor(
            remaining, device=kernel.device, dtype=torch.long
        )
        gain = torch.clamp(
            kernel[:, positions] - running.unsqueeze(1), min=0.0
        )
        scores = (target_weights.unsqueeze(1) * gain).sum(dim=0)
        best_local = int(torch.argmax(scores).item())
        best_position = int(remaining[best_local])
        order.append(int(relevant_indices[best_position]))
        gains.append(float(scores[best_local].item()))
        running.copy_(torch.maximum(running, kernel[:, best_position]))
        remaining = np.delete(remaining, best_local)
        del positions, gain, scores
    del running
    return order, gains


def _kernel_joint_greedy(
    dino_kernel: torch.Tensor,
    cell_kernel: torch.Tensor,
    dino_weights: torch.Tensor,
    cell_weights: torch.Tensor,
    dino_running: torch.Tensor,
    cell_running: torch.Tensor,
    relevant_indices: np.ndarray,
    union: Sequence[int],
    valid_cell: np.ndarray,
    n_select: int,
    alpha: float,
) -> List[int]:
    """Joint greedy over the shortlist using two precomputed kernels."""
    global_to_local = {
        int(global_index): position
        for position, global_index in enumerate(relevant_indices)
    }
    remaining = np.asarray(
        sorted({global_to_local[int(index)] for index in union}),
        dtype=np.int64,
    )
    dino_state = dino_running.clone()
    cell_state = cell_running.clone()
    dino_scale: Optional[float] = None
    cell_scale: Optional[float] = None
    picks: List[int] = []
    for _ in range(min(n_select, len(remaining))):
        positions = torch.as_tensor(
            remaining, device=dino_kernel.device, dtype=torch.long
        )
        dino_gain = torch.clamp(
            dino_kernel[:, positions] - dino_state.unsqueeze(1), min=0.0
        )
        cell_gain = torch.clamp(
            cell_kernel[:, positions] - cell_state.unsqueeze(1), min=0.0
        )
        dino_score = (dino_weights.unsqueeze(1) * dino_gain).sum(dim=0)
        cell_score = (cell_weights.unsqueeze(1) * cell_gain).sum(dim=0)
        candidate_global = relevant_indices[remaining]
        cell_score *= torch.as_tensor(
            valid_cell[candidate_global],
            device=cell_score.device,
            dtype=torch.float32,
        )
        if dino_scale is None:
            dino_scale = max(float(dino_score.max().item()), 1e-12)
            cell_scale = max(float(cell_score.max().item()), 1e-12)
        combined = alpha * dino_score / dino_scale
        combined += (1.0 - alpha) * cell_score / cell_scale
        best_local = int(torch.argmax(combined).item())
        best_position = int(remaining[best_local])
        best_global = int(relevant_indices[best_position])
        picks.append(best_global)
        dino_state.copy_(
            torch.maximum(dino_state, dino_kernel[:, best_position])
        )
        if valid_cell[best_global]:
            cell_state.copy_(
                torch.maximum(cell_state, cell_kernel[:, best_position])
            )
        remaining = np.delete(remaining, best_local)
        del positions, dino_gain, cell_gain, dino_score, cell_score, combined
    del dino_state, cell_state
    return picks


def _static_fusion(
    visual_order: Sequence[int],
    cell_order: Sequence[int],
    visual_gains: Sequence[float],
    cell_gains: Sequence[float],
    n_select: int,
    mode: str,
    rrf_k: float,
) -> List[int]:
    """RRF, normalized Borda, or normalized selected-gain fusion."""
    union = sorted(set(visual_order) | set(cell_order))
    if not union:
        return []
    visual_rank = {idx: rank + 1 for rank, idx in enumerate(visual_order)}
    cell_rank = {idx: rank + 1 for rank, idx in enumerate(cell_order)}
    visual_score = dict(zip(visual_order, visual_gains))
    cell_score = dict(zip(cell_order, cell_gains))
    scores: Dict[int, float] = {}

    if mode == "rrf":
        for idx in union:
            scores[idx] = 0.0
            if idx in visual_rank:
                scores[idx] += 1.0 / (rrf_k + visual_rank[idx])
            if idx in cell_rank:
                scores[idx] += 1.0 / (rrf_k + cell_rank[idx])
    elif mode == "borda":
        visual_den = max(len(visual_order), 1)
        cell_den = max(len(cell_order), 1)
        for idx in union:
            v = (
                1.0 - (visual_rank[idx] - 1) / visual_den
                if idx in visual_rank else 0.0
            )
            c = (
                1.0 - (cell_rank[idx] - 1) / cell_den
                if idx in cell_rank else 0.0
            )
            scores[idx] = 0.5 * (v + c)
    else:  # score/sum/mean are rank-equivalent for two equally weighted views
        visual_scale = max(max(visual_gains, default=0.0), 1e-12)
        cell_scale = max(max(cell_gains, default=0.0), 1e-12)
        for idx in union:
            scores[idx] = 0.5 * (
                visual_score.get(idx, 0.0) / visual_scale
                + cell_score.get(idx, 0.0) / cell_scale
            )
    return sorted(union, key=lambda idx: (-scores[idx], idx))[:n_select]


def _round_uncertainties(
    dino_np: np.ndarray,
    cell_np: np.ndarray,
    reliability: np.ndarray,
    labels: np.ndarray,
    selected: Sequence[int],
    num_classes: int,
    probe_epochs: int,
    probe_lr: float,
    probe_weight_decay: float,
    consistency_weight: float,
    consistency_mode: str,
    device: torch.device,
    uncertainty_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Fit/calibrate both probes and return per-target branch weights."""
    selected_np = np.asarray(selected, dtype=np.int64)
    valid = reliability > 0.0
    image_labels = labels[selected_np]
    cell_selected = selected_np[valid[selected_np]]
    cell_labels = labels[cell_selected]
    image_trainable = _can_train(image_labels)
    cell_trainable = _can_train(cell_labels)
    diagnostics: Dict[str, float] = {
        "tau_visual": 1.0,
        "tau_cell": 1.0,
        "mean_disagreement": float("nan"),
    }

    if not image_trainable:
        return (
            np.ones(len(dino_np), dtype=np.float32),
            reliability.astype(np.float32),
            diagnostics,
        )

    tau_visual = _calibrate_temperature(
        dino_np, labels, selected_np.tolist(), num_classes,
        probe_epochs, probe_lr, device,
    )
    if cell_trainable:
        tau_cell = _calibrate_temperature(
            cell_np, labels, cell_selected.tolist(), num_classes,
            probe_epochs, probe_lr, device,
        )
    else:
        tau_cell = 1.0

    if cell_trainable and consistency_weight > 0.0:
        image_probe, cell_probe = train_dual_linear(
            dino_np[selected_np],
            cell_np[selected_np],
            image_labels,
            num_classes,
            probe_epochs,
            probe_lr,
            device,
            cell_valid=valid[selected_np],
            cell_reliability=reliability[selected_np],
            consistency_weight=consistency_weight,
            consistency_mode=consistency_mode,
            weight_decay=probe_weight_decay,
        )
    else:
        image_probe = train_linear(
            dino_np[selected_np], image_labels, num_classes,
            probe_epochs, probe_lr, device, weight_decay=probe_weight_decay,
        )
        cell_probe = (
            train_linear(
                cell_np[cell_selected], cell_labels, num_classes,
                probe_epochs, probe_lr, device,
                weight_decay=probe_weight_decay,
            )
            if cell_trainable else None
        )

    image_logits = image_probe.predict_logits(dino_np, device) / tau_visual
    image_margin = margin_uncertainty_from_logits(image_logits)
    del image_probe

    if cell_probe is None:
        visual_weights = image_margin
        cell_weights = reliability * image_margin
    else:
        cell_logits = cell_probe.predict_logits(cell_np, device) / tau_cell
        cell_margin = margin_uncertainty_from_logits(cell_logits)
        if uncertainty_mode == "branch_margin":
            visual_weights = image_margin
            cell_weights = reliability * cell_margin
        else:
            disagreement = js_disagreement_from_logits(image_logits, cell_logits)
            visual_weights = (
                reliability * disagreement
                + (1.0 - reliability) * image_margin
            )
            cell_weights = reliability * disagreement
            diagnostics["mean_disagreement"] = float(
                disagreement[valid].mean() if valid.any() else 0.0
            )
        del cell_probe, cell_logits

    diagnostics["tau_visual"] = float(tau_visual)
    diagnostics["tau_cell"] = float(tau_cell)
    del image_logits
    clear_memory()
    return (
        np.clip(visual_weights, 0.0, 1.0).astype(np.float32),
        np.clip(cell_weights, 0.0, 1.0).astype(np.float32),
        diagnostics,
    )


def dual_view_uherding_sampling(**kwargs) -> List[int]:
    dino_raw = np.asarray(kwargs["image_embeddings"], dtype=np.float32)
    cell_raw = np.asarray(kwargs["nucleus_embeddings"], dtype=np.float32)
    reliability = np.clip(
        np.asarray(kwargs["nucleus_reliability"], dtype=np.float32), 0.0, 1.0
    )
    labels = np.asarray(kwargs["oracle_labels"])
    num_classes = int(kwargs["num_classes"])
    max_budget = int(kwargs["max_budget"])
    device = kwargs["device"]

    uncertainty_mode = kwargs.get("uncertainty_mode", "branch_margin")
    fusion_mode = kwargs.get("fusion_mode", "joint")
    coverage_mode = kwargs.get("coverage_mode", "dual")
    num_rounds = int(kwargs.get("num_rounds", 5))
    shortlist_multiplier = int(kwargs.get("shortlist_multiplier", 2))
    alpha = float(kwargs.get("alpha", 0.5))
    rrf_k = float(kwargs.get("rrf_k", 60.0))
    chunk_size = int(kwargs.get("chunk_size", 1000))
    n_sigma = int(kwargs.get("n_sigma", 2000))
    candidate_pool_size = kwargs.get("candidate_pool_size", 8000)
    probe_epochs = int(kwargs.get("probe_epochs", 50))
    probe_lr = float(kwargs.get("probe_lr", 1e-3))
    probe_weight_decay = float(kwargs.get("probe_weight_decay", 1e-4))
    consistency_weight = float(kwargs.get("consistency_weight", 0.0))
    consistency_mode = kwargs.get("consistency_mode", "symmetric_js")
    diag = bool(kwargs.get("diag", True))

    if uncertainty_mode not in VALID_UNCERTAINTIES:
        raise ValueError(
            f"uncertainty_mode must be one of {sorted(VALID_UNCERTAINTIES)}"
        )
    if fusion_mode not in VALID_FUSIONS:
        raise ValueError(f"fusion_mode must be one of {sorted(VALID_FUSIONS)}")
    if coverage_mode not in VALID_COVERAGE_MODES:
        raise ValueError(
            f"coverage_mode must be one of {sorted(VALID_COVERAGE_MODES)}"
        )
    if not (len(dino_raw) == len(cell_raw) == len(reliability) == len(labels)):
        raise ValueError("DINO, CellViT, reliability, and labels must align by patch")
    if dino_raw.ndim != 2 or cell_raw.ndim != 2:
        raise ValueError("image_embeddings and nucleus_embeddings must be 2-D")
    if shortlist_multiplier < 1 or chunk_size < 1:
        raise ValueError("shortlist_multiplier and chunk_size must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if rrf_k < 0.0 or consistency_weight < 0.0:
        raise ValueError("rrf_k and consistency_weight must be non-negative")
    if consistency_mode not in {
        "symmetric_js", "visual_teacher", "cell_teacher"
    }:
        raise ValueError(
            "consistency_mode must be symmetric_js, visual_teacher, or cell_teacher"
        )
    if candidate_pool_size is not None and int(candidate_pool_size) <= 0:
        candidate_pool_size = None

    n = len(dino_raw)
    budget = min(max_budget, n)
    rounds = max(1, min(num_rounds, budget))
    base, remainder = divmod(budget, rounds)
    round_sizes = [base + (1 if r < remainder else 0) for r in range(rounds)]
    valid_cell = reliability > 0.0

    dino_np = _normalize_np(dino_raw)
    dino_features = torch.as_tensor(
        dino_np, device=device, dtype=torch.float32
    )
    cell_features, missing_fraction = _build_coverage_features(
        dino_np, cell_raw, reliability, "cellvit", "mean", device
    )
    cell_np = cell_features.detach().cpu().numpy().astype(np.float32)
    dino_sigma = _adaptive_sigma(dino_features, n_ref=n_sigma)
    if int(valid_cell.sum()) >= 2:
        valid_cell_indices = torch.as_tensor(
            np.flatnonzero(valid_cell), device=device, dtype=torch.long
        )
        cell_sigma = _adaptive_sigma(
            cell_features[valid_cell_indices], n_ref=n_sigma
        )
        del valid_cell_indices
    else:
        # The cell branch will have no usable proposals; reuse a finite scale
        # for diagnostics without letting imputed missing rows define it.
        cell_sigma = dino_sigma

    selected: List[int] = []
    selected_set: set[int] = set()
    for round_index in tqdm(range(rounds), desc="DUAL-UH Rounds"):
        round_start = time.time()
        n_select = round_sizes[round_index]
        if round_index == 0:
            visual_weights_np = np.ones(n, dtype=np.float32)
            cell_weights_np = reliability.copy()
            uncertainty_diag = {
                "tau_visual": 1.0,
                "tau_cell": 1.0,
                "mean_disagreement": float("nan"),
            }
        else:
            visual_weights_np, cell_weights_np, uncertainty_diag = (
                _round_uncertainties(
                    dino_np, cell_np, reliability, labels, selected,
                    num_classes, probe_epochs, probe_lr,
                    probe_weight_decay=probe_weight_decay,
                    consistency_weight=consistency_weight,
                    consistency_mode=consistency_mode,
                    device=device,
                    uncertainty_mode=uncertainty_mode,
                )
            )

        visual_weights = torch.as_tensor(
            visual_weights_np, device=device, dtype=torch.float32
        )
        cell_weights = torch.as_tensor(
            cell_weights_np, device=device, dtype=torch.float32
        )
        dino_sigma = _labeled_positive_sigma(
            dino_features, selected, dino_sigma
        )
        valid_selected = [idx for idx in selected if valid_cell[idx]]
        cell_sigma = _labeled_positive_sigma(
            cell_features, valid_selected, cell_sigma
        )
        remaining = np.asarray(
            [idx for idx in range(n) if idx not in selected_set], dtype=np.int64
        )
        if candidate_pool_size is not None and len(remaining) > int(candidate_pool_size):
            candidate_pool = np.sort(
                np.random.choice(
                    remaining, int(candidate_pool_size), replace=False
                ).astype(np.int64)
            )
        else:
            candidate_pool = remaining

        relevant_indices = np.concatenate(
            [np.asarray(selected, dtype=np.int64), candidate_pool]
        )
        selected_positions = list(range(len(selected)))
        valid_selected_positions = [
            position for position, index in enumerate(selected)
            if valid_cell[index]
        ]
        candidate_positions = np.arange(
            len(selected), len(relevant_indices), dtype=np.int64
        )
        dino_kernel = _precompute_relevant_kernel(
            dino_features, relevant_indices, dino_sigma, chunk_size
        )
        dino_running = _initial_kernel_coverage(
            dino_kernel, selected_positions
        )
        visual_relevant_weights = visual_weights[
            torch.as_tensor(
                relevant_indices, device=device, dtype=torch.long
            )
        ]

        if coverage_mode == "dino":
            picks, _ = _kernel_greedy_order(
                dino_kernel, visual_relevant_weights, dino_running,
                candidate_positions, relevant_indices, n_select,
            )
            visual_shortlist: List[int] = picks
            cell_shortlist: List[int] = []
            cell_kernel = None
        else:
            cell_kernel = _precompute_relevant_kernel(
                cell_features, relevant_indices, cell_sigma, chunk_size
            )
            cell_running = _initial_kernel_coverage(
                cell_kernel, valid_selected_positions
            )
            cell_relevant_weights = cell_weights[
                torch.as_tensor(
                    relevant_indices, device=device, dtype=torch.long
                )
            ]
            shortlist_size = shortlist_multiplier * n_select
            visual_shortlist, visual_gains = _kernel_greedy_order(
                dino_kernel, visual_relevant_weights, dino_running,
                candidate_positions, relevant_indices, shortlist_size,
            )
            cell_candidate_positions = candidate_positions[
                valid_cell[relevant_indices[candidate_positions]]
            ]
            cell_shortlist, cell_gains = _kernel_greedy_order(
                cell_kernel, cell_relevant_weights, cell_running,
                cell_candidate_positions, relevant_indices, shortlist_size,
            )
            union = sorted(set(visual_shortlist) | set(cell_shortlist))
            if fusion_mode == "joint":
                picks = _kernel_joint_greedy(
                    dino_kernel, cell_kernel,
                    visual_relevant_weights, cell_relevant_weights,
                    dino_running, cell_running, relevant_indices,
                    union, valid_cell, n_select, alpha,
                )
            elif fusion_mode in {"visual_then_cell", "cell_then_visual"}:
                if fusion_mode == "visual_then_cell":
                    cascade_candidates = np.asarray([
                        int(np.flatnonzero(relevant_indices == idx)[0])
                        for idx in visual_shortlist if valid_cell[idx]
                    ],
                        dtype=np.int64,
                    )
                    picks, _ = _kernel_greedy_order(
                        cell_kernel, cell_relevant_weights, cell_running,
                        cascade_candidates, relevant_indices, n_select,
                    )
                    fallback_order = visual_shortlist
                else:
                    cascade_candidates = np.asarray([
                        int(np.flatnonzero(relevant_indices == idx)[0])
                        for idx in cell_shortlist
                    ], dtype=np.int64)
                    picks, _ = _kernel_greedy_order(
                        dino_kernel, visual_relevant_weights, dino_running,
                        cascade_candidates, relevant_indices, n_select,
                    )
                    fallback_order = cell_shortlist + visual_shortlist
                if len(picks) < n_select:
                    seen = set(picks)
                    for idx in fallback_order:
                        if idx not in seen:
                            picks.append(idx)
                            seen.add(idx)
                        if len(picks) == n_select:
                            break
            else:
                picks = _static_fusion(
                    visual_shortlist, cell_shortlist,
                    visual_gains, cell_gains,
                    n_select, fusion_mode, rrf_k,
                )

        if len(picks) < n_select:
            raise RuntimeError(
                f"Round {round_index} produced {len(picks)}/{n_select} items"
            )
        selected.extend(picks)
        selected_set.update(picks)

        if diag:
            disagreement_text = (
                "n/a" if np.isnan(uncertainty_diag["mean_disagreement"])
                else f"{uncertainty_diag['mean_disagreement']:.4f}"
            )
            print(
                f"[DUAL-UH b={budget} r={round_index}] "
                f"mode={coverage_mode}/{uncertainty_mode}/{fusion_mode} "
                f"pool={len(candidate_pool)} shortlists="
                f"{len(visual_shortlist)}/{len(cell_shortlist)} "
                f"picked={len(picks)} missing_cell={missing_fraction:.1%} "
                f"sigma={dino_sigma:.4f}/{cell_sigma:.4f} "
                f"tau={uncertainty_diag['tau_visual']:.1f}/"
                f"{uncertainty_diag['tau_cell']:.1f} "
                f"js={disagreement_text} "
                f"time={time.time() - round_start:.1f}s"
            )

        del visual_weights, cell_weights, dino_running, dino_kernel
        del visual_relevant_weights
        if cell_kernel is not None:
            del cell_kernel, cell_running, cell_relevant_weights
        clear_memory()

    del dino_features, cell_features
    clear_memory()
    return selected


@register_sampler("dual_view_uherding")
def _registered_dual_view_uherding(**kwargs) -> List[int]:
    return dual_view_uherding_sampling(**kwargs)


@register_sampler("disagreement_uherding")
def disagreement_uherding_sampling(**kwargs) -> List[int]:
    """DINO UCoverage control with calibrated cross-view JSD weights."""
    control_kwargs = dict(kwargs)
    control_kwargs["coverage_mode"] = "dino"
    control_kwargs["uncertainty_mode"] = "disagreement"
    return dual_view_uherding_sampling(**control_kwargs)
