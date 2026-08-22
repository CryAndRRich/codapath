"""Gaussian coverage kernel and greedy weighted facility location.

Every sampler in this project that mixes coverage with a per-point weight
shares this module: the baseline `uncertainty_herding` and the main method
`scalpel`. Keeping one implementation avoids the class of bug this codebase
already hit twice — a kernel fed rows that were not L2-normalized, and a
bandwidth passed as a squared distance.

Two invariants every caller must respect:

1. **Rows must be L2-normalized.** `gaussian_kernel` reads `A @ B.T` AS a
   cosine similarity. A non-unit row silently makes `1 - cos` negative, which
   clamps the kernel to 1 and turns coverage into a constant.
2. **`sigma` is a distance, never a squared distance.** The convention here is
   Euclidean `||u - v|| = sqrt(2 - 2cos)`. Note `1 - cos == ||u - v||^2 / 2`,
   so passing `1 - cos` as a bandwidth over-sharpens the exponent and zeroes
   out every marginal gain.
"""

from typing import List, Optional, Sequence

import numpy as np
import torch

from utils.runtime import clear_memory


def minmax(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. A constant input maps to all-zeros, which makes any
    downstream product degenerate — callers that multiply must check for it."""
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Rank-normalize to [0, 1], ties broken by index order.

    Preferred over `minmax` for probe-derived uncertainty: those distributions
    saturate near their maximum, where minmax amplifies noise in the tiny
    remaining spread instead of preserving the real ordering.
    """
    order = np.argsort(np.argsort(values))
    return (order / max(1, len(values) - 1)).astype(np.float32)


def bootstrap_sigma(features: torch.Tensor, n_ref: int = 2000) -> float:
    """Median pairwise distance over a pool subsample.

    Only for round 1, where no labeled set exists yet and `labeled_min_sigma`
    is undefined. Uncertainty Herding does not define this case either.
    """
    num_samples = features.shape[0]
    n_ref = min(n_ref, num_samples)
    index = np.random.choice(num_samples, n_ref, replace=False)
    reference = features[index]
    similarity = torch.matmul(reference, reference.T)
    distance = torch.sqrt(torch.clamp(2.0 - 2.0 * similarity, min=0.0))
    distance.fill_diagonal_(float("inf"))
    lower = torch.tril_indices(n_ref, n_ref, offset=-1)
    return max(distance[lower[0], lower[1]].median().item(), 1e-4)


def labeled_min_sigma(
    features: torch.Tensor,
    selected: Sequence[int],
    previous_sigma: float,
) -> float:
    """Uncertainty Herding radius adaptation: the minimum genuinely-positive
    pairwise distance over the CURRENT labeled set, recomputed every round.

    This is the mechanism behind the paper's Proposition 4: as the labeled set
    grows, sigma shrinks, and the objective drifts from coverage-like toward
    uncertainty-like. A sampler that fixes sigma for the whole run loses that
    property and stops being budget-adaptive.

    Near-duplicate labeled points are skipped rather than allowed to floor
    sigma: a single (near-)identical pair would collapse the kernel into an
    indicator-of-duplicates for every other pair.
    """
    if len(selected) < 2:
        return previous_sigma
    index = torch.as_tensor(selected, device=features.device, dtype=torch.long)
    labeled = features[index]
    similarity = torch.matmul(labeled, labeled.T)
    distance = torch.sqrt(torch.clamp(2.0 - 2.0 * similarity, min=0.0))
    distance.fill_diagonal_(float("inf"))
    positive = distance[(distance > 1e-6) & torch.isfinite(distance)]
    if positive.numel() == 0:
        return previous_sigma
    return max(float(positive.min().item()), 1e-3)


def gaussian_kernel(rows: torch.Tensor, cols: torch.Tensor, sigma: float) -> torch.Tensor:
    """`k(u, v) = exp(-||u - v||^2 / (2 sigma^2))` up to the constant folded
    into sigma, evaluated for every (row, col) pair. Rows must be unit-norm."""
    cosine = torch.matmul(rows, cols.T)
    return torch.exp(-torch.clamp(1.0 - cosine, min=0.0) / (sigma ** 2))


def kernel_column(
    features: torch.Tensor,
    point: torch.Tensor,
    sigma: float,
    chunk_size: int,
) -> torch.Tensor:
    """`k_sigma(x_n, point)` for every n, chunked over the pool axis."""
    num_samples = features.shape[0]
    column = torch.empty(num_samples, device=features.device, dtype=torch.float32)
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        column[start:end] = gaussian_kernel(features[start:end], point, sigma).squeeze(1)
    return column


def running_max_coverage(
    features: torch.Tensor,
    selected: Sequence[int],
    sigma: float,
    chunk_size: int,
) -> torch.Tensor:
    """`K_n = max_{j in selected} k_sigma(x_n, x_j)` at the CURRENT sigma.

    Must be rebuilt whenever sigma changes. Carrying the previous round's
    running max forward mixes kernel values computed at different bandwidths:
    the older values came from a larger sigma, so they are larger, and
    subtracting them under the new sigma under-reports every marginal gain.
    Cost is O(N * |selected|), negligible beside the O(N^2) greedy step.
    """
    num_samples = features.shape[0]
    coverage = torch.zeros(num_samples, device=features.device, dtype=torch.float32)
    if len(selected) == 0:
        return coverage
    index = torch.as_tensor(selected, device=features.device, dtype=torch.long)
    labeled = features[index]
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        coverage[start:end] = gaussian_kernel(
            features[start:end], labeled, sigma
        ).max(dim=1).values
    del labeled
    clear_memory()
    return coverage


def marginal_gains(
    features: torch.Tensor,
    weights: torch.Tensor,
    coverage: torch.Tensor,
    sigma: float,
    candidate_indices: Optional[np.ndarray],
    chunk_size: int,
) -> np.ndarray:
    """`g(i) = sum_n weights_n * max(k_sigma(x_n, x_i) - K_n, 0)` for each
    candidate i. Both axes are chunked, so peak temporary memory is about
    `chunk_size ** 2` floats regardless of pool size.

    `candidate_indices` restricts the candidate axis; None means the full pool.
    """
    num_samples = features.shape[0]
    if candidate_indices is None:
        candidates = np.arange(num_samples, dtype=np.int64)
    else:
        candidates = np.asarray(candidate_indices, dtype=np.int64)
    if len(candidates) == 0:
        return np.empty(0, dtype=np.float32)

    gains = np.zeros(len(candidates), dtype=np.float32)
    for cand_start in range(0, len(candidates), chunk_size):
        cand_end = min(cand_start + chunk_size, len(candidates))
        index = torch.as_tensor(
            candidates[cand_start:cand_end], device=features.device, dtype=torch.long
        )
        candidate_features = features[index]
        score = torch.zeros(cand_end - cand_start, device=features.device, dtype=torch.float32)
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            kernel = gaussian_kernel(features[start:end], candidate_features, sigma)
            gain = torch.clamp(kernel - coverage[start:end].unsqueeze(1), min=0.0)
            score += (weights[start:end].unsqueeze(1) * gain).sum(dim=0)
            del kernel, gain
        gains[cand_start:cand_end] = score.detach().cpu().numpy().astype(np.float32)
        del index, candidate_features, score
        clear_memory()
    return gains


def greedy_weighted_coverage(
    features: torch.Tensor,
    weights: torch.Tensor,
    coverage: torch.Tensor,
    sigma: float,
    n_select: int,
    selected_set: set,
    chunk_size: int,
) -> List[int]:
    """Greedily pick `n_select` points maximising the weighted marginal gain.

    `coverage` is updated in place after every pick, exactly as in Uncertainty
    Herding's Algorithm 1. For fixed non-negative `weights` the objective is
    monotone submodular, so the greedy solution keeps the (1 - 1/e) guarantee.
    """
    num_samples = features.shape[0]
    picks: List[int] = []
    for _ in range(n_select):
        best_index, best_score = -1, -float("inf")
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            candidates = features[start:end]
            gains = torch.zeros(end - start, device=features.device, dtype=torch.float32)
            for target_start in range(0, num_samples, chunk_size):
                target_end = min(target_start + chunk_size, num_samples)
                kernel = gaussian_kernel(features[target_start:target_end], candidates, sigma)
                gain = torch.clamp(
                    kernel - coverage[target_start:target_end].unsqueeze(1), min=0.0
                )
                gains += (weights[target_start:target_end].unsqueeze(1) * gain).sum(dim=0)
                del kernel, gain
            for taken in selected_set:
                if start <= taken < end:
                    gains[taken - start] = -float("inf")
            local_best = int(torch.argmax(gains).item())
            if gains[local_best].item() > best_score:
                best_score = gains[local_best].item()
                best_index = start + local_best
            del candidates, gains
            clear_memory()

        if best_index < 0 or best_index in selected_set:
            break
        picks.append(best_index)
        selected_set.add(best_index)
        column = kernel_column(features, features[best_index].unsqueeze(0), sigma, chunk_size)
        coverage.copy_(torch.maximum(coverage, column))
        del column
        clear_memory()
    return picks
