"""Sparse-kernel analogue of `sampling/uncertainty_herding.py`'s per-point
greedy weighted-coverage selection (Definition 5, arXiv:2412.20644), for use
when the coverage kernel is a sparse graph (e.g. `W_dual` from
`graph_al.deuce_merge`) instead of a dense Gaussian kernel on raw features.

Mirrors `uncertainty_herding.py` lines 186-221 EXACTLY in loop structure —
select 1 point, re-evaluate every remaining candidate, update the running-max
coverage, repeat — rather than the cheaper approximate-batch discount used by
the `laplace_margin` acquisition variant. See EXPERIMENT.md Hướng 3 mục
7.10.2 for why this per-point fidelity (not the cheaper approximation)
matters for the `uherding_swap_coverage` variant specifically: it is the one
variant that claims to keep UHerding's coverage mechanism UNCHANGED, and that
mechanism IS the per-point loop, not a batch approximation of it.
"""

from typing import List, Set

import scipy.sparse as sps
import torch


def greedy_coverage_sparse(
    W: sps.spmatrix,
    U: torch.Tensor,
    n_select: int,
    selected_set: Set[int],
    device: torch.device,
) -> List[int]:
    """Definition-5 weighted greedy coverage with a SPARSE kernel `W`.

    `score(x) = sum_{n : W[n,x] > 0} U(n) * max(W[n,x] - K_n, 0)`, computed
    for every candidate `x` simultaneously via scatter-add over `W`'s nonzero
    edges (nnz ~= N*k, not N^2) each step; `K_n` (running max coverage each
    pool point receives from the picks so far) updates after EVERY single
    pick, matching the per-point loop this mirrors.

    Parameters
    ----------
    W : (N, N) scipy sparse SYMMETRIC weight matrix
    U : (N,) torch tensor, per-point uncertainty weight (constant `ones(N)`
        for a cold-start round with no labels yet — matches
        `uncertainty_herding.py`'s own round-0 behavior)
    n_select : how many NEW points to pick
    selected_set : mutated in place — already-selected indices are excluded
        from candidacy and the newly picked ones are added
    device : torch device for the score/K_n tensors

    Returns
    -------
    picks : list of `n_select` newly selected indices (order = selection order)
    """
    N = W.shape[0]
    W_csc = W.tocsc()  # column slicing (per-candidate column) is efficient in CSC
    W_coo = W.tocoo()

    rows_t = torch.as_tensor(W_coo.row, device=device, dtype=torch.long)
    cols_t = torch.as_tensor(W_coo.col, device=device, dtype=torch.long)
    vals_t = torch.as_tensor(W_coo.data, device=device, dtype=torch.float32)

    K_n = torch.zeros(N, device=device, dtype=torch.float32)
    picks: List[int] = []

    for _ in range(n_select):
        gain_edges = torch.clamp(vals_t - K_n[rows_t], min=0.0) * U[rows_t]
        scores = torch.zeros(N, device=device, dtype=torch.float32)
        scores.scatter_add_(0, cols_t, gain_edges)

        if selected_set:
            excluded = torch.tensor(list(selected_set), device=device, dtype=torch.long)
            scores[excluded] = -float("inf")

        best_idx = int(torch.argmax(scores).item())
        picks.append(best_idx)
        selected_set.add(best_idx)

        col_start, col_end = W_csc.indptr[best_idx], W_csc.indptr[best_idx + 1]
        col_rows = W_csc.indices[col_start:col_end]
        col_vals = W_csc.data[col_start:col_end]
        if len(col_rows) > 0:
            col_rows_t = torch.as_tensor(col_rows, device=device, dtype=torch.long)
            col_vals_t = torch.as_tensor(col_vals, device=device, dtype=torch.float32)
            K_n[col_rows_t] = torch.maximum(K_n[col_rows_t], col_vals_t)

    return picks
