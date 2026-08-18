"""kNN similarity graph construction — self-tuning Gaussian kernel per
SARGraphAL (arXiv:2204.00005, `references/SARGraphAL.md`), reimplemented from
scratch in torch (chunked, GPU-friendly) instead of adding the `graphlearning`
PyPI dependency the official code
(`repos/MSTAR-Active-Learning/Python/active_learning.py`,
`CNNVAE_graph_learning.py`) calls into. This project's pool sizes (tens of
thousands, not millions) make brute-force chunked kNN tractable with the same
O(N²)-chunking pattern already used throughout `sampling/scalpel.py`
(`_greedy_coverage_batch`), so no FAISS/Annoy is needed.

Distance convention: L2-normalize features, then use the TRUE Euclidean
distance on the unit sphere, `sqrt(2 - 2*cos)` — NOT `1 - cos` (a common bug
in this codebase's history, see CLAUDE.md's `sqrt(2-2cos)` vs `1-cos` note in
the nucleus_coverage/uncertainty_herding fixes: `1-cos` is only half of the
squared distance, and squaring it again below would silently distort the
kernel's denominator). This also deviates from the official code's raw
Euclidean on the un-normalized 32-dim CNNVAE latent, kept for consistency
with the rest of this project's established convention (CLAUDE.md "Bài học
chung cho repo": always L2-normalize before treating a dot product as cosine).

Kernel formula — CORRECTED 2026-08-17 after re-reading the actual PDF
(`pdfs/2204.00005v1.pdf`, Section 4.4 "GRAPH CONSTRUCTIONS", not Sec 2.1 as
a previous, uncorroborated pass mistakenly claimed): the paper's OWN printed
formula is
`w_ij = exp(-4|xi-xj|²/d_k(xi)²)` (ASYMMETRIC, only the row point's own
k-th-neighbor radius), and the paper states it "symmetrized the weight
matrix by replacing W with W + W^T" (plain sum, no averaging). There is NO
paper-vs-code discrepancy in the kernel formula direction — a prior pass
through this file incorrectly asserted the paper describes a SYMMETRIC
formula (`d_k(xi)·d_k(xj)`) that the code then silently deviates from; that
claim does not survive a direct read of the PDF and has been removed. The
`graphlearning` v1.7.5 package's own default `kernel='gaussian'` (confirmed
by reading its source) matches the paper's asymmetric formula exactly; its
separate `kernel='symgaussian'` option (the literal `d_k(xi)·d_k(xj)`
formula) is a real, distinct choice available in that package, just not the
one SARGraphAL's paper or official scripts use — kept here as `kernel=
"symmetric"` because it is a legitimate documented alternative in the
software SARGraphAL depends on, not because the paper text calls for it.

One remaining, INCONSEQUENTIAL difference: the paper's stated symmetrization
is `W+W^T` (unnormalized), while this file and `graphlearning`'s actual code
both use `(W+W.T)/2` (or `.maximum` for the symmetric kernel). Uniformly
rescaling every edge weight by a constant factor does not change the
Laplace-learning solution (`L' = cL` for a rescaled `W' = cW` leaves
`L_uu'u = -L_ul'y` unchanged after dividing out `c`) and does not change
which point wins an argmax-based greedy selection (`max(c·a - c·b, 0) =
c·max(a-b,0)`, so relative ordering is preserved) — so this detail does not
affect any downstream use of `W` in this project. Keeping `/2` here bounds
weights to `(0,1]`, which matters for the DEUCE merge threshold `γ=1.0`
elsewhere in `graph_al/` (see `EXPERIMENT.md` Hướng 3 mục 7.6) — a deliberate
reason to keep it, not an oversight.
"""

from typing import Tuple

import numpy as np
import scipy.sparse as sps
import torch
import torch.nn.functional as F


def _pairwise_distance_chunk(row: torch.Tensor, all_feats: torch.Tensor) -> torch.Tensor:
    """True Euclidean distance on L2-normalized vectors: sqrt(2-2cos)."""
    sim = torch.matmul(row, all_feats.T)
    return torch.sqrt(torch.clamp(2.0 - 2.0 * sim, min=0.0))


def _knn_indices_distances(features: torch.Tensor, k: int, chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked brute-force kNN search (excludes self)."""
    N = features.shape[0]
    device = features.device
    knn_idx = torch.empty((N, k), dtype=torch.long, device=device)
    knn_dist = torch.empty((N, k), dtype=torch.float32, device=device)
    row_ids = torch.arange(N, device=device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        dist = _pairwise_distance_chunk(features[start:end], features)
        local_rows = torch.arange(end - start, device=device)
        dist[local_rows, row_ids[start:end]] = float("inf")  # exclude self
        vals, idx = torch.topk(dist, k, dim=1, largest=False)
        knn_idx[start:end] = idx
        knn_dist[start:end] = vals

    return knn_idx, knn_dist


def knn_graph(
    features: torch.Tensor,
    k: int = 20,
    kernel: str = "asymmetric",
    normalize: bool = True,
    chunk_size: int = 2000,
) -> sps.csr_matrix:
    """Builds a symmetric kNN weight matrix with the SARGraphAL self-tuning kernel.

    Parameters
    ----------
    features : (N, D) torch tensor, any device
    k : number of nearest neighbors (paper/official default = 20)
    kernel : "asymmetric" (official code's actual behavior, default) or
        "symmetric" (literal paper-text formula) — see module docstring.
    normalize : L2-normalize features first (project convention, default True)
    chunk_size : row-chunk size for the O(N²) distance computation

    Returns
    -------
    W : (N, N) scipy.sparse.csr_matrix, symmetric, zero diagonal
    """
    if kernel not in ("asymmetric", "symmetric"):
        raise ValueError(f"Unknown kernel={kernel!r}, expected 'asymmetric' or 'symmetric'")
    if features.shape[0] <= k:
        raise ValueError(f"Need more than k={k} points to build a kNN graph, got {features.shape[0]}")

    feats = F.normalize(features, p=2, dim=1) if normalize else features
    N = feats.shape[0]
    knn_idx, knn_dist = _knn_indices_distances(feats, k, chunk_size)
    d_k = knn_dist[:, -1].clamp(min=1e-8)  # distance to k-th neighbor, per node

    if kernel == "asymmetric":
        denom = d_k[:, None] ** 2
    else:
        denom = d_k[:, None] * d_k[knn_idx]

    weights = torch.exp(-4.0 * (knn_dist ** 2) / denom)

    rows = torch.arange(N, device=feats.device).repeat_interleave(k)
    cols = knn_idx.flatten()
    vals = weights.flatten()

    W = sps.coo_matrix(
        (vals.cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())),
        shape=(N, N),
    ).tocsr()

    if kernel == "asymmetric":
        W = (W + W.T) * 0.5
    else:
        # Already a symmetric formula; .maximum fills in edges found from
        # only one direction's kNN list (kNN is not symmetric even though
        # the weight formula is) without double-counting mutual ones.
        W = W.maximum(W.T)

    W.setdiag(0)
    W.eliminate_zeros()
    return W.tocsr()


# ---------------------------------------------------------------------------
# DEUCE's own graph construction (Option 2E) — UMAP-style GRAPHNORM +
# fuzzy-union symmetrization, verified against `pdfs/DEUCE_2502.00305.pdf`
# Sec 3.2.3 "Graph normalization"/"Symmetrization". This is what
# `sampling/graph_deuce.py` actually uses to build W_visual/W_cell before
# `graph_al.deuce_merge.merge_dual_neighbor_graphs` — NOT the SARGraphAL
# `knn_graph()` above (that one stays for whoever implements "Tổ hợp A",
# see EXPERIMENT.md Hướng 3).
# ---------------------------------------------------------------------------

def graphnorm_weights(
    knn_dist: torch.Tensor,
    max_bisect_iter: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """UMAP fuzzy-simplicial-set normalization (DEUCE Sec 3.2.3 "Graph
    normalization"). For each node i, finds tau_i > 0 solving
    `sum_j exp(-(d_ij - rho_i)/tau_i) = log2(k)`, rho_i = distance to the
    NEAREST neighbor (first column of `knn_dist`, which `_knn_indices_distances`
    already returns sorted ascending).

    The target function is monotonically increasing in tau_i, from the count
    of ties at rho_i (usually 1, since exp(0)=1 for the rho_i term itself) as
    tau_i->0+, up to k as tau_i->inf — a unique root exists whenever
    log2(k) lies in that range (true for any k > 2, i.e. every k used in this
    project). Solved via vectorized bisection across all N nodes at once (no
    closed form) — the same approach `umap-learn`'s own `smooth_knn_dist`
    uses, reimplemented here directly to avoid the dependency.

    Parameters
    ----------
    knn_dist : (N, k) distances to each node's k nearest neighbors, sorted
        ascending along dim=1 (as `_knn_indices_distances` returns).

    Returns
    -------
    rho : (N,) nearest-neighbor distance per node
    tau : (N,) solved normalization factor per node
    w_tilde : (N, k) directed fuzzy weights, `w_tilde[:, 0] == 1` always
        (the rho-distance neighbor always gets `exp(0) == 1`).
    """
    N, k = knn_dist.shape
    device = knn_dist.device
    rho = knn_dist[:, 0]
    target = float(np.log2(k))
    shifted = knn_dist - rho[:, None]  # (N, k); first column is always 0

    lo = torch.full((N,), 1e-6, device=device, dtype=torch.float32)
    hi = torch.full((N,), 1.0, device=device, dtype=torch.float32)

    def _f(tau: torch.Tensor) -> torch.Tensor:
        return torch.exp(-shifted / tau[:, None]).sum(dim=1)

    for _ in range(max_bisect_iter):
        too_low = _f(hi) < target
        if not bool(too_low.any()):
            break
        hi = torch.where(too_low, hi * 2.0, hi)

    for _ in range(max_bisect_iter):
        mid = 0.5 * (lo + hi)
        too_low = _f(mid) < target
        lo = torch.where(too_low, mid, lo)
        hi = torch.where(too_low, hi, mid)

    tau = 0.5 * (lo + hi)
    w_tilde = torch.exp(-shifted / tau[:, None])
    return rho, tau, w_tilde


def fuzzy_symmetrize(w_tilde: torch.Tensor, knn_idx: torch.Tensor, n_total: int) -> sps.csr_matrix:
    """DEUCE Sec 3.2.3 "Symmetrization": `W_sym = W + W^T - W ⊙ W^T` (fuzzy
    union of neighbors and reverse-neighbors, valid because `w_tilde ∈ (0,1]`
    can be read as fuzzy neighborhood memberships)."""
    k = w_tilde.shape[1]
    rows = torch.arange(n_total, device=w_tilde.device).repeat_interleave(k)
    cols = knn_idx.flatten()
    vals = w_tilde.flatten()

    W = sps.coo_matrix(
        (vals.cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())),
        shape=(n_total, n_total),
    ).tocsr()

    W_sym = (W + W.T - W.multiply(W.T)).tocsr()
    W_sym.setdiag(0)
    W_sym.eliminate_zeros()
    return W_sym


def knn_graph_umap(
    features: torch.Tensor,
    k: int = 20,
    normalize: bool = True,
    chunk_size: int = 2000,
) -> sps.csr_matrix:
    """kNN + GRAPHNORM + fuzzy-union symmetrization (DEUCE's own graph
    construction, Option 2E) — the pipeline `sampling/graph_deuce.py` uses
    for BOTH the visual and cell graphs before merging (see EXPERIMENT.md
    Hướng 3 mục 7.4)."""
    if features.shape[0] <= k:
        raise ValueError(f"Need more than k={k} points to build a kNN graph, got {features.shape[0]}")
    feats = F.normalize(features, p=2, dim=1) if normalize else features
    knn_idx, knn_dist = _knn_indices_distances(feats, k, chunk_size)
    _, _, w_tilde = graphnorm_weights(knn_dist)
    return fuzzy_symmetrize(w_tilde, knn_idx, feats.shape[0])


def knn_graph_partial(
    features_subset: torch.Tensor,
    global_indices: np.ndarray,
    n_total: int,
    k: int = 20,
    normalize: bool = True,
    chunk_size: int = 2000,
) -> sps.csr_matrix:
    """`knn_graph_umap` restricted to `features_subset`, scattered into a
    full `(n_total, n_total)` matrix at `global_indices` — nodes OUTSIDE
    `global_indices` are isolated (all-zero row/col). Used for the cell
    graph: patches with no detected nucleus (`reliability == 0`) get no cell
    embedding and thus no cell-graph edges at all — after
    `deuce_merge.merge_dual_neighbor_graphs`, every edge touching such a
    patch automatically falls back to single-neighbor (visual-only) weight,
    with no special-casing needed in the merge itself (see EXPERIMENT.md
    Hướng 3 mục 2, decision (i))."""
    n_sub = features_subset.shape[0]
    if n_sub <= k:
        raise ValueError(
            f"Need more than k={k} points in the reliable subset to build a kNN graph, got {n_sub}"
        )
    W_sub = knn_graph_umap(features_subset, k=k, normalize=normalize, chunk_size=chunk_size)

    global_indices = np.asarray(global_indices)
    W_sub_coo = W_sub.tocoo()
    global_rows = global_indices[W_sub_coo.row]
    global_cols = global_indices[W_sub_coo.col]
    W_full = sps.coo_matrix(
        (W_sub_coo.data, (global_rows, global_cols)),
        shape=(n_total, n_total),
    ).tocsr()
    W_full.eliminate_zeros()
    return W_full