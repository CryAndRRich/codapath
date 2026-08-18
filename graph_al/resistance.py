"""Effective resistance / commute-time distance on the AL graph.

Why this exists
---------------
SARGraphAL's acquisition is uncertainty-only (verified against the official
`jwcalder/MSTAR-Active-Learning` code: acquisition = uncertainty / VOpt / MC /
MCVOpt, no diversity or representativeness term anywhere). This module supplies
the missing coverage side in a way that is intrinsic to the SAME graph the
uncertainty is computed on, rather than bolting on a Euclidean kernel over raw
features the way `uncertainty_herding.py` does.

Effective resistance `R(i,j)` is the natural graph metric for this: it is small
when many short, high-weight paths connect `i` and `j` (they are redundant to
label together) and large when they are connected only through a bottleneck.
Commute time is `C(i,j) = vol(G) * R(i,j)` — the same quantity up to the global
constant `vol(G) = sum(W)`, so it induces an identical ranking and the same
greedy choices. Both names in the task description therefore reduce to one
implementation; `commute_time_to_set` is a thin, explicitly-scaled wrapper.

How it is computed
------------------
Exact `R(i,j) = (e_i - e_j)^T L^+ (e_i - e_j)` needs the pseudo-inverse of an
N x N Laplacian — impossible at N=100k. We use the Spielman-Srivastava
Johnson-Lindenstrauss sketch (Spielman & Srivastava, "Graph sparsification by
effective resistances", SICOMP 2011, Sec. 4):

    Z = Q W_e^{1/2} B L^+      with Q a (k x m) random +-1/sqrt(k) matrix

then `R(i,j) ~= ||Z_i - Z_j||^2` with relative error `eps` for
`k = O(log N / eps^2)`. Building `Z` costs exactly `k` linear solves against
`L` — the SAME Jacobi-preconditioned CG that `laplace.py` already runs 9x per
acquisition round — so one sketch (k ~ 100, computed once and cached for the
whole run) is cheaper than a couple of acquisition rounds.

`L` is singular (constant null vector on a connected graph), so every solve is
projected onto the mean-zero subspace before and after CG; this returns the
pseudo-inverse solution rather than a diverging one. Resistances only ever
appear as DIFFERENCES `Z_i - Z_j`, so the free additive constant per column
cancels and does not affect any distance.

Measured accuracy (N=2000 kNN graph, k=10, `min` resistance to 25 labeled
nodes, vs exact dense `pinv(L)`; reproduced by
`tests/test_graph_al_resistance.py`):

    n_sketch | median rel.err | Spearman vs exact
        25   |     0.304      |      0.533
        50   |     0.198      |      0.688
       100   |     0.139      |      0.794
       200   |     0.085      |      0.873
       400   |     0.045      |      0.926

The greedy consumes the RANKING, not the values, so Spearman is the column
that matters — hence the default `n_sketch=200` rather than the textbook 100.
The sketch is computed once per graph and reused for every round and every
budget, so its cost is amortised over the whole run; raise it if the coverage
term looks noisy.
"""

from typing import Optional

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla


def _incidence(W: sps.spmatrix):
    """Signed edge-incidence `B` (m x N) and edge weights `w` for the upper
    triangle of a symmetric `W`. `L = B^T diag(w) B`."""
    W = sps.triu(W, k=1).tocoo()
    m = W.nnz
    rows = np.repeat(np.arange(m), 2)
    cols = np.empty(2 * m, dtype=np.int64)
    cols[0::2] = W.row
    cols[1::2] = W.col
    vals = np.empty(2 * m, dtype=np.float64)
    vals[0::2] = 1.0
    vals[1::2] = -1.0
    B = sps.coo_matrix((vals, (rows, cols)), shape=(m, W.shape[0])).tocsr()
    return B, np.asarray(W.data, dtype=np.float64)


def _solve_laplacian(L: sps.spmatrix, rhs: np.ndarray, tol: float, maxiter: int):
    """Pseudo-inverse solve `L x = rhs` for a singular (connected) Laplacian.

    The constant vector spans `ker(L)`, so `rhs` is projected to mean zero
    (making the system consistent) and the solution is re-centred. Jacobi
    preconditioning matches `laplace.py::laplace_learning`.

    Returns `(x, converged)`. Unlike `laplace_learning`, a single
    non-converged solve is NOT fatal here — the sketch averages over
    `n_sketch` independent projections, so a few bad columns degrade accuracy
    rather than invalidating the result. The caller counts them and warns,
    because silently returning a garbage embedding would show up only as an
    inexplicably weak coverage term much later.
    """
    rhs = rhs - rhs.mean()
    diag = np.asarray(L.diagonal())
    precond = sps.diags(1.0 / np.sqrt(np.maximum(diag, 1e-12)))
    A = (precond @ L @ precond).tocsr()
    b = precond @ rhs
    try:
        sol, info = spla.cg(A, b, rtol=tol, atol=0.0, maxiter=maxiter)
    except TypeError:  # older scipy
        sol, info = spla.cg(A, b, tol=tol, maxiter=maxiter)
    x = precond @ sol
    return x - x.mean(), info == 0


def resistance_embedding(
    W: sps.spmatrix,
    n_sketch: int = 200,
    seed: int = 0,
    tol: float = 1e-6,
    maxiter: int = 1000,
    exact_below: int = 1500,
) -> np.ndarray:
    """Embedding `Z` (N, d) with `||Z_i - Z_j||^2 ~= R_eff(i,j)`.

    Compute it ONCE per graph and reuse it for every round and every budget —
    it depends only on `W`, never on labels.

    Parameters
    ----------
    W : (N, N) symmetric sparse weight matrix (e.g. `graph_al.graph.knn_graph`)
    n_sketch : number of JL projections = number of CG solves. Error decays as
        `1/sqrt(n_sketch)` (confirmed empirically, see the table above); the
        default 200 buys Spearman ~0.87 against exact resistances.
    exact_below : below this N, skip the sketch and use the exact dense
        pseudo-inverse (used by the tests, and cheap for small pools).
    """
    N = W.shape[0]
    if N <= exact_below:
        L = np.asarray(sps.csgraph.laplacian(W, normed=False).todense(), dtype=np.float64)
        Linv = np.linalg.pinv(L)
        # R(i,j) = Linv_ii + Linv_jj - 2 Linv_ij is reproduced exactly by the
        # embedding rows of any square root of the (PSD) centred Linv.
        vals, vecs = np.linalg.eigh(Linv)
        vals = np.clip(vals, 0.0, None)
        return (vecs * np.sqrt(vals)).astype(np.float64)

    L = sps.csgraph.laplacian(W, normed=False).tocsr()
    B, w_e = _incidence(W)
    m = len(w_e)

    rng = np.random.default_rng(seed)
    Z = np.empty((N, n_sketch), dtype=np.float64)
    n_failed = 0
    for j in range(n_sketch):
        q = rng.choice(np.array([-1.0, 1.0]), size=m) / np.sqrt(n_sketch)
        rhs = B.T @ (np.sqrt(w_e) * q)          # one row of Q W^{1/2} B
        Z[:, j], converged = _solve_laplacian(L, rhs, tol, maxiter)
        n_failed += not converged
    if n_failed:
        print(f"[resistance] WARNING: {n_failed}/{n_sketch} CG solves hit maxiter="
              f"{maxiter} without reaching tol={tol}. The resistance estimates are "
              f"correspondingly noisier; raise `maxiter` or loosen `tol`. A large "
              f"count usually means a badly conditioned graph (very small `k`, or "
              f"many near-isolated nodes).")
    return Z


def resistance_to_set(
    Z: np.ndarray,
    labeled_indices: np.ndarray,
    reduction: str = "min",
    chunk_size: int = 4096,
) -> np.ndarray:
    """Per-node effective resistance to the labeled set, from `Z`.

    `reduction="min"` gives `min_{s in S} R(i,s)` — the coreset/facility-location
    reading: HIGH means the node is far (in graph terms) from everything already
    labeled, i.e. poorly covered, so it is directly usable as a coverage score
    where larger = more worth acquiring. `"mean"` averages over `S` instead,
    which is smoother but lets one distant labeled point mask genuine local
    redundancy.
    """
    if reduction not in ("min", "mean"):
        raise ValueError(f"reduction must be 'min' or 'mean', got {reduction!r}")
    labeled_indices = np.asarray(labeled_indices, dtype=np.int64)
    if len(labeled_indices) == 0:
        raise ValueError("resistance_to_set needs at least one labeled index")

    ZL = Z[labeled_indices]                       # (m, d)
    zl_sq = (ZL ** 2).sum(axis=1)                 # (m,)
    out = np.empty(Z.shape[0], dtype=np.float64)
    for start in range(0, Z.shape[0], chunk_size):
        end = min(start + chunk_size, Z.shape[0])
        blk = Z[start:end]
        # ||z_i - z_s||^2 without materialising the difference tensor
        d2 = (blk ** 2).sum(axis=1)[:, None] + zl_sq[None, :] - 2.0 * (blk @ ZL.T)
        np.maximum(d2, 0.0, out=d2)
        out[start:end] = d2.min(axis=1) if reduction == "min" else d2.mean(axis=1)
    return out


def commute_time_to_set(
    W: sps.spmatrix,
    Z: np.ndarray,
    labeled_indices: np.ndarray,
    reduction: str = "min",
) -> np.ndarray:
    """Commute-time distance `C = vol(G) * R`, `vol(G) = sum(W)`.

    Provided because the task names commute time and effective resistance as
    alternatives; note the two differ only by this positive global constant, so
    any rank-based or argmax-based use of them is IDENTICAL. Prefer
    `resistance_to_set` unless an absolute scale is genuinely needed.
    """
    vol = float(W.sum())
    return vol * resistance_to_set(Z, labeled_indices, reduction=reduction)
