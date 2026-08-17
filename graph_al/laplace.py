"""Laplace learning (harmonic function graph SSL), Zhu-Ghahramani-Lafferty 2003
— reimplemented directly from `graphlearning` v1.7.5's `ssl.laplace._fit`
(the exact class `repos/MSTAR-Active-Learning/Python/active_learning.py` calls
via `gl.ssl.laplace(W).fit(...)`, at its defaults: `normalization=
'combinatorial'`, `reweighting='none'`, `order=1`, `tau=0`) using only scipy,
no extra dependency.

Solves `L_uu @ u_u = -L_ul @ y_l` for the harmonic extension `u` on unlabeled
nodes, `L = D - W` (combinatorial graph Laplacian), Jacobi-preconditioned CG —
line-for-line the same linear system `graphlearning` builds. `u` is NOT a
probability distribution: labeled rows are the exact one-hot label vector,
solved rows are the free real-valued harmonic extension (can go negative or
exceed 1) — `argmax` still gives the predicted class, and `laplace_margin`
below reproduces SARGraphAL's own default "Uncertainty Sampling" acquisition
(`references/SARGraphAL.md`, `active_learning.py::acquisition_function`,
`uncertainty_method="smallest_margin"`).

Caveat: this re-solves the full linear system from scratch on every call (as
the official code itself does, once per single-point sequential AL pick — see
`active_learning.py::active_learning_loop`). Conjugate gradient with Jacobi
preconditioning is what `graphlearning` itself uses, so cost should scale
similarly, but N here (PathMNIST/HistoSet/SkinTissue pools) can be far larger
than MSTAR's few thousand points — profile before running many sequential
iterations on a large pool.
"""

from typing import Optional

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    onehot = np.zeros((len(labels), num_classes), dtype=np.float64)
    onehot[np.arange(len(labels)), labels] = 1.0
    return onehot


def _cg_solve(A: sps.spmatrix, b: np.ndarray, tol: float):
    try:
        return spla.cg(A, b, rtol=tol, atol=0.0)
    except TypeError:
        # older scipy: `cg` takes `tol`, not `rtol`/`atol`
        return spla.cg(A, b, tol=tol)


def laplace_learning(
    W: sps.spmatrix,
    labeled_indices: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    tol: float = 1e-5,
) -> np.ndarray:
    """Harmonic function extension of `labels[labeled_indices]` over the full graph `W`.

    Parameters
    ----------
    W : (N, N) scipy sparse weight matrix (e.g. from `graph_al.graph.knn_graph`)
    labeled_indices : (m,) int array, indices of labeled nodes
    labels : (m,) int array, class label per `labeled_indices[i]` (0..num_classes-1)
    num_classes : total number of classes
    tol : conjugate gradient tolerance

    Returns
    -------
    u : (N, num_classes) float64 array — exact one-hot at `labeled_indices`,
        solved harmonic extension elsewhere.
    """
    N = W.shape[0]
    labeled_indices = np.asarray(labeled_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labeled_indices) < 1:
        raise ValueError("laplace_learning needs at least 1 labeled point")
    if len(np.unique(labels)) < 2:
        raise ValueError("laplace_learning needs at least 2 distinct labels among labeled_indices")

    L = sps.csgraph.laplacian(W, normed=False).tocsc()

    unlabeled_mask = np.ones(N, dtype=bool)
    unlabeled_mask[labeled_indices] = False
    unlabeled_indices = np.where(unlabeled_mask)[0]

    F_onehot = _one_hot(labels, num_classes)

    b_full = -(L[:, labeled_indices] @ F_onehot)
    b = b_full[unlabeled_indices, :]
    A = L[unlabeled_indices, :][:, unlabeled_indices].tocsr()

    diag = np.asarray(A.diagonal())
    precond = sps.diags(1.0 / np.sqrt(diag + 1e-10))
    A_precond = (precond @ A @ precond).tocsr()
    b_precond = precond @ b

    v = np.zeros_like(b)
    for c in range(num_classes):
        sol, info = _cg_solve(A_precond, np.asarray(b_precond[:, c]).ravel(), tol)
        if info != 0:
            raise RuntimeError(f"Conjugate gradient did not converge for class {c} (info={info})")
        v[:, c] = sol
    v = precond @ v

    u = np.zeros((N, num_classes), dtype=np.float64)
    u[unlabeled_indices, :] = v
    u[labeled_indices, :] = F_onehot
    return u


def laplace_margin(u: np.ndarray, candidate_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """`1 - (top1 - top2)` of `u` — SARGraphAL's default `smallest_margin` acquisition."""
    rows = u if candidate_indices is None else u[candidate_indices]
    sorted_rows = np.sort(rows, axis=1)
    return (1.0 - (sorted_rows[:, -1] - sorted_rows[:, -2])).astype(np.float32)


def personalized_pagerank(
    W: sps.spmatrix,
    labeled_indices: np.ndarray,
    damping: float = 0.85,
    tol: float = 1e-6,
) -> np.ndarray:
    """Personalized PageRank restarting uniformly on `labeled_indices` — the
    coverage signal for the `laplace_plus_ppr` acquisition variant (Option
    3E, EXPERIMENT.md Hướng 3 mục 7.10.3). NOT from SARGraphAL or DEUCE
    directly; PPR itself is standard, but the symmetric reformulation below
    is what makes it solvable with the SAME `_cg_solve` used by
    `laplace_learning`, avoiding a second, non-symmetric linear solver.

    Standard personalized PageRank solves
        pi = damping * P^T @ pi + (1 - damping) * r,   P = D^-1 @ W (row-stochastic),
    which is NOT symmetric even when `W` is. Substituting `q = D^-1/2 @ pi`
    and using `P^T = W @ D^-1` (since `W` is symmetric) gives
        (I - damping * S) @ q = (1 - damping) * D^-1/2 @ r,   S = D^-1/2 @ W @ D^-1/2,
    where `S` (symmetric normalized adjacency) has eigenvalues in `[-1, 1]`,
    so `(I - damping*S)` is symmetric POSITIVE DEFINITE for any
    `damping < 1` (every eigenvalue `1 - damping*lambda > 0`) — solvable by
    CG directly, with `pi = D^1/2 @ q` recovered at the end.

    Parameters
    ----------
    W : (N, N) scipy sparse SYMMETRIC weight matrix (e.g. `W_dual` from
        `graph_al.deuce_merge.merge_dual_neighbor_graphs`)
    labeled_indices : (m,) int array, restart distribution is uniform over these
    damping : teleport probability (DEUCE-unrelated default 0.85, standard PageRank value)
    tol : conjugate gradient tolerance

    Returns
    -------
    pi : (N,) float64 array, personalized PageRank score per node. LOW pi
        means far/hard-to-reach from `labeled_indices` under a random walk —
        callers turn this into a "coverage" signal via e.g.
        `rank_normalize(-pi)` (high value = worth labeling), NOT `pi` directly.
    """
    N = W.shape[0]
    labeled_indices = np.asarray(labeled_indices, dtype=np.int64)
    if len(labeled_indices) < 1:
        raise ValueError("personalized_pagerank needs at least 1 labeled point")
    if not (0.0 < damping < 1.0):
        raise ValueError(f"damping must be in (0,1), got {damping}")

    deg = np.asarray(W.sum(axis=1)).ravel()
    deg_safe = np.where(deg > 0, deg, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(deg_safe)

    D_inv_sqrt = sps.diags(d_inv_sqrt)
    S = (D_inv_sqrt @ W @ D_inv_sqrt).tocsr()

    r = np.zeros(N, dtype=np.float64)
    r[labeled_indices] = 1.0 / len(labeled_indices)

    A = (sps.eye(N, format="csr") - damping * S).tocsr()
    b = (1.0 - damping) * d_inv_sqrt * r

    q, info = _cg_solve(A, b, tol)
    if info != 0:
        raise RuntimeError(f"Conjugate gradient did not converge for personalized_pagerank (info={info})")

    return np.sqrt(deg_safe) * q
