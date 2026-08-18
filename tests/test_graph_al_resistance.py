"""Sketch-vs-exact accuracy for `graph_al/resistance.py`.

The greedy consumes the RANKING of resistances, so Spearman against the exact
dense `pinv(L)` is the number that matters, not the raw relative error.
"""
import numpy as np
import scipy.sparse as sps

from graph_al.resistance import (
    commute_time_to_set,
    resistance_embedding,
    resistance_to_set,
)


def _knn_graph(N=600, k=10, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    D = np.sqrt(np.maximum(2 - 2 * X @ X.T, 0))
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    dk = np.take_along_axis(D, idx, 1)[:, -1:]
    w = np.exp(-4 * np.take_along_axis(D, idx, 1) ** 2 / dk ** 2)
    W = sps.coo_matrix(
        (w.ravel(), (np.repeat(np.arange(N), k), idx.ravel())), shape=(N, N)
    ).tocsr()
    W = (W + W.T) * 0.5
    W.setdiag(0)
    W.eliminate_zeros()
    return W.tocsr()


def _exact_resistance(W):
    L = np.asarray(sps.csgraph.laplacian(W, normed=False).todense())
    P = np.linalg.pinv(L)
    diag = np.diag(P)
    return diag[:, None] + diag[None, :] - 2 * P


def _spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def test_exact_branch_matches_pinv():
    W = _knn_graph()
    labeled = np.arange(0, 600, 40)
    expected = _exact_resistance(W)[:, labeled].min(axis=1)
    Z = resistance_embedding(W, exact_below=10_000)
    assert np.allclose(resistance_to_set(Z, labeled), expected, atol=1e-8)


def test_sketch_ranks_agree_with_exact():
    W = _knn_graph()
    labeled = np.arange(0, 600, 40)
    expected = _exact_resistance(W)[:, labeled].min(axis=1)
    Z = resistance_embedding(W, n_sketch=200, seed=0, exact_below=0)
    assert _spearman(resistance_to_set(Z, labeled), expected) > 0.75


def test_commute_time_is_resistance_times_volume():
    W = _knn_graph()
    labeled = np.arange(0, 600, 40)
    Z = resistance_embedding(W, exact_below=10_000)
    assert np.allclose(
        commute_time_to_set(W, Z, labeled), W.sum() * resistance_to_set(Z, labeled)
    )


def test_reduction_guard():
    W = _knn_graph()
    Z = resistance_embedding(W, exact_below=10_000)
    try:
        resistance_to_set(Z, np.array([0, 1]), reduction="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown reduction")
