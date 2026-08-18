import numpy as np
import pytest
import scipy.sparse as sps
from scipy.spatial.distance import cdist

from graph_al.deuce_acquisition import (
    deuce_native_round,
    fps_on_graph,
    hdbscan_propagate_uncertainty,
    similarity_to_distance,
)


def _two_cluster_graph(seed=0, n_per=30, k=6):
    """2 well-separated clusters bridged by a few points, so the resulting
    kNN graph is a SINGLE connected component (required by
    sklearn HDBSCAN's sparse-precomputed path — see module docstring)."""
    rng = np.random.default_rng(seed)
    c1 = rng.normal(size=(n_per, 2)) * 0.4 + np.array([0.0, 0.0])
    c2 = rng.normal(size=(n_per, 2)) * 0.4 + np.array([2.2, 2.2])
    bridge = np.array([[0.9, 0.9], [1.1, 1.1], [1.3, 1.3]])
    X = np.vstack([c1, c2, bridge])
    n = X.shape[0]
    dist = cdist(X, X)
    rows, cols, vals = [], [], []
    for i in range(n):
        order = np.argsort(dist[i])
        order = order[order != i][:k]
        for j in order:
            w = float(np.exp(-dist[i, j]))
            rows.append(i)
            cols.append(j)
            vals.append(w)
    W = sps.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)
    return W, n_per


def test_similarity_to_distance_is_monotonic_decreasing():
    W = sps.csr_matrix(np.array([[0, 0.5, 0], [0.5, 0, 0.25], [0, 0.25, 0]]))
    D = similarity_to_distance(W)
    assert np.allclose(D.data, 1.0 / W.data)


def test_hdbscan_propagate_uncertainty_amplifies_high_uncertainty_cluster():
    W, n_per = _two_cluster_graph()
    n = W.shape[0]
    u = np.random.default_rng(1).random(n) * 0.2
    u[n_per:2 * n_per] += 0.6  # boost cluster 2's raw uncertainty

    u_tilde = hdbscan_propagate_uncertainty(W, u, min_cluster_size=5)
    assert u_tilde.shape == (n,)
    assert not np.isnan(u_tilde).any()
    assert u_tilde[n_per:2 * n_per].mean() > u_tilde[:n_per].mean()
    # propagation only ADDS to the raw signal, never below it
    assert np.all(u_tilde >= u - 1e-9)


def test_hdbscan_propagate_uncertainty_handles_disconnected_graph():
    """sklearn's HDBSCAN sparse-precomputed path raises if the graph has
    >1 connected component (verified empirically 2026-08-18) — this must be
    handled gracefully (largest component clustered, rest treated as noise),
    not crash the whole acquisition round."""
    rng = np.random.default_rng(2)
    n_per = 15
    c1 = rng.normal(size=(n_per, 2)) * 0.1
    c2 = rng.normal(size=(n_per, 2)) * 0.1 + np.array([50.0, 50.0])
    X = np.vstack([c1, c2])
    n = X.shape[0]
    dist = cdist(X, X)
    k = 4
    rows, cols, vals = [], [], []
    for i in range(n):
        order = np.argsort(dist[i])
        order = order[order != i][:k]
        for j in order:
            rows.append(i)
            cols.append(j)
            vals.append(float(np.exp(-dist[i, j])))
    W = sps.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)
    n_comp, _ = sps.csgraph.connected_components(W, directed=False)
    assert n_comp > 1  # sanity: this fixture really is disconnected

    u = rng.random(n)
    u_tilde = hdbscan_propagate_uncertainty(W, u, min_cluster_size=3)
    assert u_tilde.shape == (n,)
    assert not np.isnan(u_tilde).any()


def test_fps_on_graph_returns_unique_picks_and_excludes():
    W, n_per = _two_cluster_graph()
    dist_graph = similarity_to_distance(W)
    excluded = {0, 1, 2}
    picks = fps_on_graph(dist_graph, start_idx=5, n_select=10, excluded=excluded)
    assert len(picks) == len(set(picks))
    assert not (set(picks) & excluded)
    assert 5 in picks  # start_idx is always the first pick


def test_deuce_native_round_picks_unique_and_excludes_selected():
    W, n_per = _two_cluster_graph()
    n = W.shape[0]
    u = np.random.default_rng(3).random(n)
    selected_set = {0, 1}
    picks = deuce_native_round(
        W, u, min_cluster_size=5, fps_starts=3, n_select=8, selected_set=selected_set,
    )
    assert len(picks) == 8
    assert len(set(picks)) == 8
    assert not (set(picks) & selected_set)


def test_deuce_native_round_second_call_excludes_first():
    W, n_per = _two_cluster_graph()
    n = W.shape[0]
    u = np.random.default_rng(4).random(n)
    selected_set = set()
    picks1 = deuce_native_round(
        W, u, min_cluster_size=5, fps_starts=3, n_select=6, selected_set=selected_set,
    )
    selected_set.update(picks1)
    picks2 = deuce_native_round(
        W, u, min_cluster_size=5, fps_starts=3, n_select=6, selected_set=selected_set,
    )
    assert not (set(picks1) & set(picks2))


def test_deuce_native_round_returns_empty_when_pool_exhausted():
    W, n_per = _two_cluster_graph()
    n = W.shape[0]
    u = np.random.default_rng(5).random(n)
    selected_set = set(range(n))  # everything already labeled
    picks = deuce_native_round(
        W, u, min_cluster_size=5, fps_starts=3, n_select=5, selected_set=selected_set,
    )
    assert picks == []
