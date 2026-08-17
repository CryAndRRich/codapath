import numpy as np
import scipy.sparse as sps
import torch

from graph_al.deuce_merge import merge_dual_neighbor_graphs
from graph_al.graph import (
    _knn_indices_distances,
    fuzzy_symmetrize,
    graphnorm_weights,
    knn_graph_partial,
    knn_graph_umap,
)


def _three_tight_clusters(seed: int = 0, per_cluster: int = 12, dim: int = 8):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, dim)) * 5.0
    points = np.concatenate([
        centers[c] + 0.05 * rng.normal(size=(per_cluster, dim))
        for c in range(3)
    ], axis=0)
    labels = np.repeat(np.arange(3), per_cluster)
    return torch.as_tensor(points, dtype=torch.float32), labels


def test_graphnorm_weights_satisfies_target_equation():
    features, _ = _three_tight_clusters()
    feats = torch.nn.functional.normalize(features, p=2, dim=1)
    knn_idx, knn_dist = _knn_indices_distances(feats, k=5, chunk_size=7)

    rho, tau, w_tilde = graphnorm_weights(knn_dist)

    assert torch.all(w_tilde[:, 0] > 0.999)  # exp(0) at the nearest-neighbor column
    achieved = w_tilde.sum(dim=1)
    target = np.log2(5)
    assert torch.allclose(achieved, torch.full_like(achieved, target), atol=1e-3)
    assert torch.all(tau > 0)
    assert torch.equal(rho, knn_dist[:, 0])


def test_fuzzy_symmetrize_is_symmetric_and_bounded():
    features, _ = _three_tight_clusters()
    feats = torch.nn.functional.normalize(features, p=2, dim=1)
    knn_idx, knn_dist = _knn_indices_distances(feats, k=5, chunk_size=7)
    _, _, w_tilde = graphnorm_weights(knn_dist)

    W_sym = fuzzy_symmetrize(w_tilde, knn_idx, feats.shape[0])

    np.testing.assert_allclose(W_sym.toarray(), W_sym.T.toarray(), atol=1e-6)
    assert W_sym.min() >= 0.0
    assert W_sym.max() <= 1.0 + 1e-6
    assert np.all(W_sym.diagonal() == 0.0)


def test_knn_graph_umap_within_cluster_dominates_cross_cluster():
    features, labels = _three_tight_clusters()
    W = knn_graph_umap(features, k=5, chunk_size=7).toarray()

    within = W[np.ix_(labels == 0, labels == 0)]
    within = within[within > 0]
    cross = W[np.ix_(labels == 0, labels != 0)]

    assert within.mean() > cross.mean() * 5


def test_knn_graph_partial_isolates_excluded_nodes():
    features, labels = _three_tight_clusters(per_cluster=15)
    n_total = features.shape[0]
    reliable_idx = np.where(labels != 2)[0]  # drop the 3rd cluster entirely
    subset = features[reliable_idx]

    W_full = knn_graph_partial(subset, reliable_idx, n_total, k=5, chunk_size=7)

    excluded_idx = np.where(labels == 2)[0]
    W_dense = W_full.toarray()
    assert np.all(W_dense[excluded_idx, :] == 0.0)
    assert np.all(W_dense[:, excluded_idx] == 0.0)
    # reliable block should still have real edges
    assert W_dense[np.ix_(reliable_idx, reliable_idx)].sum() > 0


def test_merge_dual_neighbor_graphs_matches_hand_computed_formula():
    # 3x3 toy graphs, hand-picked overlapping/non-overlapping edges.
    W1 = sps.csr_matrix(np.array([
        [0.0, 0.4, 0.0],
        [0.4, 0.0, 0.6],
        [0.0, 0.6, 0.0],
    ]))
    W2 = sps.csr_matrix(np.array([
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]))

    W_dual = merge_dual_neighbor_graphs(W1, W2, gamma=1.0).toarray()

    # edge (0,1): present in both -> 0.4*0.5 + 1.0
    assert np.isclose(W_dual[0, 1], 0.4 * 0.5 + 1.0)
    assert np.isclose(W_dual[1, 0], 0.4 * 0.5 + 1.0)
    # edge (1,2): present only in W1 -> keep 0.6
    assert np.isclose(W_dual[1, 2], 0.6)
    assert np.isclose(W_dual[2, 1], 0.6)
    # edge (0,2): absent in both -> 0
    assert W_dual[0, 2] == 0.0


def test_merge_dual_neighbor_graphs_never_double_counts_mutual_edges():
    W1 = sps.csr_matrix(np.array([[0.0, 0.3], [0.3, 0.0]]))
    W2 = sps.csr_matrix(np.array([[0.0, 0.7], [0.7, 0.0]]))
    W_dual = merge_dual_neighbor_graphs(W1, W2, gamma=1.0).toarray()
    expected = 0.3 * 0.7 + 1.0
    assert np.isclose(W_dual[0, 1], expected)
    assert np.isclose(W_dual[1, 0], expected)
