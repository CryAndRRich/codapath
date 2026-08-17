import numpy as np
import scipy.sparse as sps
import torch

from graph_al.graph import knn_graph


def _three_tight_clusters(seed: int = 0, per_cluster: int = 12, dim: int = 8):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, dim)) * 5.0
    points = np.concatenate([
        centers[c] + 0.05 * rng.normal(size=(per_cluster, dim))
        for c in range(3)
    ], axis=0)
    labels = np.repeat(np.arange(3), per_cluster)
    return torch.as_tensor(points, dtype=torch.float32), labels


def test_knn_graph_is_symmetric_zero_diagonal_nonnegative():
    features, _ = _three_tight_clusters()
    for kernel in ("asymmetric", "symmetric"):
        W = knn_graph(features, k=5, kernel=kernel, chunk_size=7)
        assert isinstance(W, sps.csr_matrix)
        np.testing.assert_allclose(W.toarray(), W.T.toarray(), atol=1e-6)
        assert np.all(W.diagonal() == 0.0)
        assert W.min() >= 0.0
        assert W.max() <= 1.0 + 1e-6


def test_knn_graph_within_cluster_weight_dominates_cross_cluster():
    features, labels = _three_tight_clusters()
    W = knn_graph(features, k=5, kernel="symmetric", chunk_size=7).toarray()

    within = W[np.ix_(labels == 0, labels == 0)]
    within = within[within > 0]
    cross = W[np.ix_(labels == 0, labels != 0)]

    assert within.mean() > cross.mean() * 5


def test_knn_graph_rejects_bad_kernel_and_too_few_points():
    features, _ = _three_tight_clusters()
    try:
        knn_graph(features, k=5, kernel="bogus")
        assert False, "expected ValueError for unknown kernel"
    except ValueError:
        pass

    tiny = torch.randn(4, 8)
    try:
        knn_graph(tiny, k=5)
        assert False, "expected ValueError when k >= N"
    except ValueError:
        pass
