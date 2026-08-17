import numpy as np
import torch

from graph_al.graph import knn_graph
from graph_al.laplace import laplace_learning, laplace_margin


def _two_tight_clusters(seed: int = 1, per_cluster: int = 15, dim: int = 6):
    rng = np.random.default_rng(seed)
    center_a = rng.normal(size=dim) * 5.0
    center_b = -center_a
    points = np.concatenate([
        center_a + 0.05 * rng.normal(size=(per_cluster, dim)),
        center_b + 0.05 * rng.normal(size=(per_cluster, dim)),
    ], axis=0)
    labels = np.concatenate([np.zeros(per_cluster), np.ones(per_cluster)]).astype(np.int64)
    return torch.as_tensor(points, dtype=torch.float32), labels


def test_laplace_learning_recovers_well_separated_clusters():
    features, labels = _two_tight_clusters()
    W = knn_graph(features, k=5, kernel="symmetric", chunk_size=10)

    labeled_indices = np.array([0, len(labels) - 1])  # one per cluster
    u = laplace_learning(W, labeled_indices, labels[labeled_indices], num_classes=2)

    predicted = np.argmax(u, axis=1)
    assert np.mean(predicted == labels) == 1.0


def test_laplace_learning_labeled_rows_are_exact_one_hot():
    features, labels = _two_tight_clusters()
    W = knn_graph(features, k=5, kernel="symmetric", chunk_size=10)

    labeled_indices = np.array([0, 3, len(labels) - 1, len(labels) - 4])
    u = laplace_learning(W, labeled_indices, labels[labeled_indices], num_classes=2)

    for idx, label in zip(labeled_indices, labels[labeled_indices]):
        expected = np.zeros(2)
        expected[label] = 1.0
        np.testing.assert_allclose(u[idx], expected)


def test_laplace_learning_rejects_single_class():
    features, labels = _two_tight_clusters()
    W = knn_graph(features, k=5, kernel="symmetric", chunk_size=10)
    try:
        laplace_learning(W, np.array([0, 1]), np.array([0, 0]), num_classes=2)
        assert False, "expected ValueError for single-class labeled set"
    except ValueError:
        pass


def test_laplace_margin_matches_manual_computation():
    u = np.asarray([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    margin = laplace_margin(u)
    np.testing.assert_allclose(margin, [1.0 - 0.8, 1.0 - 0.0, 1.0 - 0.6], atol=1e-6)

    subset = laplace_margin(u, candidate_indices=np.array([1, 2]))
    np.testing.assert_allclose(subset, [1.0 - 0.0, 1.0 - 0.6], atol=1e-6)
