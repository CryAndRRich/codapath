import numpy as np
import torch

from graph_al.graph import knn_graph_umap
from graph_al.laplace import personalized_pagerank


def _two_weakly_connected_clusters(seed: int = 2, per_cluster: int = 20, dim: int = 6):
    rng = np.random.default_rng(seed)
    center_a = rng.normal(size=dim) * 5.0
    center_b = -center_a
    points = np.concatenate([
        center_a + 0.05 * rng.normal(size=(per_cluster, dim)),
        center_b + 0.05 * rng.normal(size=(per_cluster, dim)),
    ], axis=0)
    labels = np.concatenate([np.zeros(per_cluster), np.ones(per_cluster)]).astype(np.int64)
    return torch.as_tensor(points, dtype=torch.float32), labels


def test_ppr_concentrates_on_the_labeled_cluster():
    features, labels = _two_weakly_connected_clusters()
    W = knn_graph_umap(features, k=5, chunk_size=10)

    labeled_in_a = np.where(labels == 0)[0][:1]
    pi = personalized_pagerank(W, labeled_in_a, damping=0.85)

    assert pi.shape == (len(labels),)
    assert np.all(pi >= 0.0)
    mean_a = pi[labels == 0].mean()
    mean_b = pi[labels == 1].mean()
    assert mean_a > mean_b


def test_ppr_rejects_empty_labeled_set():
    features, _ = _two_weakly_connected_clusters()
    W = knn_graph_umap(features, k=5, chunk_size=10)
    try:
        personalized_pagerank(W, np.array([], dtype=np.int64))
        assert False, "expected ValueError for empty labeled_indices"
    except ValueError:
        pass


def test_ppr_rejects_bad_damping():
    features, _ = _two_weakly_connected_clusters()
    W = knn_graph_umap(features, k=5, chunk_size=10)
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            personalized_pagerank(W, np.array([0]), damping=bad)
            assert False, f"expected ValueError for damping={bad}"
        except ValueError:
            pass
