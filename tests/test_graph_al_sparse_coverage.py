import numpy as np
import scipy.sparse as sps
import torch

from graph_al.sparse_coverage import greedy_coverage_sparse


def test_greedy_coverage_sparse_picks_unique_points():
    # 6-node graph: two tight triangles connected by one weak bridge edge.
    dense = np.zeros((6, 6))
    for i, j, w in [
        (0, 1, 0.9), (1, 2, 0.9), (0, 2, 0.9),
        (3, 4, 0.9), (4, 5, 0.9), (3, 5, 0.9),
        (2, 3, 0.1),
    ]:
        dense[i, j] = w
        dense[j, i] = w
    W = sps.csr_matrix(dense)
    U = torch.ones(6, dtype=torch.float32)

    picks = greedy_coverage_sparse(W, U, n_select=3, selected_set=set(), device=torch.device("cpu"))

    assert len(picks) == 3
    assert len(set(picks)) == 3
    assert all(0 <= p < 6 for p in picks)


def test_greedy_coverage_sparse_first_pick_matches_uniform_weight_argmax():
    dense = np.zeros((4, 4))
    dense[0, 1] = dense[1, 0] = 0.9
    dense[0, 2] = dense[2, 0] = 0.9
    dense[0, 3] = dense[3, 0] = 0.9
    dense[1, 2] = dense[2, 1] = 0.1
    W = sps.csr_matrix(dense)
    U = torch.ones(4, dtype=torch.float32)

    picks = greedy_coverage_sparse(W, U, n_select=1, selected_set=set(), device=torch.device("cpu"))

    # node 0 has the most/strongest edges -> highest sum(max(w-0,0)) at step 1
    assert picks == [0]


def test_greedy_coverage_sparse_respects_already_selected():
    dense = np.eye(3) * 0.0
    dense[0, 1] = dense[1, 0] = 0.5
    dense[1, 2] = dense[2, 1] = 0.9
    W = sps.csr_matrix(dense)
    U = torch.ones(3, dtype=torch.float32)

    already = {1}
    picks = greedy_coverage_sparse(W, U, n_select=2, selected_set=already, device=torch.device("cpu"))

    assert 1 not in picks
    assert set(picks) == {0, 2}
