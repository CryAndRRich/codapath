import importlib.util

import numpy as np
import pytest


if importlib.util.find_spec("torch") is None:
    pytest.skip("dual-view UH tests require torch", allow_module_level=True)

import torch

from sampling.dual_view_uherding import (
    _greedy_branch_order,
    _initial_kernel_coverage,
    _kernel_greedy_order,
    _precompute_relevant_kernel,
    _running_max_coverage,
    _static_fusion,
    _weighted_marginal_scores,
    disagreement_uherding_sampling,
    dual_view_uherding_sampling,
)


CPU = torch.device("cpu")


@pytest.mark.parametrize(
    "fusion",
    [
        "joint", "rrf", "borda", "score",
        "visual_then_cell", "cell_then_visual",
    ],
)
def test_dual_view_variants_return_unique_budget(fusion):
    rng = np.random.default_rng(21)
    n = 18
    reliability = np.ones(n, dtype=np.float32)
    reliability[:3] = 0.0
    selected = dual_view_uherding_sampling(
        image_embeddings=rng.normal(size=(n, 6)).astype(np.float32),
        nucleus_embeddings=rng.normal(size=(n, 5)).astype(np.float32),
        nucleus_reliability=reliability,
        oracle_labels=np.arange(n) % 3,
        num_classes=3,
        max_budget=6,
        device=CPU,
        uncertainty_mode="branch_margin",
        fusion_mode=fusion,
        num_rounds=2,
        shortlist_multiplier=2,
        candidate_pool_size=None,
        chunk_size=n,
        n_sigma=n,
        probe_epochs=1,
        diag=False,
    )
    assert len(selected) == 6
    assert len(set(selected)) == 6
    assert min(selected) >= 0 and max(selected) < n


def test_disagreement_control_returns_unique_budget():
    rng = np.random.default_rng(4)
    n = 16
    selected = disagreement_uherding_sampling(
        image_embeddings=rng.normal(size=(n, 5)).astype(np.float32),
        nucleus_embeddings=rng.normal(size=(n, 4)).astype(np.float32),
        nucleus_reliability=np.ones(n, dtype=np.float32),
        oracle_labels=np.arange(n) % 2,
        num_classes=2,
        max_budget=6,
        device=CPU,
        num_rounds=2,
        candidate_pool_size=None,
        chunk_size=n,
        n_sigma=n,
        probe_epochs=1,
        diag=False,
    )
    assert len(selected) == 6
    assert len(set(selected)) == 6


def test_dual_view_falls_back_to_visual_when_every_cell_view_is_missing():
    rng = np.random.default_rng(44)
    n = 12
    selected = dual_view_uherding_sampling(
        image_embeddings=rng.normal(size=(n, 5)).astype(np.float32),
        nucleus_embeddings=np.zeros((n, 4), dtype=np.float32),
        nucleus_reliability=np.zeros(n, dtype=np.float32),
        oracle_labels=np.arange(n) % 2,
        num_classes=2,
        max_budget=4,
        device=CPU,
        num_rounds=2,
        candidate_pool_size=None,
        chunk_size=n,
        n_sigma=n,
        probe_epochs=1,
        diag=False,
    )
    assert len(selected) == 4
    assert len(set(selected)) == 4


def test_ucov_marginal_gain_has_diminishing_returns():
    features = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]],
            dtype=torch.float32,
        ),
        p=2,
        dim=1,
    )
    weights = torch.tensor([1.0, 0.5, 0.8, 0.3])
    sigma = 0.8
    state_a = _running_max_coverage(features, [0], sigma, chunk_size=4)
    state_b = _running_max_coverage(features, [0, 2], sigma, chunk_size=4)
    gain_a = _weighted_marginal_scores(
        features, weights, state_a, sigma, np.asarray([3]), 4
    )[0]
    gain_b = _weighted_marginal_scores(
        features, weights, state_b, sigma, np.asarray([3]), 4
    )[0]
    assert gain_a >= gain_b - 1e-6


def test_precomputed_kernel_greedy_matches_direct_objective():
    rng = np.random.default_rng(12)
    features = torch.nn.functional.normalize(
        torch.as_tensor(rng.normal(size=(8, 4)), dtype=torch.float32),
        p=2,
        dim=1,
    )
    weights = torch.as_tensor(rng.uniform(size=8), dtype=torch.float32)
    sigma = 0.9
    selected = [0, 3]
    candidates = np.asarray([1, 2, 4, 5, 6, 7])
    direct_state = _running_max_coverage(features, selected, sigma, chunk_size=8)
    direct_order, _ = _greedy_branch_order(
        features, weights, direct_state, sigma, candidates, 3, 8
    )

    relevant = np.asarray(selected + candidates.tolist())
    kernel = _precompute_relevant_kernel(features, relevant, sigma, chunk_size=3)
    kernel_state = _initial_kernel_coverage(kernel, [0, 1])
    kernel_order, _ = _kernel_greedy_order(
        kernel,
        weights[torch.as_tensor(relevant)],
        kernel_state,
        np.arange(2, len(relevant)),
        relevant,
        3,
    )
    assert kernel_order == direct_order


def test_sum_and_mean_score_fusion_are_rank_equivalent():
    args = ([1, 2], [2, 3], [10.0, 4.0], [8.0, 2.0], 2)
    assert _static_fusion(*args, mode="sum", rrf_k=60.0) == _static_fusion(
        *args, mode="mean", rrf_k=60.0
    )


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError, match="uncertainty_mode"):
        dual_view_uherding_sampling(
            image_embeddings=np.ones((4, 2), dtype=np.float32),
            nucleus_embeddings=np.ones((4, 2), dtype=np.float32),
            nucleus_reliability=np.ones(4, dtype=np.float32),
            oracle_labels=np.asarray([0, 1, 0, 1]),
            num_classes=2,
            max_budget=2,
            device=CPU,
            uncertainty_mode="invalid",
            diag=False,
        )
