import importlib.util
from unittest.mock import patch

import numpy as np
import pytest
import torch

if importlib.util.find_spec("torchvision") is None:
    pytest.skip("full sampler environment requires torchvision", allow_module_level=True)

import sampling.graph_deuce as graph_deuce_module
from sampling.graph_deuce import graph_deuce_sampling


def _toy_pool(n=60, dino_dim=16, cell_dim=10, num_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(num_classes, dino_dim)) * 4.0
    labels = rng.integers(0, num_classes, size=n)
    dino = np.stack([centers[labels[i]] + 0.2 * rng.normal(size=dino_dim) for i in range(n)]).astype(np.float32)

    cell = rng.normal(size=(n, cell_dim)).astype(np.float32)
    reliability = np.ones(n, dtype=np.float32)
    reliability[:8] = 0.0  # exercise the missing-nucleus / isolated-node path
    cell[:8] = 0.0

    return dino, cell, reliability, labels.astype(np.int64)


@pytest.mark.parametrize("variant", [
    "laplace_margin",
    "uherding_swap_uncertainty",
    "uherding_swap_coverage",
    "laplace_plus_ppr",
])
def test_all_variants_return_unique_budget(variant):
    dino, cell, reliability, labels = _toy_pool()

    selected = graph_deuce_sampling(
        image_embeddings=dino,
        nucleus_embeddings=cell,
        nucleus_reliability=reliability,
        oracle_labels=labels,
        num_classes=3,
        max_budget=15,
        device=torch.device("cpu"),
        acquisition_variant=variant,
        k=5,
        chunk_size=32,
        vae_epochs=3,
        probe_epochs=2,
    )

    assert len(selected) == 15
    assert len(set(selected)) == 15
    assert min(selected) >= 0 and max(selected) < len(labels)


def test_unknown_acquisition_variant_rejected():
    dino, cell, reliability, labels = _toy_pool()
    with pytest.raises(ValueError, match="acquisition_variant"):
        graph_deuce_sampling(
            image_embeddings=dino,
            nucleus_embeddings=cell,
            nucleus_reliability=reliability,
            oracle_labels=labels,
            num_classes=3,
            max_budget=5,
            device=torch.device("cpu"),
            acquisition_variant="bogus",
            k=5,
            chunk_size=32,
            vae_epochs=2,
        )


def test_default_variant_is_laplace_margin():
    dino, cell, reliability, labels = _toy_pool()
    # Should not raise, and should behave like acquisition_variant="laplace_margin"
    selected = graph_deuce_sampling(
        image_embeddings=dino,
        nucleus_embeddings=cell,
        nucleus_reliability=reliability,
        oracle_labels=labels,
        num_classes=3,
        max_budget=10,
        device=torch.device("cpu"),
        k=5,
        chunk_size=32,
        vae_epochs=2,
    )
    assert len(selected) == 10


def test_all_missing_nucleus_raises_clear_error():
    """Toàn bộ patch thiếu nucleus (reliability=0 hết) là 1 case biên hợp lệ
    (không phải bug) nhưng graph_deuce cần báo lỗi RÕ RÀNG thay vì crash mơ hồ
    trong DataLoader (mục 2 EXPERIMENT.md: coverage cell cần >=1 patch reliable)."""
    dino, cell, reliability, labels = _toy_pool()
    reliability[:] = 0.0
    with pytest.raises(ValueError, match="reliability>0"):
        graph_deuce_sampling(
            image_embeddings=dino,
            nucleus_embeddings=cell,
            nucleus_reliability=reliability,
            oracle_labels=labels,
            num_classes=3,
            max_budget=10,
            device=torch.device("cpu"),
            k=5,
            chunk_size=32,
            vae_epochs=2,
        )


@pytest.mark.parametrize("variant", ["laplace_margin", "laplace_plus_ppr"])
def test_cold_start_round_uses_real_coverage_greedy_not_uniform_tiebreak(variant):
    """Regression for the 2026-08-17 bug found while investigating low
    budget=25 accuracy: round 1 (no labels revealed yet, Laplace learning
    undefined) used to feed `scores=ones(N)` into `_select_batch_with_discount`,
    whose first pick is `torch.argmax` over an all-tied array — always the
    LOWEST index, unrelated to any coverage property (violates this project's
    coverage-only-round-1 convention, see CLAUDE.md's 2x2 sampler table).
    Fixed to dispatch cold-start rounds to `greedy_coverage_sparse` (the same
    genuine facility-location greedy `uherding_swap_coverage`'s own round 1
    already used) instead. This test locks in that dispatch."""
    dino, cell, reliability, labels = _toy_pool()

    with patch(
        "sampling.graph_deuce.greedy_coverage_sparse",
        wraps=graph_deuce_module.greedy_coverage_sparse,
    ) as mock_greedy:
        graph_deuce_sampling(
            image_embeddings=dino,
            nucleus_embeddings=cell,
            nucleus_reliability=reliability,
            oracle_labels=labels,
            num_classes=3,
            max_budget=15,
            device=torch.device("cpu"),
            acquisition_variant=variant,
            k=5,
            chunk_size=32,
            vae_epochs=2,
            probe_epochs=2,
        )

    assert mock_greedy.call_count >= 1
    first_call_U = mock_greedy.call_args_list[0].args[1]
    assert torch.all(first_call_U == 1.0)


def test_second_call_with_same_pool_reuses_cached_graph_no_vae_retrain():
    """`run.py` calls graph_deuce_sampling once per budget in cumulative_budget
    (and once per acquisition_variant tried in the same session) with the
    SAME `image_embeddings`/`nucleus_embeddings` numpy arrays every time — VAE
    training is fully unsupervised and budget-independent, so retraining both
    VAEs from scratch on every call would be pure waste. `_build_dual_graph`
    (which calls `train_vae`) must NOT be invoked again on the second call
    with the identical arrays."""
    dino, cell, reliability, labels = _toy_pool()
    graph_deuce_module._GRAPH_CACHE.clear()

    graph_deuce_sampling(
        image_embeddings=dino, nucleus_embeddings=cell, nucleus_reliability=reliability,
        oracle_labels=labels, num_classes=3, max_budget=10, device=torch.device("cpu"),
        acquisition_variant="laplace_margin", k=5, chunk_size=32, vae_epochs=2,
    )

    with patch(
        "sampling.graph_deuce._build_dual_graph",
        wraps=graph_deuce_module._build_dual_graph,
    ) as mock_build:
        graph_deuce_sampling(
            image_embeddings=dino, nucleus_embeddings=cell, nucleus_reliability=reliability,
            oracle_labels=labels, num_classes=3, max_budget=15, device=torch.device("cpu"),
            acquisition_variant="uherding_swap_coverage", k=5, chunk_size=32, vae_epochs=2,
        )
    mock_build.assert_not_called()
    graph_deuce_module._GRAPH_CACHE.clear()


@pytest.mark.parametrize("variant", ["laplace_margin", "laplace_plus_ppr"])
def test_per_point_returns_unique_budget(variant):
    dino, cell, reliability, labels = _toy_pool()
    selected = graph_deuce_sampling(
        image_embeddings=dino, nucleus_embeddings=cell, nucleus_reliability=reliability,
        oracle_labels=labels, num_classes=3, max_budget=15, device=torch.device("cpu"),
        acquisition_variant=variant, per_point=True, k=5, chunk_size=32, vae_epochs=2,
    )
    assert len(selected) == 15
    assert len(set(selected)) == 15


@pytest.mark.parametrize("variant", ["laplace_margin", "laplace_plus_ppr"])
def test_per_point_resolves_laplace_learning_far_more_often_than_round_based(variant):
    """Confirms the whole point of `per_point=True`: Laplace learning gets
    re-solved after EVERY pick past cold-start (matching the original
    SARGraphAL sequential AL loop, see graph_al/laplace.py docstring),
    instead of once per round (~5 rounds total regardless of budget)."""
    dino, cell, reliability, labels = _toy_pool()

    with patch(
        "sampling.graph_deuce.laplace_learning", wraps=graph_deuce_module.laplace_learning,
    ) as mock_batch:
        graph_deuce_sampling(
            image_embeddings=dino, nucleus_embeddings=cell, nucleus_reliability=reliability,
            oracle_labels=labels, num_classes=3, max_budget=15, device=torch.device("cpu"),
            acquisition_variant=variant, per_point=False, k=5, chunk_size=32, vae_epochs=2,
        )
    batch_calls = mock_batch.call_count

    with patch(
        "sampling.graph_deuce.laplace_learning", wraps=graph_deuce_module.laplace_learning,
    ) as mock_perpoint:
        graph_deuce_sampling(
            image_embeddings=dino, nucleus_embeddings=cell, nucleus_reliability=reliability,
            oracle_labels=labels, num_classes=3, max_budget=15, device=torch.device("cpu"),
            acquisition_variant=variant, per_point=True, k=5, chunk_size=32, vae_epochs=2,
        )
    perpoint_calls = mock_perpoint.call_count

    assert perpoint_calls > batch_calls


def test_reliable_subset_smaller_than_k_raises_clear_error():
    """Subset reliable quá nhỏ (<=k) phải báo lỗi rõ từ knn_graph_partial,
    không phải lỗi mơ hồ khác."""
    dino, cell, reliability, labels = _toy_pool()
    reliability[:] = 0.0
    reliability[:3] = 1.0  # only 3 reliable patches, k=5 needs >5
    with pytest.raises(ValueError, match="reliable subset"):
        graph_deuce_sampling(
            image_embeddings=dino,
            nucleus_embeddings=cell,
            nucleus_reliability=reliability,
            oracle_labels=labels,
            num_classes=3,
            max_budget=10,
            device=torch.device("cpu"),
            k=5,
            chunk_size=32,
            vae_epochs=2,
        )
