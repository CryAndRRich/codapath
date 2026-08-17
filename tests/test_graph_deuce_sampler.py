import importlib.util

import numpy as np
import pytest
import torch

if importlib.util.find_spec("torchvision") is None:
    pytest.skip("full sampler environment requires torchvision", allow_module_level=True)

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
