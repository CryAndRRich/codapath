import importlib.util

import numpy as np
import pytest
import torch


if importlib.util.find_spec("torchvision") is None:
    pytest.skip("full sampler environment requires torchvision", allow_module_level=True)

from sampling.nucleus_al import nucleus_al_sampling


@pytest.mark.parametrize(
    "source,mode",
    [
        ("crop_dino", "cell_margin"),
        ("cellvit_embedding", "disagreement"),
        ("cellvit_embedding", "fusion_concat"),
        ("cellvit_embedding", "fusion_add"),
    ],
)
def test_all_planned_variants_return_unique_budget(source, mode):
    rng = np.random.default_rng(7)
    n = 30
    selected = nucleus_al_sampling(
        image_embeddings=rng.normal(size=(n, 8)).astype(np.float32),
        nucleus_embeddings=rng.normal(size=(n, 6)).astype(np.float32),
        nucleus_reliability=np.ones(n, dtype=np.float32),
        oracle_labels=np.arange(n) % 2,
        num_classes=2,
        max_budget=25,
        device=torch.device("cpu"),
        cell_source=source,
        uncertainty_mode=mode,
        num_rounds=5,
        chunk_size=30,
        n_sigma=30,
        probe_epochs=1,
        fusion_min_labels_per_class=1,
        diag=False,
    )
    assert len(selected) == 25
    assert len(set(selected)) == 25
    assert min(selected) >= 0 and max(selected) < n


def test_crop_dino_rejects_unplanned_uncertainty_axis():
    with pytest.raises(ValueError, match="cell_margin only"):
        nucleus_al_sampling(
            image_embeddings=np.ones((2, 2), dtype=np.float32),
            nucleus_embeddings=np.ones((2, 2), dtype=np.float32),
            nucleus_reliability=np.ones(2, dtype=np.float32),
            oracle_labels=np.asarray([0, 1]), num_classes=2, max_budget=1,
            device=torch.device("cpu"), cell_source="crop_dino",
            uncertainty_mode="disagreement", diag=False,
        )
