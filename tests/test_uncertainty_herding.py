import importlib.util

import numpy as np
import pytest


if importlib.util.find_spec("torch") is None:
    pytest.skip("UHerding tests require torch", allow_module_level=True)

import torch

from sampling.uncertainty_herding import _ece, uncertainty_herding_sampling


def test_ece_is_small_for_correct_confident_predictions():
    logits = np.array([[12.0, -4.0], [-3.0, 11.0]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    assert _ece(logits, labels, n_bins=2) < 1e-4


def test_multiround_uherding_returns_unique_budget_on_cpu():
    rng = np.random.default_rng(17)
    n = 24
    selected = uncertainty_herding_sampling(
        image_embeddings=rng.normal(size=(n, 6)).astype(np.float32),
        oracle_labels=np.arange(n) % 3,
        num_classes=3,
        max_budget=10,
        num_rounds=2,
        probe_epochs=1,
        probe_lr=1e-3,
        chunk_size=n,
        device=torch.device("cpu"),
    )
    assert len(selected) == 10
    assert len(set(selected)) == 10
    assert min(selected) >= 0 and max(selected) < n
