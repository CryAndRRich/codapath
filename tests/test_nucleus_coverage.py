import importlib.util

import numpy as np
import pytest

# torch, not just torchvision: this module imports the full sampler stack, and
# a plain-numpy environment must skip rather than error out during collection.
if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
    pytest.skip("full sampler environment requires torch + torchvision", allow_module_level=True)

import torch

from sampling.nucleus_coverage import (
    _build_coverage_features,
    _labeled_min_sigma,
    _norm_or_none,
    _running_max_coverage,
    nucleus_coverage_sampling,
)
from sampling.scalpel import _k_gaussian


CPU = torch.device("cpu")


@pytest.mark.parametrize("coverage_source", ["dino", "cellvit", "concat"])
def test_all_coverage_sources_return_unique_budget(coverage_source):
    rng = np.random.default_rng(7)
    n = 40
    reliability = np.ones(n, dtype=np.float32)
    reliability[:5] = 0.0  # exercise the missing-nucleus path
    selected = nucleus_coverage_sampling(
        image_embeddings=rng.normal(size=(n, 8)).astype(np.float32),
        nucleus_embeddings=rng.normal(size=(n, 6)).astype(np.float32),
        nucleus_reliability=reliability,
        oracle_labels=np.arange(n) % 2,
        num_classes=2,
        max_budget=25,
        device=CPU,
        coverage_source=coverage_source,
        num_rounds=5, chunk_size=40, n_sigma=40, probe_epochs=1,
        diag=False,
    )
    assert len(selected) == 25
    assert len(set(selected)) == 25
    assert min(selected) >= 0 and max(selected) < n
    # A degenerate all-zero score makes argmax return the lowest remaining
    # index every step, i.e. exactly 0, 1, 2, ... That must not happen here.
    assert selected != list(range(len(selected)))


def test_unknown_coverage_source_rejected():
    with pytest.raises(ValueError, match="coverage_source"):
        nucleus_coverage_sampling(
            image_embeddings=np.ones((5, 2), dtype=np.float32),
            nucleus_embeddings=np.ones((5, 2), dtype=np.float32),
            nucleus_reliability=np.ones(5, dtype=np.float32),
            oracle_labels=np.array([0, 1, 0, 1, 0]), num_classes=2, max_budget=2,
            device=CPU, coverage_source="bogus", diag=False,
        )


def test_unknown_missing_impute_rejected():
    with pytest.raises(ValueError, match="missing_impute"):
        nucleus_coverage_sampling(
            image_embeddings=np.ones((5, 2), dtype=np.float32),
            nucleus_embeddings=np.ones((5, 2), dtype=np.float32),
            nucleus_reliability=np.ones(5, dtype=np.float32),
            oracle_labels=np.array([0, 1, 0, 1, 0]), num_classes=2, max_budget=2,
            device=CPU, coverage_source="cellvit", missing_impute="bogus", diag=False,
        )


def test_mismatched_array_lengths_rejected():
    with pytest.raises(ValueError, match="align by patch"):
        nucleus_coverage_sampling(
            image_embeddings=np.ones((5, 2), dtype=np.float32),
            nucleus_embeddings=np.ones((4, 2), dtype=np.float32),
            nucleus_reliability=np.ones(5, dtype=np.float32),
            oracle_labels=np.array([0, 1, 0, 1, 0]), num_classes=2, max_budget=2,
            device=CPU, coverage_source="dino", diag=False,
        )


def test_all_missing_reliability_does_not_crash():
    """coverage_source='cellvit' with reliability all zero leaves nothing to
    impute from, so every coverage vector is the zero vector. The run must
    still complete — the degenerate-coverage guard falls back to uncertainty
    instead of silently ranking by index."""
    rng = np.random.default_rng(3)
    n = 20
    selected = nucleus_coverage_sampling(
        image_embeddings=rng.normal(size=(n, 8)).astype(np.float32),
        nucleus_embeddings=rng.normal(size=(n, 6)).astype(np.float32),
        nucleus_reliability=np.zeros(n, dtype=np.float32),
        oracle_labels=np.arange(n) % 2, num_classes=2, max_budget=10,
        device=CPU, coverage_source="cellvit",
        num_rounds=5, chunk_size=20, n_sigma=20, probe_epochs=1, diag=False,
    )
    assert len(selected) == 10
    assert len(set(selected)) == 10


# ---------------------------------------------------------------------------
# Kernel invariants — each of these caught a real, silent bug.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("coverage_source", ["dino", "cellvit", "concat"])
@pytest.mark.parametrize("missing_impute", ["mean", "zero"])
def test_rows_stay_unit_norm(coverage_source, missing_impute):
    """_k_gaussian and _adaptive_sigma read `A @ B.T` AS a cosine. A row with
    norm sqrt(2) (the naive concat) pushes the dot product above 1, makes
    `1 - cos` negative, and pins the kernel at 1 for most pairs."""
    rng = np.random.default_rng(11)
    n = 30
    reliability = np.ones(n, dtype=np.float32)
    reliability[:4] = 0.0
    features, missing_frac = _build_coverage_features(
        rng.normal(size=(n, 8)).astype(np.float32),
        rng.normal(size=(n, 6)).astype(np.float32),
        reliability, coverage_source, missing_impute, CPU,
    )
    norms = features.norm(p=2, dim=1)
    if coverage_source == "cellvit" and missing_impute == "zero":
        assert torch.allclose(norms[:4], torch.zeros(4), atol=1e-6)
        assert torch.allclose(norms[4:], torch.ones(n - 4), atol=1e-5)
    else:
        assert torch.allclose(norms, torch.ones(n), atol=1e-5)
    expected_missing = 0.0 if coverage_source == "dino" else 4 / n
    assert missing_frac == pytest.approx(expected_missing)
    assert float((features @ features.T).max()) <= 1.0 + 1e-5


def test_mean_impute_makes_missing_patches_typical_not_novel():
    """A zero-filled CellViT half has cosine 0 with everything, including other
    zero rows, so the greedy keeps treating those patches as unexplored. Mean
    imputation must instead place them at the centre of the valid vectors."""
    rng = np.random.default_rng(5)
    n = 24
    dino = rng.normal(size=(n, 8)).astype(np.float32)
    nucleus = rng.normal(size=(n, 6)).astype(np.float32)
    reliability = np.ones(n, dtype=np.float32)
    reliability[:3] = 0.0

    zeroed, _ = _build_coverage_features(dino, nucleus, reliability, "cellvit", "zero", CPU)
    imputed, _ = _build_coverage_features(dino, nucleus, reliability, "cellvit", "mean", CPU)

    assert torch.allclose(zeroed[:3] @ zeroed.T, torch.zeros(3, n), atol=1e-6)
    # Imputed rows are identical to each other and similar to the valid ones.
    assert torch.allclose(imputed[0], imputed[1], atol=1e-6)
    assert float((imputed[:3] @ imputed[3:].T).mean()) > float(
        (zeroed[:3] @ zeroed[3:].T).mean()
    )


def test_labeled_min_sigma_uses_euclidean_distance_not_squared():
    """sigma is squared again inside the kernel, so it must be a distance
    (sqrt(2 - 2cos)) and not a squared distance (1 - cos)."""
    theta = 0.7
    features = torch.tensor(
        [[1.0, 0.0], [float(np.cos(theta)), float(np.sin(theta))], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    expected = float(np.sqrt(2.0 - 2.0 * np.cos(theta)))
    assert _labeled_min_sigma(features) == pytest.approx(expected, rel=1e-5)
    assert _labeled_min_sigma(features) > (1.0 - np.cos(theta))  # not the squared form


def test_labeled_min_sigma_needs_two_points():
    with pytest.raises(ValueError, match="at least 2"):
        _labeled_min_sigma(torch.ones(1, 4))


def test_running_max_coverage_matches_bruteforce():
    """K_n must be rebuilt from scratch each round because sigma changes; this
    is the function that does it."""
    rng = np.random.default_rng(13)
    features = torch.nn.functional.normalize(
        torch.as_tensor(rng.normal(size=(17, 5)), dtype=torch.float32), p=2, dim=1
    )
    selected = [2, 9, 14]
    sigma = 0.8
    got = _running_max_coverage(features, selected, sigma, chunk_size=4)
    expected = _k_gaussian(features, features[selected], sigma).max(dim=1).values
    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.allclose(
        _running_max_coverage(features, [], sigma, 4), torch.zeros(17)
    )


def test_norm_or_none_flags_constant_input():
    assert _norm_or_none(np.full(6, 3.5, dtype=np.float32)) is None
    normalized = _norm_or_none(np.array([1.0, 2.0, 4.0], dtype=np.float32))
    assert normalized is not None
    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)


def test_sigma_floor_ratio_validated():
    with pytest.raises(ValueError, match="sigma_floor_ratio"):
        nucleus_coverage_sampling(
            image_embeddings=np.ones((5, 2), dtype=np.float32),
            nucleus_embeddings=np.ones((5, 2), dtype=np.float32),
            nucleus_reliability=np.ones(5, dtype=np.float32),
            oracle_labels=np.array([0, 1, 0, 1, 0]), num_classes=2, max_budget=2,
            device=CPU, coverage_source="dino", sigma_floor_ratio=1.0, diag=False,
        )
