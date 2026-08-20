import numpy as np
import pytest

from nucleus.ragged import (
    pool_ragged_features,
    pool_ragged_moments,
    pool_ragged_rff,
)


def test_confidence_weighted_pool_and_empty_patch():
    features = np.asarray([[1.0, 0.0], [3.0, 2.0], [9.0, 9.0]])
    offsets = np.asarray([0, 2, 2, 3])
    confidence = np.asarray([0.25, 0.75, 0.5])

    view = pool_ragged_features(features, offsets, confidence)

    np.testing.assert_allclose(view.patch_features[0], [2.5, 1.5])
    np.testing.assert_allclose(view.patch_features[1], [0.0, 0.0])
    np.testing.assert_allclose(view.patch_features[2], [9.0, 9.0])
    np.testing.assert_array_equal(view.cell_counts, [2, 0, 1])
    np.testing.assert_array_equal(view.reliability, [1.0, 0.0, 1.0])


def test_mean_confidence_reliability_is_bounded():
    view = pool_ragged_features(
        np.ones((2, 1)), np.asarray([0, 2]),
        confidence=np.asarray([-2.0, 3.0]),
        reliability_mode="mean_confidence",
    )
    np.testing.assert_allclose(view.reliability, [0.5])


def test_rejects_invalid_offsets():
    with pytest.raises(ValueError, match="offsets"):
        pool_ragged_features(np.ones((2, 3)), np.asarray([0, 2, 1]))


def test_moment_pool_preserves_dispersion_and_qc():
    features = np.asarray([[1.0, 0.0], [3.0, 2.0], [9.0, 9.0]])
    offsets = np.asarray([0, 2, 2, 3])
    confidence = np.asarray([0.25, 0.75, 0.5])

    view = pool_ragged_moments(features, offsets, confidence)

    np.testing.assert_allclose(view.patch_features[0, :2], [2.5, 1.5])
    np.testing.assert_allclose(
        view.patch_features[0, 2:4], [np.sqrt(0.75), np.sqrt(0.75)]
    )
    np.testing.assert_allclose(view.patch_features[0, -2:], [np.log(3.0), 0.5])
    np.testing.assert_array_equal(view.patch_features[1], np.zeros(6))
    np.testing.assert_allclose(view.patch_features[2, 2:4], [0.0, 0.0])
    np.testing.assert_array_equal(view.reliability, [1.0, 0.0, 1.0])


def test_rff_pool_is_deterministic_and_preserves_distribution_information():
    # The two patches have the same raw mean (zero) but different empirical
    # cell distributions. Raw mean pooling collapses them; Gaussian mean-map
    # pooling must not.
    features = np.asarray(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    offsets = np.asarray([0, 2, 4])
    mean_view = pool_ragged_features(features, offsets)
    first = pool_ragged_rff(
        features, offsets, output_dim=128, bandwidth=0.7, seed=9
    )
    second = pool_ragged_rff(
        features, offsets, output_dim=128, bandwidth=0.7, seed=9
    )

    np.testing.assert_allclose(mean_view.patch_features[0], mean_view.patch_features[1])
    assert not np.allclose(first.patch_features[0], first.patch_features[1])
    np.testing.assert_allclose(first.patch_features, second.patch_features)
    assert first.metadata["rff_bandwidth"] == pytest.approx(0.7)


def test_rff_pool_handles_empty_patch_and_zero_confidence():
    view = pool_ragged_rff(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([0, 2, 2]),
        confidence=np.zeros(2, dtype=np.float32),
        output_dim=16,
        bandwidth=1.0,
    )
    assert np.linalg.norm(view.patch_features[0]) > 0.0
    np.testing.assert_array_equal(view.patch_features[1], np.zeros(16))
    np.testing.assert_array_equal(view.reliability, [1.0, 0.0])
