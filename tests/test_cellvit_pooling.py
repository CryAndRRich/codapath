import numpy as np
import pytest

from features.cellvit.pooling import pool_cells_mean


def test_confidence_weighted_pool_and_empty_patch():
    features = np.asarray([[1.0, 0.0], [3.0, 2.0], [9.0, 9.0]])
    offsets = np.asarray([0, 2, 2, 3])
    confidence = np.asarray([0.25, 0.75, 0.5])

    view = pool_cells_mean(features, offsets, confidence)

    np.testing.assert_allclose(view.patch_features[0], [2.5, 1.5])
    np.testing.assert_allclose(view.patch_features[1], [0.0, 0.0])
    np.testing.assert_allclose(view.patch_features[2], [9.0, 9.0])
    np.testing.assert_array_equal(view.cell_counts, [2, 0, 1])
    np.testing.assert_array_equal(view.reliability, [1.0, 0.0, 1.0])


def test_mean_confidence_reliability_is_bounded():
    view = pool_cells_mean(
        np.ones((2, 1)), np.asarray([0, 2]),
        confidence=np.asarray([-2.0, 3.0]),
        reliability_mode="mean_confidence",
    )
    np.testing.assert_allclose(view.reliability, [0.5])


def test_rejects_invalid_offsets():
    with pytest.raises(ValueError, match="offsets"):
        pool_cells_mean(np.ones((2, 3)), np.asarray([0, 2, 1]))
