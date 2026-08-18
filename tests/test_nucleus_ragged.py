import numpy as np
import pytest

from nucleus.ragged import pool_ragged_features


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


def test_kde_pooling_follows_the_mode_not_the_centroid():
    """5 cells clustered at [1,0] plus one distant outlier at [0,9].

    Mean pooling lets the single outlier drag the patch descriptor a long way;
    KDE pooling weights by local density so the descriptor stays near the
    dominant cell population.
    """
    cluster = np.array([[1.0, 0.0], [1.1, 0.0], [0.9, 0.0], [1.0, 0.1], [1.0, -0.1]],
                       dtype=np.float32)
    feats = np.vstack([cluster, np.array([[0.0, 9.0]], dtype=np.float32)])
    offsets = np.array([0, 6])

    mean_pooled = pool_ragged_features(feats, offsets, pool_mode="mean").patch_features[0]
    kde_pooled = pool_ragged_features(feats, offsets, pool_mode="kde").patch_features[0]

    mode = np.array([1.0, 0.0], dtype=np.float32)
    assert np.linalg.norm(kde_pooled - mode) < np.linalg.norm(mean_pooled - mode)


def test_kde_reduces_to_mean_for_degenerate_patches():
    """One cell, or numerically identical cells, must give exactly mean pooling."""
    single = np.array([[5.0, 5.0]], dtype=np.float32)
    offsets = np.array([0, 1])
    assert np.allclose(
        pool_ragged_features(single, offsets, pool_mode="kde").patch_features,
        pool_ragged_features(single, offsets, pool_mode="mean").patch_features,
    )

    identical = np.repeat(np.array([[2.0, 3.0]], dtype=np.float32), 4, axis=0)
    offsets = np.array([0, 4])
    assert np.allclose(
        pool_ragged_features(identical, offsets, pool_mode="kde").patch_features,
        pool_ragged_features(identical, offsets, pool_mode="mean").patch_features,
    )


def test_pool_mode_guard():
    feats = np.array([[1.0, 0.0]], dtype=np.float32)
    try:
        pool_ragged_features(feats, np.array([0, 1]), pool_mode="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown pool_mode")
