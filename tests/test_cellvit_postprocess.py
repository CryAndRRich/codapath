import numpy as np
import pytest

from features.cellvit.postprocess import stack_prediction_maps_numpy


def test_stack_prediction_maps_matches_cellvit_semantics_without_numba():
    rng = np.random.default_rng(4)
    type_map = rng.normal(size=(2, 5, 7, 6)).astype(np.float16)
    binary_map = rng.normal(size=(2, 5, 7, 2)).astype(np.float16)
    hv_map = rng.normal(size=(2, 5, 7, 2)).astype(np.float16)

    result = stack_prediction_maps_numpy(type_map, binary_map, hv_map)

    assert result.shape == (2, 5, 7, 4)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[..., 0], np.argmax(type_map, axis=-1))
    np.testing.assert_array_equal(result[..., 1], np.argmax(binary_map, axis=-1))
    np.testing.assert_array_equal(result[..., 2:], hv_map.astype(np.float32))


@pytest.mark.parametrize(
    "binary_shape,hv_shape",
    [((1, 4, 4, 3), (1, 4, 4, 2)), ((1, 4, 4, 2), (1, 4, 4, 1))],
)
def test_stack_prediction_maps_rejects_invalid_channels(binary_shape, hv_shape):
    with pytest.raises(ValueError, match="two channels"):
        stack_prediction_maps_numpy(
            np.zeros((1, 4, 4, 6), dtype=np.float32),
            np.zeros(binary_shape, dtype=np.float32),
            np.zeros(hv_shape, dtype=np.float32),
        )
