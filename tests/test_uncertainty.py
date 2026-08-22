import numpy as np

from sampling.uncertainty import (
    js_disagreement_from_logits,
    margin_uncertainty_from_logits,
    row_layer_norm,
)


def test_margin_uncertainty_orders_tie_above_confident_prediction():
    logits = np.asarray([[0.0, 0.0], [8.0, -8.0]], dtype=np.float32)
    uncertainty = margin_uncertainty_from_logits(logits)
    assert uncertainty[0] > uncertainty[1]
    assert np.all((0.0 <= uncertainty) & (uncertainty <= 1.0))


def test_jsd_is_symmetric_zero_for_equal_and_bounded():
    first = np.asarray([[2.0, -1.0], [8.0, -8.0]], dtype=np.float32)
    second = np.asarray([[2.0, -1.0], [-8.0, 8.0]], dtype=np.float32)
    forward = js_disagreement_from_logits(first, second)
    reverse = js_disagreement_from_logits(second, first)
    np.testing.assert_allclose(forward, reverse, atol=1e-6)
    assert forward[0] < 1e-6
    assert 0.99 < forward[1] <= 1.0


def test_row_layer_norm_removes_per_row_location_and_scale():
    values = np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    normalized = row_layer_norm(values)
    np.testing.assert_allclose(normalized[0], normalized[1], atol=1e-5)
    np.testing.assert_allclose(normalized.mean(axis=1), 0.0, atol=1e-6)
