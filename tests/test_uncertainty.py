import numpy as np

from sampling.uncertainty import (
    js_disagreement_from_logits,
    margin_uncertainty_from_logits,
    rank_normalize,
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


# --- rank_normalize: the fix for SCALPEL's weight-scale collapse ----------

def test_rank_normalize_maps_a_constant_vector_to_a_neutral_value():
    """NOT to zeros. An all-zero weight makes every marginal gain zero and
    `argmax` silently returns index order -- the same trap CLAUDE.md records
    for min-max on a constant vector. Ties share their average rank, so a
    constant input is the limiting case and lands at 0.5."""
    out = rank_normalize(np.full(5, 0.3, dtype=np.float32))
    assert np.allclose(out, 0.5)


def test_rank_normalize_spans_the_unit_interval_and_preserves_order():
    values = np.asarray([3.0, 1.0, 4.0, 1.5, 2.0], dtype=np.float32)
    out = rank_normalize(values)
    assert out.min() == 0.0 and out.max() == 1.0
    assert list(np.argsort(out)) == list(np.argsort(values))


def test_rank_normalize_averages_ties():
    """Equal inputs must get equal outputs, or array position becomes a
    tie-breaker that no part of the method intends."""
    out = rank_normalize(np.asarray([1.0, 2.0, 2.0, 2.0, 5.0], dtype=np.float32))
    assert out[1] == out[2] == out[3]


def test_rank_normalize_respects_a_mask_and_leaves_the_rest_at_zero():
    mask = np.asarray([True, True, False, True])
    out = rank_normalize(np.asarray([3.0, 1.0, 99.0, 2.0], dtype=np.float32), mask)
    assert out[2] == 0.0, "a masked-out point must not receive a rank"
    assert sorted(out[mask]) == [0.0, 0.5, 1.0]


def test_rank_normalize_handles_a_single_member_group():
    out = rank_normalize(np.asarray([7.0], dtype=np.float32))
    assert out[0] == 0.5, "one point has no ordering, so it must be neutral"


def test_rank_normalize_handles_an_empty_group():
    out = rank_normalize(np.asarray([1.0, 2.0], dtype=np.float32), np.asarray([False, False]))
    assert np.all(out == 0.0)


def test_rank_equalizes_two_wildly_different_scales_where_minmax_does_not():
    """THE regression test for the weight collapse.

    A real 14-class histoset run mixed Jensen-Shannon disagreement (measured
    mean ~0.02) with the visual margin (~0.7) raw. Because nucleus-free
    patches take the margin branch, they swept the top of the weight ranking:
    all 200 of the top-200 weights were nucleus-free patches where a fair
    share is ~18 of 200. Rounds 1-4 then put 150 of 160 picks into 5 of the
    14 classes and never sampled 4 of them, while round 0 (U=1, no weight)
    had covered all 14.

    Min-max is the obvious fix and is NOT sufficient -- it rescales the range
    but not the shape, and JS is long-tailed. This asserts on the quantity
    that actually matters: the share of the top-K weights taken by the
    fallback group.
    """
    rng = np.random.default_rng(0)
    n = 22400
    reliability = np.ones(n, dtype=np.float32)
    reliability[rng.choice(n, int(0.0924 * n), replace=False)] = 0.0
    valid = reliability > 0.0
    disagreement = np.abs(rng.normal(0.02, 0.01, n)).astype(np.float32)
    margin = rng.uniform(0.4, 0.95, n).astype(np.float32)

    def top_share(weights, k=200):
        top = np.argsort(-weights)[:k]
        return int((~valid[top]).sum())

    fair = int(round(200 * (~valid).mean()))

    raw = reliability * disagreement + (1.0 - reliability) * margin
    assert top_share(raw) > 150, "fixture no longer reproduces the original collapse"

    def minmax(v):
        lo, hi = v.min(), v.max()
        return np.zeros_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)

    scaled = reliability * minmax(disagreement) + (1.0 - reliability) * minmax(margin)
    assert top_share(scaled) > 100, "min-max was expected to remain badly skewed"

    ranked = (
        reliability * rank_normalize(disagreement, valid)
        + (1.0 - reliability) * rank_normalize(margin)
    )
    assert abs(top_share(ranked) - fair) <= 15, (
        f"ranked weights put {top_share(ranked)} nucleus-free patches in the "
        f"top-200; a fair share is about {fair}"
    )


def test_ranked_weight_keeps_a_partially_reliable_point_from_losing_its_margin():
    """`reliability_mode="mean_confidence"` makes rho continuous, so a patch
    with rho=0.3 draws 70% of its weight from the margin while still counting
    as `valid`. Ranking the margin only inside the nucleus-free group would
    hand every such patch a margin term of exactly 0 -- zeroing most of its
    weight. The margin is defined pool-wide, so it is ranked pool-wide."""
    rng = np.random.default_rng(1)
    n = 2000
    reliability = rng.uniform(0.0, 1.0, n).astype(np.float32)
    reliability[rng.choice(n, 100, replace=False)] = 0.0
    valid = reliability > 0.0
    disagreement = np.abs(rng.normal(0.02, 0.01, n)).astype(np.float32)
    margin = rng.uniform(0.4, 0.95, n).astype(np.float32)

    weights = (
        reliability * rank_normalize(disagreement, valid)
        + (1.0 - reliability) * rank_normalize(margin)
    )
    partial = (reliability > 0.0) & (reliability < 1.0)
    assert partial.sum() > 0
    assert not np.any(weights[partial] == 0.0), (
        "a partially reliable patch lost its entire margin contribution"
    )
