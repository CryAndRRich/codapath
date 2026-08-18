from identity import sample_order_fingerprint


def test_sample_fingerprint_depends_on_membership_and_order():
    original = sample_order_fingerprint(["a", "b"])
    assert original == sample_order_fingerprint(["a", "b"])
    assert original != sample_order_fingerprint(["b", "a"])
    assert original != sample_order_fingerprint(["a", "c"])


def test_fingerprint_uses_unambiguous_boundaries():
    assert sample_order_fingerprint(["ab", "c"]) != sample_order_fingerprint(
        ["a", "bc"]
    )
