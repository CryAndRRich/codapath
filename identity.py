"""Stable identifiers shared by feature caches and dataset loaders."""

import hashlib
from typing import Sequence


def sample_order_fingerprint(sample_ids: Sequence[str]) -> str:
    """Return a SHA256 digest that changes with membership or row order."""
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()