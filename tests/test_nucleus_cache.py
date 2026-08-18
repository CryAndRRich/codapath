import json

import numpy as np
import pytest

from nucleus.cache import load_nucleus_cache, save_nucleus_cache


def test_cache_round_trip_and_exact_sample_order(tmp_path):
    sample_ids = ["a/0.png", "b/1.png", "b/2.png"]
    save_nucleus_cache(
        str(tmp_path),
        offsets=np.asarray([0, 2, 2, 3]),
        confidence=np.asarray([0.7, 0.9, 0.8]),
        sample_ids=sample_ids,
        cellvit_embeddings=np.arange(12, dtype=np.float16).reshape(3, 4),
        cell_dino_features=np.arange(9, dtype=np.float16).reshape(3, 3),
        bboxes=np.zeros((3, 4), dtype=np.int32),
        manifest={"dataset": "toy", "seed": 42},
    )

    cache = load_nucleus_cache(
        str(tmp_path), expected_sample_ids=sample_ids, mmap_mode="r"
    )
    assert cache.num_patches == 3
    assert cache.num_cells == 3
    assert cache.features("cellvit_embedding").dtype == np.float16
    assert cache.manifest["sample_fingerprint"]

    with pytest.raises(ValueError, match="sample order"):
        load_nucleus_cache(
            str(tmp_path), expected_sample_ids=list(reversed(sample_ids))
        )


def test_manifest_source_flags_ignore_stale_optional_file(tmp_path):
    save_nucleus_cache(
        str(tmp_path),
        offsets=np.asarray([0, 1]),
        confidence=np.asarray([1.0]),
        sample_ids=["x"],
        cellvit_embeddings=np.ones((1, 2), dtype=np.float32),
        manifest={},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["has_cellvit_embeddings"] = False
    manifest_path.write_text(json.dumps(manifest))

    cache = load_nucleus_cache(str(tmp_path))
    with pytest.raises(ValueError, match="does not contain"):
        cache.features("cellvit_embedding")


def test_cache_rejects_nonfinite_features(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        save_nucleus_cache(
            str(tmp_path), offsets=np.asarray([0, 1]),
            confidence=np.asarray([1.0]), sample_ids=["x"], manifest={},
            cellvit_embeddings=np.asarray([[np.nan]], dtype=np.float32),
        )
