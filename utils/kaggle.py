"""Locate Kaggle Dataset caches by (dataset, seed, backbone) instead of a path.

Publishing `/kaggle/working/<name>` as a Kaggle Dataset remounts it one level
deeper than the path anyone writes down: `/kaggle/input/<slug>/<name>/<name>`.
A notebook that hard-codes the expected mount point breaks the first time the
slug changes, and fails in a way that is tedious to debug from a stack trace
(the error surfaces deep inside `main.py`, long after the cache lookup).

Every notebook that reads a cache produced by another notebook hits this same
problem, so the search lives here once instead of being copy-pasted (and
re-drifting) into each one. The caller supplies a search root -- typically
`find_data_root()` -- plus the (dataset, seed, backbone) that name the cache
files, per the naming convention `features/visual.py` and
`scripts/extract_cellvit_features.py` actually use. Nothing here changes that
convention; it only searches for it.

`find_dir_containing` matches a fixed multi-segment path (e.g.
`"pathmnist_seed42/manifest.json"`) under some number of unknown leading
directories -- it does not care how deep the file is, but it does require
that exact trailing segment sequence. `find_visual_cache`/`find_vlm_cache`
rely on this safely because their cache files are self-naming
(`pathmnist_seed42_facebook_dinov2-base_train.npy` already encodes dataset and
seed in the FILENAME, independent of whatever directories it sits under).
`find_cellvit_cache` cannot: its cache is a bare `manifest.json` inside a
directory conventionally named `{dataset}_seed{seed}`, so if someone
re-uploads that cache with the dataset and seed split across two directory
levels instead (`pathmnist/seed42/manifest.json`), no `find_dir_containing`
depth search recovers it -- the trailing segment `pathmnist_seed42` simply
does not exist on disk as one name. `find_cellvit_cache` therefore does not
use `find_dir_containing` at all: it globs for every `manifest.json` under the
search roots (arbitrarily deep, directory names ignored entirely) and reads
`dataset`/`seed` back out of each manifest's own JSON content, which
`scripts/extract_cellvit_features.py` always writes. This is strictly more
reliable than inferring dataset/seed from path segments, since it cannot be
fooled by a re-upload that changes directory layout without touching what the
manifest itself says.

This module is the READ side only. Archive WRITING lives in `utils/archive.py`
instead: every publishing notebook (both extraction notebooks,
`run_al_baseline.ipynb`, `run_al_main.ipynb`, `extract_vlm_features.ipynb`)
writes to `/kaggle/working` top-level and deletes the loose cache afterward --
no terminal exists in a "Save & Run All" session, so the zip must be the only
thing left for the Output tab to show. `kaggle.py` is the READ side,
`archive.py` is the WRITE side; keeping them separate mirrors that split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

__all__ = [
    "find_dir_containing",
    "find_data_root",
    "find_visual_cache",
    "find_vlm_cache",
    "find_cellvit_cache",
]

# The two places a published Kaggle Dataset can end up mounted, checked in
# order. Kept here so every notebook searches the same candidates rather than
# each hard-coding its own guess.
_DEFAULT_SEARCH_ROOTS = (
    Path("/kaggle/input/datasets/cryandrrich/nckh2026"),
    Path("/kaggle/input/nckh2026"),
)


def find_dir_containing(
    probe: str,
    hint: Optional[str] = None,
    roots: Sequence[Path] = (),
    max_depth: int = 3,
) -> Optional[Path]:
    """Return the directory `D` such that `D / probe` exists.

    `probe` is a relative path (e.g. `"pathmnist_seed42/manifest.json"`) that
    only exists once the right directory has been found. Every depth from 0 to
    `max_depth` is tried under each root in turn, so a cache mounted one level
    deeper than expected is still found and its actual location printed by the
    caller -- never assumed.
    """
    probe_path = Path(probe)
    up = len(probe_path.parts) - 1
    search_roots: List[Path] = []
    if hint:
        search_roots.append(Path(hint))
    search_roots.extend(roots)
    search_roots.extend(_DEFAULT_SEARCH_ROOTS)
    search_roots.append(Path("/kaggle/input"))

    for root in search_roots:
        if not root.exists():
            continue
        for depth in range(max_depth + 1):
            pattern = "/".join(["*"] * depth + list(probe_path.parts))
            for hit in sorted(root.glob(pattern)):
                return hit.parents[up]
    return None


def find_data_root(candidates: Sequence[Path] = ()) -> Path:
    """First existing directory among the dataset-image mount candidates.

    Caller-supplied candidates are tried FIRST and the standard Kaggle mount
    points after them, rather than instead of them: a notebook naming its own
    dataset should not thereby lose the fallback that finds a remounted slug.
    Duplicates are dropped so a candidate that repeats a default is not probed
    twice.

    Falls back to the first candidate (which may not exist yet) so a caller
    gets a concrete path to report in an assertion message rather than `None`.
    """
    search: List[Path] = []
    for path in [*candidates, *_DEFAULT_SEARCH_ROOTS]:
        if path not in search:
            search.append(path)
    return next((path for path in search if path.exists()), search[0])


def find_visual_cache(
    dataset: str,
    seed: int,
    backbone: str,
    hint: Optional[str] = None,
) -> Optional[Path]:
    """Directory containing `{dataset}_seed{seed}_{backbone}_train.npy` **and**
    its manifest -- the naming convention `features/visual.py` writes.

    A cache missing its manifest is not returned: row alignment cannot be
    verified without it, and `features/visual.py::get_or_extract_features`
    already refuses such a cache on its own. Returning it here anyway would
    just move the same rejection to a more confusing place.
    """
    safe_backbone = backbone.replace("/", "_")
    base = f"{dataset}_seed{seed}_{safe_backbone}"
    found = find_dir_containing(f"{base}_train.npy", hint=hint)
    if found is None:
        return None
    if not (found / f"{base}_manifest.json").is_file():
        return None
    return found


def find_vlm_cache(
    dataset: str,
    seed: int,
    vlm_name: str,
    hint: Optional[str] = None,
) -> Optional[Path]:
    """Directory containing `{dataset}_seed{seed}_{vlm_safe}_train.npy` **and**
    its manifest -- the naming convention `features/vlm.py::vlm_feature_cache_paths`
    writes. Same shape as `find_visual_cache`; kept separate rather than shared
    because the VLM cache has a second manifest (`_proj_manifest.json`) the
    DINOv2 cache does not, and this must not silently accept one without it.

    `vlm_name` is sanitized the same way `features/vlm.py::_safe_name` does
    (`/` and `:` both replaced), since a VLM name like `MahmoodLab/CONCH`
    cannot appear in a filename as-is.
    """
    safe_vlm = vlm_name.replace("/", "_").replace(":", "_")
    base = f"{dataset}_seed{seed}_{safe_vlm}"
    found = find_dir_containing(f"{base}_train.npy", hint=hint)
    if found is None:
        return None
    if not (found / f"{base}_manifest.json").is_file():
        return None
    if not (found / f"{base}_proj_manifest.json").is_file():
        return None
    return found


def find_cellvit_cache(dataset: str, seed: int, hint: Optional[str] = None) -> Optional[Path]:
    """Directory `D` such that `D / f"{dataset}_seed{seed}"` holds the CellViT
    cache -- the parent `main.py::_load_cell_view` joins that name onto.

    Finds every `manifest.json` under the search roots (any depth, directory
    names ignored) and keeps the one whose own JSON content names this
    `dataset`/`seed` -- see the module docstring for why content, not path
    segments. `{dataset}_seed{seed}` need not exist as a real directory: when
    the cache was instead uploaded as `<dataset>/seed<seed>/manifest.json`
    (dataset and seed split across two levels), a symlink named
    `{dataset}_seed{seed}` pointing at that real directory is created under a
    fresh temp dir and that temp dir returned, so `main.py`'s own
    `os.path.join` still resolves to the right cache without main.py knowing
    the upload used a different layout.
    """
    search_roots: List[Path] = []
    if hint:
        search_roots.append(Path(hint))
    search_roots.extend(_DEFAULT_SEARCH_ROOTS)
    search_roots.append(Path("/kaggle/input"))

    canonical_name = f"{dataset}_seed{seed}"
    for root in search_roots:
        if not root.exists():
            continue
        for manifest_path in sorted(root.rglob("manifest.json")):
            try:
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("dataset") != dataset or manifest.get("seed") != seed:
                continue
            cache_dir = manifest_path.parent
            if cache_dir.name == canonical_name:
                return cache_dir.parent
            return _canonical_cellvit_parent(cache_dir, canonical_name)
    return None


def _canonical_cellvit_parent(cache_dir: Path, canonical_name: str) -> Path:
    """Symlink farm so a non-canonically-named cache directory still resolves
    under `os.path.join(parent, canonical_name)`, without touching the
    (possibly read-only, Kaggle-input-mounted) cache directory itself."""
    import tempfile

    parent = Path(tempfile.mkdtemp(prefix="cellvit_cache_"))
    link = parent / canonical_name
    if not link.exists():
        link.symlink_to(cache_dir, target_is_directory=True)
    return parent
