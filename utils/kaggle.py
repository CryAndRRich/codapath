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

This module is the READ side only. Archive WRITING is deliberately not here:
`extract_visual_features.ipynb` and `extract_nucleus_features.ipynb` already
write to `/kaggle/working` top-level and delete the loose cache afterward (no
terminal exists in a "Save & Run All" session, so the zip must be the only
thing left for the Output tab to show), while `run_al_sampler.ipynb` writes to
an `archive/` subdirectory and keeps the originals. Those two patterns differ
on purpose for different constraints, and unifying them is a separate decision
this module should not make silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

__all__ = [
    "find_dir_containing",
    "find_data_root",
    "find_visual_cache",
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

    Falls back to the first candidate (which may not exist yet) so a caller
    gets a concrete path to report in an assertion message rather than `None`.
    """
    search = list(candidates) or list(_DEFAULT_SEARCH_ROOTS)
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


def find_cellvit_cache(dataset: str, seed: int, hint: Optional[str] = None) -> Optional[Path]:
    """Directory containing `{dataset}_seed{seed}/manifest.json` -- the layout
    `scripts/extract_cellvit_features.py` writes."""
    return find_dir_containing(f"{dataset}_seed{seed}/manifest.json", hint=hint)
