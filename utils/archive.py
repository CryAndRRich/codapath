"""Naming for the archives the extraction notebooks publish.

A cache archive is only useful if its name says which cache it holds. Two
archives that differ only in backbone, checkpoint, seed or cell cap are not
interchangeable — mounting the wrong one produces a run that completes and is
wrong — so the name carries every axis that makes them differ, and the Kaggle
slug derived from it cannot collide even after truncation.
"""

from __future__ import annotations

import hashlib

# Kaggle dataset slugs are limited; keep well inside it.
SLUG_LIMIT = 50
_DIGEST_LENGTH = 6


def slugify(stem: str, limit: int = SLUG_LIMIT) -> str:
    """Return a lowercase-hyphen slug of `stem`, unique even when truncated.

    A plain `stem[:limit]` is the trap here: `..._dinov2-base` and
    `..._dinov2-large` differ only in their tail, so truncation maps two
    different caches onto one slug, and the second `kaggle datasets create`
    either fails or silently versions over the first. Appending a digest of the
    FULL stem keeps the head readable and the result distinct.
    """
    if limit < _DIGEST_LENGTH + 2:
        raise ValueError(f"limit must leave room for a digest, got {limit}")
    slug = stem.replace("_", "-").replace(".", "-").lower()
    slug = "-".join(part for part in slug.split("-") if part)
    if len(slug) <= limit:
        return slug
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
    head = slug[: limit - _DIGEST_LENGTH - 1].rstrip("-")
    return f"{head}-{digest}"


def visual_archive_stem(dataset: str, seed: int, vit_name: str) -> str:
    """Archive name for a DINOv2 visual-feature cache.

    Seed is in the name because ImageFolder splits are drawn from a seeded
    generator, so a cache is only valid for the seed that built it; the backbone
    is in the name because the sampler resolves caches by filename.

    One dataset and one seed, not lists: a notebook run produces exactly one
    archive, so a name that could describe several would be describing
    something the run cannot produce.
    """
    return f"visual-dinov2_{dataset}_seed{seed}_{vit_name.replace('/', '_')}"


def results_archive_stem(dataset: str, sampler: str, seed: int) -> str:
    """Archive name for one sampler's AL results on one dataset.

    Same three axes that make two result sets non-interchangeable, in the same
    order the cache stems use: what was run on, what ran, and which split.

    One seed, not a list. A run that swept several configurations would produce
    one archive covering all of them, and the name could not say which result
    inside belonged to which configuration -- so the sweep lives in repeated
    notebook runs, and each one names itself completely.
    """
    return f"{dataset}_{sampler}_seed{seed}"


def vlm_archive_stem(dataset: str, seed: int, vlm_name: str, style: str) -> str:
    """Archive name for a VLM (e.g. CONCH) feature + text-prototype cache.

    `style` is in the name because it is NOT a tuning knob on top of a fixed
    cache -- it names a different text prototype, computed from a different
    description file, living in a different `{dataset}_{style}_text.npy`. Two
    styles are two artifacts, not two configurations of one artifact, so
    archiving one under a name that could mean either would make the wrong
    one indistinguishable from the right one.
    """
    return f"vlm_{dataset}_seed{seed}_{vlm_name.replace('/', '_')}_{style}"


def nucleus_archive_stem(
    dataset: str,
    seed: int,
    checkpoint_stem: str,
    dino_name: str,
    max_cells_per_patch=None,
) -> str:
    """Archive name for a CellViT nucleus cache.

    The cell cap belongs in the name: it changes which cells exist in the cache
    at all, and every variant of an experiment must share one cap to stay
    comparable.
    """
    cap_tag = "all" if max_cells_per_patch is None else f"max{max_cells_per_patch}"
    return (
        f"cellvit-nucleus_{dataset}_seed{seed}_{checkpoint_stem}_"
        f"{dino_name.replace('/', '_')}_{cap_tag}"
    )
