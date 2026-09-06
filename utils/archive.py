"""Naming for the archives the extraction notebooks publish.

A cache archive is only useful if its name says which cache it holds. Two
archives that differ only in backbone, checkpoint, seed or cell cap are not
interchangeable — mounting the wrong one produces a run that completes and is
wrong — so the name carries every axis that makes them differ, and the Kaggle
slug derived from it cannot collide even after truncation.
"""

from __future__ import annotations

import hashlib
from typing import Optional

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


def main_archive_stem(
    dataset: str,
    sampler: str,
    seed: int,
    encoder: str = "dinov2",
    run_name: Optional[str] = None,
) -> str:
    """Archive name for `run_al_main.ipynb`'s AL results.

    Same shape as `results_archive_stem`, plus `encoder` -- the axis
    `results_archive_stem` never needed because every baseline runs on
    DINOv2 only. A DINOv2 `pact` run and a CONCH `pact` run of the same
    config are two different protocols (different feature space at every
    stage: coverage kernel, disagreement probes, evaluation probe --
    so they must not share an archive name. Only
    appended when non-default, so an unchanged DINOv2 run's archive name
    matches what `results_archive_stem` would have produced -- mirroring how
    `_default_run_name` only appends `encoder` when it is not `"dinov2"`.

    `run_name`, when given, is `main._default_run_name`'s output and is used
    INSTEAD of `sampler` -- because that is the string that already encodes
    every variant axis of the run (cell_pooling, use_lora, aux_loss, augment,
    ...). Without it a 48-combination sweep produces 48 zips all called
    `pathmnist_pact_seed42.zip`: each download overwrites the last in the
    browser, and each Kaggle Dataset publish versions over the previous one.
    The `sampler` name is already a prefix of `run_name`, so passing it does
    not lose information -- it adds the axes that distinguish the runs.
    """
    stem = f"{dataset}_{run_name or sampler}_seed{seed}"
    # `run_name` comes from `main._default_run_name`, which ALREADY appends the
    # encoder when it is not the default -- so appending it again here produced
    # `..._pact_disagreement_conch_seed42_conch`. The suffix is only needed
    # when the caller passed a bare sampler name instead.
    suffix = encoder.replace("/", "_")
    if encoder != "dinov2" and not stem.endswith(f"_{suffix}_seed{seed}"):
        stem = f"{stem}_{suffix}"
    return stem


def vlm_archive_stem(dataset: str, seed: int, vlm_name: str, styles) -> str:
    """Archive name for a VLM (e.g. CONCH) feature + text-prototype cache.

    Unlike every other stem here, `styles` is a LIST -- and unlike every other
    axis in this module, that is not a sweep. CONCH's vision tower never sees a
    prompt, so the image features are identical for every style and one
    extraction serves all of them; encoding a style's text afterwards is a
    handful of short strings. One run therefore produces one archive holding
    every style, each in its own `{dataset}_{style}_text.npy`, and the name has
    to say which ones are inside.

    The styles are sorted so that the same set always produces the same name
    regardless of the order they were listed in -- otherwise re-running with the
    list reordered would publish a second archive holding identical data.

    A single style still reads as plain `..._llm_short`, so an archive built
    before this took a list keeps its name.
    """
    if isinstance(styles, str):
        styles = [styles]
    if not styles:
        raise ValueError("at least one description style is required")
    return (
        f"vlm_{dataset}_seed{seed}_{vlm_name.replace('/', '_')}"
        f"_{'-'.join(sorted(styles))}"
    )


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
