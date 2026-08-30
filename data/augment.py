"""Pixel-space augmentation for the final-training pass.

**Only `flip_rotate` -- deliberately no color jitter.** H&E-stained
histopathology tiles carry diagnostic information in their color (stain
intensity correlates with tissue state), and this project has already lost a
method to a stain-shortcut failure from exactly this kind of augmentation
(see CLAUDE.md's deleted-methods note: "the stain-shortcut SCALPEL v9").
Flip and 90-degree rotation are label-preserving for a tile with no fixed
orientation (a patch of tissue does not have an "up"), so they add
positional invariance without touching color statistics at all.

This module builds a `torchvision.transforms` pipeline meant to compose
AFTER `default_transform()`'s resize (or CONCH's own `preprocess`), never
instead of it -- `training/finetune.py` applies the encoder's own transform
first, then this, so augmentation always sees the same resolution/
normalization the frozen backbone was calibrated on.
"""

from __future__ import annotations

import torchvision.transforms as transforms

__all__ = ["AUGMENT_KINDS", "build_augment_transform"]

AUGMENT_KINDS = ("none", "flip_rotate")


def build_augment_transform(kind: str):
    """Return a `transforms.Compose` (possibly empty) for `kind`.

    `"none"` returns an empty `Compose` (identity) rather than `None`, so a
    caller can always do `transforms.Compose([base_transform, augment])`
    without a branch for the no-augment case.
    """
    if kind not in AUGMENT_KINDS:
        raise ValueError(f"AUGMENT must be one of {AUGMENT_KINDS}, got {kind!r}")
    if kind == "none":
        return transforms.Compose([])
    # RandomHorizontalFlip/RandomVerticalFlip together cover all 4 flip
    # states; angle in {0, 90, 180, 270} via a fixed-choice rotation keeps
    # every rotation exact (no interpolation artifact a continuous-angle
    # rotation would introduce, and none of the tissue-boundary padding a
    # non-multiple-of-90 rotation would need).
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomChoice([
            transforms.RandomRotation(degrees=(0, 0)),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.RandomRotation(degrees=(180, 180)),
            transforms.RandomRotation(degrees=(270, 270)),
        ]),
    ])
