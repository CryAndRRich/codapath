"""CONCH (or another pathology VLM) as a second, optional image encoder.

Verified 2026-08-29 against `repos/CONCH` (clone of `mahmoodlab/CONCH`) and
`pdfs/CONCH_2307.12914.pdf` directly — every claim below is read from the
actual model code, not from the README or the paper's prose alone.

**Two embedding spaces, and they are not interchangeable** (`coca_model.py`,
`CoCa.encode_image`):

    encode_image(x, proj_contrast=False, normalize=False)   512-d, pre-projection
        -> linear probe, coverage kernel, everything pact's disagreement
           machinery reads
    encode_image(x, proj_contrast=True,  normalize=True)    512-d, projected + L2-normed
        -> comparing an image against text (the round-1 cold-start prior)

The defaults on `encode_image` are `normalize=True, proj_contrast=True` --
i.e. the TEXT-COMPARISON space, not the probe space. Getting the probe space
requires passing both kwargs explicitly, which is exactly the mistake this
module exists to make impossible: `extract_vlm_features` always requests both
spaces by name (`RAW_SPACE` / `PROJ_SPACE` below) and writes both, in the same
forward pass, so a caller can never get one when it silently needed the other.

**Resolution is 448x448**, not 224 (`conch_ViT-B-16.json`). PathMNIST's own
images are 224, so `preprocess` upscales 2x -- that is what produced the
paper's 79.1% zero-shot number on CRC100K, and it is what `create_model_from_pretrained`
returns by default. This module never builds its own resize/crop pipeline; it
always uses the `preprocess` object the factory returns.

**Normalization is OpenAI CLIP stats, not ImageNet** (`factory.py::create_model`
overwrites `model.visual.image_mean/std` after loading the checkpoint). Never
hand-roll `transforms.Normalize` for this model.

**Tokenizer is CONCH's own** (`conch_byte_level_bpe_uncased.json`, vocab 32007),
not `open_clip`'s. `pip install open_clip_torch` is not enough; the `conch`
package itself must be installed.

**`logit_scale` is a LEARNED parameter**, used as `softmax(logits *
model.logit_scale.exp())` in the official zero-shot code
(`downstream/zeroshot_path.py`). Never hard-code a temperature.

The `conch` package is not installed in this environment (nor is `open_clip`,
which it wraps), so every function here that touches an actual model imports
`conch` lazily, inside the function body -- exactly the pattern
`main.py::_load_cell_view` already uses for the optional `cellvit` package.
That keeps this module importable, and its pure logic (cache naming, manifest
shape, the official prompt ensemble math, class-order validation) testable,
without the package installed.

## A second family: QuiltNet (added 2026-09-15)

`image_encoder="quilt"` runs the same pipeline on **QuiltNet-B-16-PMB**
(`wisdomik/QuiltNet-B-16-PMB`), so the text-prior rows are not a property of
one checkpoint. It is an `open_clip` **CustomTextCLIP**: a timm ViT-B/16 image
tower (`visual.trunk`, 768-d) with a `visual.head` projection to 512-d, paired
with a **PubMedBERT** text tower. The repo ships `open_clip_pytorch_model.bin`
and no HF-format weights, and its `config.json` names a `BertCLIPModel` with
no `auto_map`, so it loads through `open_clip`, never `transformers`.

**Why this checkpoint and not PLIP.** The reported text rows run the
`llm_morphology` descriptions, which measure 100-130 PubMedBERT tokens.
QuiltNet's text context is **256**, so every description fits whole (measured:
0 of 39 truncated across all three datasets). A 77-token CLIP tower -- PLIP,
or QuiltNet-B-32 -- truncates **every one of them** (measured: 39 of 39, with
only ~53-61% of tokens surviving), which would make the second VLM's row
measure a different prior than the CONCH rows rather than the same prior on a
different model.

**The two families are told apart by the VISION tower, in both directions.**
`extract_vlm_image_features` and `encode_text_prototypes` dispatch on the same
attributes, so they cannot disagree about what a model is. Both tests are
POSITIVE (CONCH: `visual.use_attentional_pool_contrast`; QuiltNet:
`visual.trunk` + `visual.head`) rather than "not the other one" -- a negative
test routes anything unrecognised, including a checkpoint that failed to load,
into the wrong branch and raises about the wrong thing.

RAW/PROJ for QuiltNet is `visual.trunk(x)` then `visual.head(raw)`, because
`TimmModel.forward` is exactly `head(trunk(x))` -- so PROJ is derived from RAW
without a second trunk pass, the same saving the CONCH path makes. Verified
against the real checkpoint: PROJ matches `model.encode_image` to 7.8e-08.
Note the widths DIFFER here (768 vs 512), unlike CONCH where both spaces are
512 and a wrong space is invisible to any shape check.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "RAW_SPACE",
    "PROJ_SPACE",
    "vlm_feature_cache_paths",
    "text_prototype_cache_paths",
    "load_conch",
    "load_quilt",
    "load_vlm",
    "vlm_family",
    "CONCH_FAMILY",
    "QUILT_FAMILY",
    "extract_vlm_image_features",
    "get_or_extract_vlm_features",
    "extract_vlm_features_shard",
    "assemble_vlm_feature_shards",
    "encode_text_prototypes",
    "zero_shot_logits",
]

# The two spaces by name, used consistently as dict keys and file-name
# fragments so "which space is this" is never inferred from context.
RAW_SPACE = "raw"    # proj_contrast=False, normalize=False -- for the probe
PROJ_SPACE = "proj"  # proj_contrast=True,  normalize=True  -- for text comparison

# Which loader/extractor a checkpoint needs. This is a decision about MODEL
# ARCHITECTURE, not about the weights: CONCH is CoCa (a custom `conch`
# package, `visual.forward_no_head` + `proj_contrast`), QuiltNet is an
# `open_clip` CustomTextCLIP whose vision tower is a TimmModel
# (`visual.trunk` + `visual.head`). Neither has the other's attributes, so a
# single code path cannot serve both and a wrong guess raises
# `AttributeError` deep inside a forward pass rather than here.
# Filled by `_encode_text_prototypes_open_clip` on every call: how much of each
# prompt this text tower actually read. `None` until a CLIP-family tower has
# encoded something (CONCH's 128-token limit fits every style in this project,
# so its path leaves this alone). The notebook writes it into the text
# manifest, because "which prior was this row measured under" is not
# recoverable from the prototypes afterwards.
LAST_TEXT_TRUNCATION: Optional[Dict[str, object]] = None

CONCH_FAMILY = "conch"
QUILT_FAMILY = "quilt"

# Substrings that identify a family in a checkpoint name. Matched on the
# lowercased name so "wisdomik/QuiltNet-B-16-PMB" and a lowercased spelling
# behave the same.
_FAMILY_MARKERS = (
    (CONCH_FAMILY, ("conch",)),
    (QUILT_FAMILY, ("quilt",)),
)


def vlm_family(vlm_name: str) -> str:
    """Which family `vlm_name` belongs to, or raise naming the supported set.

    Deliberately a hard failure rather than a default: defaulting to CONCH
    would send a QuiltNet checkpoint into `create_model_from_pretrained
    ("conch_ViT-B-16", ...)`, which fails with a shape/key error about a
    config file the user never chose. Every caller that loads a model routes
    through here, so adding a third VLM is one entry in `_FAMILY_MARKERS`
    plus its loader and extractor branch.
    """
    lowered = vlm_name.lower()
    for family, markers in _FAMILY_MARKERS:
        if any(marker in lowered for marker in markers):
            return family
    raise ValueError(
        f"Cannot tell which VLM family {vlm_name!r} belongs to. Supported: "
        f"{CONCH_FAMILY} (e.g. 'MahmoodLab/CONCH'), {QUILT_FAMILY} (e.g. "
        f"'wisdomik/QuiltNet-B-16-PMB'). Add a marker to "
        "features/vlm.py::_FAMILY_MARKERS together with a loader and an "
        "extractor branch."
    )


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def vlm_feature_cache_paths(cache_dir: str, dataset_key: str, seed: int, vlm_name: str) -> Dict[str, str]:
    """Filenames for a VLM's per-image feature cache, both spaces.

    Mirrors `features/visual.py::_feature_cache_paths` exactly (same
    dataset+seed+backbone key, same `_train`/`_test`/`_manifest` suffixes) so
    the naming convention this project already relies on elsewhere is not
    reinvented -- only the two extra `_proj_*` members are new, because a VLM
    (unlike DINOv2) has two embedding spaces worth caching.
    """
    base = f"{dataset_key}_seed{seed}_{_safe_name(vlm_name)}"
    return {
        "train": os.path.join(cache_dir, f"{base}_train.npy"),
        "test": os.path.join(cache_dir, f"{base}_test.npy"),
        "proj_train": os.path.join(cache_dir, f"{base}_proj_train.npy"),
        "proj_test": os.path.join(cache_dir, f"{base}_proj_test.npy"),
        "manifest": os.path.join(cache_dir, f"{base}_manifest.json"),
        "proj_manifest": os.path.join(cache_dir, f"{base}_proj_manifest.json"),
    }


def text_prototype_cache_paths(cache_dir: str, dataset_key: str, style: str) -> Dict[str, str]:
    """Filenames for one (dataset, description style)'s text prototypes.

    Not keyed by seed or VLM name: the text side does not depend on the train/
    test split, and this project has one VLM (CONCH) at a time. If a second VLM
    is ever added, `style` alone stops being a unique key and this must change
    -- deliberately not solved preemptively.
    """
    base = f"{dataset_key}_{style}"
    return {
        "prototypes": os.path.join(cache_dir, f"{base}_text.npy"),
        "manifest": os.path.join(cache_dir, f"{base}_text_manifest.json"),
    }


def _atomic_save_npy(path: str, array: np.ndarray) -> None:
    """Write-then-rename, matching `features/visual.py`'s cache-write pattern:
    two workers racing on a miss must never leave a half-written array that
    still loads."""
    temporary = f"{path}.tmp{os.getpid()}"
    np.save(temporary, array)
    os.replace(temporary if temporary.endswith(".npy") else temporary + ".npy", path)


def _atomic_save_json(path: str, payload: dict) -> None:
    temporary = f"{path}.tmp{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


# --------------------------------------------------------------------------
# Model loading and raw extraction -- touches the `conch` package.
# --------------------------------------------------------------------------

def load_conch(vlm_name: str, device: torch.device, hf_token: Optional[str] = None):
    """Load a CONCH-family model and its matching preprocessing transform.

    Returns `(model, preprocess)`. `preprocess` MUST be used as-is for every
    image this model ever sees -- it carries the 448x448 resize/crop and the
    OpenAI CLIP normalization the factory computed from the checkpoint, and
    hand-rolling an equivalent transform does not crash -- it silently shifts
    every embedding.
    """
    from conch.open_clip_custom import create_model_from_pretrained

    checkpoint = vlm_name if vlm_name.startswith("hf_hub:") else f"hf_hub:{vlm_name}"
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", checkpoint, hf_auth_token=hf_token or None, device=device,
    )
    model.eval()
    return model, preprocess


def load_quilt(vlm_name: str, device: torch.device, hf_token: Optional[str] = None):
    """Load a QuiltNet-family (`open_clip` CustomTextCLIP) checkpoint.

    QuiltNet-B-16-PMB (Ikezogwo et al., NeurIPS 2023, `wisdomik/QuiltNet-B-16-PMB`)
    is a timm ViT-B/16 image tower paired with a **PubMedBERT** text tower, so
    it is neither CONCH's CoCa nor a stock HF `CLIPModel`: the repo ships
    `open_clip_pytorch_model.bin` and no HF-format weights, and its
    `config.json` names a `BertCLIPModel` with no `auto_map`. It therefore
    loads through `open_clip`, not `transformers`.

    The reason this checkpoint is used at all is its **256-token** text
    context. `llm_morphology` descriptions run 100-130 PubMedBERT tokens, so
    they fit whole -- where a 77-token CLIP tower (PLIP, QuiltNet-B-32) would
    truncate every one of them and measure a different prior than the CONCH
    rows do.

    Returns `(model, preprocess)` with the same contract as `load_conch`:
    `preprocess` comes from the checkpoint's own factory and must be used
    as-is (224x224, OpenAI CLIP normalization).

    `open_clip` is imported lazily, like `conch`, so this module stays
    importable without it.
    """
    import open_clip

    checkpoint = vlm_name if vlm_name.startswith("hf-hub:") else f"hf-hub:{vlm_name}"
    model, _, preprocess = open_clip.create_model_and_transforms(checkpoint)
    model.eval()
    model = model.to(device)
    return model, preprocess


def load_vlm(vlm_name: str, device: torch.device, hf_token: Optional[str] = None):
    """Load any supported VLM, dispatching on `vlm_family(vlm_name)`.

    Prefer this over `load_conch`/`load_quilt` in code that is meant to work
    for more than one checkpoint (the extraction worker, the notebook). The
    family-specific loaders stay public because the preflight cells assert on
    family-specific attributes.
    """
    family = vlm_family(vlm_name)
    if family == CONCH_FAMILY:
        return load_conch(vlm_name, device, hf_token=hf_token)
    return load_quilt(vlm_name, device, hf_token=hf_token)


@torch.inference_mode()
def extract_vlm_image_features(
    dataloader,
    model,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """ONE trunk pass per batch, both embedding spaces derived from it.

    Returns `(raw, proj)`: `raw` is `proj_contrast=False, normalize=False`
    (768 for CONCH's captioning head width is NOT this -- it is the 512-d
    pre-projection vector `forward_no_head` returns, matching `embed_dim` in
    `conch_ViT-B-16.json`); `proj` is `proj_contrast=True, normalize=True`.

    **This calls `model.visual.forward_no_head` ONCE per batch, not
    `model.encode_image` twice.** An earlier version called `encode_image`
    for RAW and again for PROJ, and its docstring claimed this was "the SAME
    forward pass" -- it was not: `CoCa.encode_image` re-enters
    `self.visual(images)` (`forward_no_head` or the full `forward`) on every
    call, so that version ran the 12-layer ViT-B trunk on 448x448 pixels
    TWICE per batch, the single most expensive step in this notebook.

    Reading `conch/open_clip_custom/vision_tower.py::VisualModel.forward` and
    `.forward_no_head` side by side shows they share every step up to the
    pooled, un-normalized, un-projected vector (`forward_no_head`'s return
    value): `trunk(x)` -> `attn_pool_contrast(x)[:, 0]` -> `ln_contrast(...)`.
    `forward` (PROJ_SPACE) then does exactly one more matmul,
    `pooled @ proj_contrast`, followed by L2-normalize. So PROJ is derived
    from RAW here instead of recomputed -- `raw @ model.visual.proj_contrast`
    then `F.normalize`, using the model's OWN weight rather than a
    reimplementation, which is why this is bit-exact (verified against a real
    `_build_vision_tower` to 0.0 absolute difference in both spaces) and not
    an approximation.

    This depends on `conch_ViT-B-16.json`'s `attentional_pool_contrast: true`
    (the only vendored config, and the one this project uses) taking the
    `attn_pool_contrast` branch in both methods; `use_attentional_pool_contrast`
    is checked below and this function raises rather than silently falling
    back to the slow path for a config where the assumption does not hold --
    a silent fallback would look like a working run at a cost nobody asked
    for.
    """
    import torch.nn.functional as F
    from tqdm import tqdm

    model = model.to(device)
    model.eval()

    # Which pair of "one trunk pass -> RAW, then one projection -> PROJ" steps
    # to run. Decided once, outside the loop, and by ARCHITECTURE rather than
    # by the checkpoint name: `visual.trunk`/`visual.head` is open_clip's
    # TimmModel shape (QuiltNet), `visual.forward_no_head`/`proj_contrast` is
    # CONCH's CoCa shape. Neither has the other's attributes, so a wrong
    # branch raises here instead of producing a plausible wrong array.
    visual = getattr(model, "visual", None)
    is_conch = visual is not None and getattr(visual, "use_attentional_pool_contrast", False)
    # open_clip's TimmModel: `forward` is exactly `head(trunk(x))`, so the
    # trunk output IS the pre-projection space and the head IS the projection.
    is_quilt = (
        not is_conch
        and visual is not None
        and hasattr(visual, "trunk")
        and hasattr(visual, "head")
    )

    if not is_conch and not is_quilt:
        raise AttributeError(
            "extract_vlm_image_features supports two vision-tower shapes: CONCH's "
            "(model.visual.use_attentional_pool_contrast True, so RAW and PROJ share "
            "one trunk pass via forward_no_head + proj_contrast) and open_clip's "
            "TimmModel (model.visual.trunk + model.visual.head, as QuiltNet uses). "
            "This checkpoint has neither -- extend this function for it rather than "
            "silently falling back to two full forward passes."
        )

    raw_batches: List[np.ndarray] = []
    proj_batches: List[np.ndarray] = []
    for images, _ in tqdm(dataloader, desc="Extracting VLM features", leave=False):
        images = images.to(device, non_blocking=True)
        if is_conch:
            # RAW_SPACE: exactly what encode_image(proj_contrast=False, normalize=False)
            # returns -- the trunk + attention-pool + layernorm, nothing more.
            raw = visual.forward_no_head(images, normalize=False)
            # PROJ_SPACE: the one additional step encode_image(proj_contrast=True,
            # normalize=True) takes past that same `raw` vector -- see the
            # docstring above for the exact line-by-line correspondence.
            proj = F.normalize(raw @ visual.proj_contrast, dim=-1)
        else:
            # RAW_SPACE for open_clip's TimmModel: the TRUNK output, i.e. the
            # pooled pre-projection vector (768-d on QuiltNet's ViT-B/16).
            # This is the analogue of CONCH's pre-projection space and what a
            # linear probe should train on.
            raw = visual.trunk(images)
            # PROJ_SPACE: `TimmModel.forward` is exactly `head(trunk(x))`
            # (verified against open_clip's own source), so applying the head
            # to `raw` reproduces `encode_image` without a second trunk pass.
            # L2-normalized to match CONCH's normalize=True, which
            # `zero_shot_logits` assumes.
            proj = F.normalize(visual.head(raw), dim=-1)
        raw_batches.append(raw.cpu().numpy().astype(np.float32))
        proj_batches.append(proj.cpu().numpy().astype(np.float32))

    return np.vstack(raw_batches), np.vstack(proj_batches)


def get_or_extract_vlm_features(
    train_loader,
    test_loader,
    dataset_key: str,
    seed: int,
    vlm_name: str,
    device: torch.device,
    cache_dir: str = "vlm_features",
    train_fingerprint: Optional[str] = None,
    test_fingerprint: Optional[str] = None,
    model=None,
    hf_token: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Return `{"train", "test", "proj_train", "proj_test"}` for a VLM.

    Same cache-validity contract as `features/visual.py::get_or_extract_features`
    (dataset + seed + backbone + fingerprint must all agree, manifest written
    last) with one addition: BOTH manifests (`_manifest.json` and
    `_proj_manifest.json`) must be present and valid, because a partially
    cached extraction -- the raw space written but not the projected one, from
    an interrupted run -- must re-extract rather than silently return a
    mismatched pair.

    `model`, if given, is used as-is on a cache miss instead of loading a new
    one. Unlike DINOv2 (one fixed 224+ImageNet transform, so `main.py` never
    needs the model before knowing there is a cache miss), building a VLM's
    OWN dataloader requires its `preprocess`, which only exists after calling
    `load_conch` -- so the typical caller already has a loaded model in hand
    by the time this runs, and passing it here avoids loading the checkpoint
    a second time. `model=None` falls back to `load_conch(vlm_name, ...)`,
    which is what a cache HIT never pays for either way: the load only
    happens inside the miss branch below.
    """
    paths = vlm_feature_cache_paths(cache_dir, dataset_key, seed, vlm_name)
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)

    if all(os.path.exists(paths[key]) for key in ("train", "test", "proj_train", "proj_test")):
        cached = {key: np.load(paths[key]) for key in ("train", "test", "proj_train", "proj_test")}
        shapes_ok = (
            cached["train"].shape[0] == n_train
            and cached["test"].shape[0] == n_test
            and cached["proj_train"].shape[0] == n_train
            and cached["proj_test"].shape[0] == n_test
        )
        manifests_ok = False
        if shapes_ok and os.path.exists(paths["manifest"]) and os.path.exists(paths["proj_manifest"]):
            with open(paths["manifest"], "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            with open(paths["proj_manifest"], "r", encoding="utf-8") as handle:
                proj_manifest = json.load(handle)
            manifests_ok = (
                manifest.get("dataset") == dataset_key
                and manifest.get("seed") == seed
                and manifest.get("backbone") == vlm_name
                and manifest.get("space") == RAW_SPACE
                and proj_manifest.get("space") == PROJ_SPACE
                and (train_fingerprint is None or manifest.get("train_fingerprint") == train_fingerprint)
                and (test_fingerprint is None or manifest.get("test_fingerprint") == test_fingerprint)
            )
        if manifests_ok:
            print(f"[vlm] Loaded cache -> {paths['train']} + {paths['proj_train']}")
            return cached
        print("[vlm] Cache metadata/order mismatch or partial -- re-extracting.")

    print(f"[vlm] Cache miss -- extracting {vlm_name} features for '{dataset_key}' (seed={seed}).")
    owns_model = model is None
    if owns_model:
        model, _ = load_vlm(vlm_name, device, hf_token=hf_token)
    train_raw, train_proj = extract_vlm_image_features(train_loader, model, device)
    test_raw, test_proj = extract_vlm_image_features(test_loader, model, device)
    if owns_model:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(cache_dir, exist_ok=True)
    _atomic_save_npy(paths["train"], train_raw)
    _atomic_save_npy(paths["test"], test_raw)
    _atomic_save_npy(paths["proj_train"], train_proj)
    _atomic_save_npy(paths["proj_test"], test_proj)
    # Manifests LAST -- what `get_or_extract_vlm_features` trusts on the next
    # call, same reasoning as `features/visual.py`.
    _atomic_save_json(paths["manifest"], {
        "schema_version": 1,
        "dataset": dataset_key,
        "seed": seed,
        "backbone": vlm_name,
        "space": RAW_SPACE,
        "train_fingerprint": train_fingerprint,
        "test_fingerprint": test_fingerprint,
        "train_shape": list(train_raw.shape),
        "test_shape": list(test_raw.shape),
    })
    _atomic_save_json(paths["proj_manifest"], {
        "schema_version": 1,
        "dataset": dataset_key,
        "seed": seed,
        "backbone": vlm_name,
        "space": PROJ_SPACE,
        "train_fingerprint": train_fingerprint,
        "test_fingerprint": test_fingerprint,
        "train_shape": list(train_proj.shape),
        "test_shape": list(test_proj.shape),
    })
    print(f"[vlm] Saved cache -> {paths['train']} + {paths['proj_train']}")
    return {"train": train_raw, "test": test_raw, "proj_train": train_proj, "proj_test": test_proj}


# --------------------------------------------------------------------------
# Sharded extraction -- one process per GPU on a Kaggle T4 x2 session.
# --------------------------------------------------------------------------
#
# Mirrors `features/visual.py`'s DINOv2 sharding, with ONE structural
# difference: a VLM produces TWO arrays per split (RAW_SPACE and PROJ_SPACE)
# from the same forward pass, so a shard file holds both. Splitting them
# across separate shard passes would double the 448x448 forward cost, which
# is the entire expense of this notebook.
#
# `_shard_bounds` is IMPORTED from features/visual.py rather than redefined:
# the contiguous-split rule (and its remainder handling) must be the single
# definition both extractors share, or a future edit to one silently produces
# a different partition than the other while every shape check still passes.


def _vlm_shard_dir(cache_dir: str, base: str, split: str, shard_count: int) -> str:
    return os.path.join(cache_dir, f".vlm_shards_{base}_{split}_of{shard_count}")


def extract_vlm_features_shard(
    train_loader,
    test_loader,
    dataset_key: str,
    seed: int,
    vlm_name: str,
    device: torch.device,
    shard_index: int,
    shard_count: int,
    cache_dir: str = "vlm_features",
    model=None,
    hf_token: Optional[str] = None,
) -> None:
    """Extract one contiguous row range of train AND test into a shard file.

    Each shard writes ONE `.npz` per split holding both spaces (`raw`, `proj`)
    -- not two `.npy` files -- so a shard is either complete for both spaces
    or absent. A half-written pair (raw present, proj missing) is exactly the
    partial state `get_or_extract_vlm_features` already refuses to trust, and
    at the shard level it would be assembled into a cache that loads and is
    wrong.

    A finished shard is skipped, so a session that dies half way through
    resumes instead of repeating a 448x448 pass it already paid for.

    The model is loaded ONCE per worker and reused across both splits. It is
    passed in by the caller when the caller already has one (the notebook
    needs `preprocess` to build the loaders, so it always does).
    """
    from features.visual import _shard_bounds

    if shard_count < 1:
        raise ValueError("shard_count must be positive")

    paths = vlm_feature_cache_paths(cache_dir, dataset_key, seed, vlm_name)
    if all(os.path.exists(paths[key]) for key in ("train", "test", "proj_train", "proj_test")):
        print(f"[vlm] shard {shard_index}: complete cache exists, nothing to do")
        return

    base = f"{dataset_key}_seed{seed}_{_safe_name(vlm_name)}"
    owns_model = model is None
    for split, loader in (("train", train_loader), ("test", test_loader)):
        shard_dir = _vlm_shard_dir(cache_dir, base, split, shard_count)
        os.makedirs(shard_dir, exist_ok=True)
        shard_path = os.path.join(shard_dir, f"{shard_index:03d}.npz")
        start, stop = _shard_bounds(len(loader.dataset), shard_index, shard_count)

        if os.path.exists(shard_path):
            with np.load(shard_path) as existing:
                rows_ok = (
                    "raw" in existing and "proj" in existing
                    and existing["raw"].shape[0] == stop - start
                    and existing["proj"].shape[0] == stop - start
                )
            if rows_ok:
                print(f"[vlm] shard {shard_index} {split}: reusing rows [{start}:{stop})")
                continue
            print(f"[vlm] shard {shard_index} {split}: stale/partial shard, recomputing")

        if model is None:
            model, _ = load_vlm(vlm_name, device, hf_token=hf_token)

        subset = torch.utils.data.Subset(loader.dataset, range(start, stop))
        shard_loader = torch.utils.data.DataLoader(
            subset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=True,
        )
        raw, proj = extract_vlm_image_features(shard_loader, model, device)
        if raw.shape[0] != stop - start:
            raise RuntimeError(
                f"shard {shard_index} {split}: extracted {raw.shape[0]} rows, "
                f"expected {stop - start}"
            )
        temporary = f"{shard_path}.tmp{os.getpid()}"
        np.savez(temporary, raw=raw, proj=proj)
        os.replace(temporary if temporary.endswith(".npz") else temporary + ".npz", shard_path)
        print(f"[vlm] shard {shard_index} {split}: wrote rows [{start}:{stop}) {raw.shape}")
        del raw, proj

    if owns_model and model is not None:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def assemble_vlm_feature_shards(
    dataset_key: str,
    seed: int,
    vlm_name: str,
    shard_count: int,
    n_train: int,
    n_test: int,
    cache_dir: str = "vlm_features",
    train_fingerprint: Optional[str] = None,
    test_fingerprint: Optional[str] = None,
    keep_shards: bool = False,
) -> Dict[str, np.ndarray]:
    """Concatenate per-GPU shards into the standard two-space cache.

    Both manifests are written LAST and only after every array is complete and
    its row count checks out -- the same rule the serial path follows, and the
    reason a reader can trust a manifest's existence as proof of a whole
    cache. A missing shard raises rather than concatenating a short array.
    """
    from features.visual import _shard_bounds

    paths = vlm_feature_cache_paths(cache_dir, dataset_key, seed, vlm_name)
    base = f"{dataset_key}_seed{seed}_{_safe_name(vlm_name)}"

    assembled: Dict[str, np.ndarray] = {}
    for split, expected_rows in (("train", n_train), ("test", n_test)):
        shard_dir = _vlm_shard_dir(cache_dir, base, split, shard_count)
        raw_parts, proj_parts = [], []
        for shard_index in range(shard_count):
            shard_path = os.path.join(shard_dir, f"{shard_index:03d}.npz")
            if not os.path.exists(shard_path):
                raise FileNotFoundError(
                    f"Missing {split} shard {shard_index} at {shard_path}. Every "
                    "shard must finish before assembly; re-run the extraction "
                    "cell to fill the gap."
                )
            start, stop = _shard_bounds(expected_rows, shard_index, shard_count)
            with np.load(shard_path) as part:
                raw, proj = part["raw"], part["proj"]
                if raw.shape[0] != stop - start or proj.shape[0] != stop - start:
                    raise RuntimeError(
                        f"{split} shard {shard_index} has {raw.shape[0]}/"
                        f"{proj.shape[0]} rows, expected {stop - start}"
                    )
                raw_parts.append(raw.copy())
                proj_parts.append(proj.copy())
        assembled[split] = np.vstack(raw_parts)
        assembled[f"proj_{split}"] = np.vstack(proj_parts)
        del raw_parts, proj_parts

    for key in ("train", "test", "proj_train", "proj_test"):
        _atomic_save_npy(paths[key], assembled[key])

    _atomic_save_json(paths["manifest"], {
        "schema_version": 1,
        "dataset": dataset_key,
        "seed": seed,
        "backbone": vlm_name,
        "space": RAW_SPACE,
        "train_fingerprint": train_fingerprint,
        "test_fingerprint": test_fingerprint,
        "train_shape": list(assembled["train"].shape),
        "test_shape": list(assembled["test"].shape),
        "shard_count": shard_count,
    })
    _atomic_save_json(paths["proj_manifest"], {
        "schema_version": 1,
        "dataset": dataset_key,
        "seed": seed,
        "backbone": vlm_name,
        "space": PROJ_SPACE,
        "train_fingerprint": train_fingerprint,
        "test_fingerprint": test_fingerprint,
        "train_shape": list(assembled["proj_train"].shape),
        "test_shape": list(assembled["proj_test"].shape),
        "shard_count": shard_count,
    })
    print(f"[vlm] Assembled {shard_count} shards -> {paths['train']} + {paths['proj_train']}")

    if not keep_shards:
        import shutil

        for split in ("train", "test"):
            shutil.rmtree(_vlm_shard_dir(cache_dir, base, split, shard_count), ignore_errors=True)
    return assembled


# --------------------------------------------------------------------------
# Text-side embedding and the zero-shot classifier built from it.
# --------------------------------------------------------------------------

def description_sha256(descriptions: Dict[str, str]) -> str:
    """Hash of the description text, in class-name-sorted order so the hash is
    independent of dict insertion order.

    Written into the text-prototype manifest: if a description changes but this
    hash in the cached manifest does not, every cold-start result built on the
    stale prototype is silently meaningless. `test_description_file_roundtrip`
    (planned, generate_class_description step) is the other half of this
    guard.
    """
    ordered = {key: descriptions[key] for key in sorted(descriptions)}
    blob = json.dumps(ordered, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@torch.inference_mode()
def _encode_text_prototypes_open_clip(model, vlm_name, class_prompts, device) -> torch.Tensor:
    """The same prompt-ensemble math as `encode_text_prototypes`, for an
    `open_clip` CustomTextCLIP (QuiltNet): normalize each prompt embedding
    INDIVIDUALLY, mean, normalize the mean.

    Kept as a separate function rather than a branch inside the loop because
    the tokenizer is obtained differently (`open_clip.get_tokenizer` keyed on
    the hub name, not CONCH's bundled BPE) even though `encode_text` is
    spelled the same.

    **Truncation is not expected here and is reported if it happens.**
    QuiltNet-B-16-PMB's PubMedBERT tower takes 256 tokens, and the
    `llm_morphology` descriptions this project runs measure 100-130 of them,
    so they fit whole -- which is the reason this checkpoint was chosen over a
    77-token CLIP tower. `open_clip`'s tokenizer pads and truncates to
    `context_length` silently, so the count below is the only thing that would
    notice a longer description file arriving later.
    """
    import open_clip
    import torch.nn.functional as F

    hub = vlm_name if vlm_name.startswith("hf-hub:") else f"hf-hub:{vlm_name}"
    tokenizer = open_clip.get_tokenizer(hub)
    limit = int(getattr(tokenizer, "context_length", 0) or 0)

    truncated = 0
    lengths = []
    prototypes = []
    for prompts in class_prompts:
        if not prompts:
            raise ValueError("a class's prompt list must not be empty")
        prompts = list(prompts)
        if limit:
            # Count real (unpadded) length per prompt, so a description file
            # that outgrows the tower cannot pass unnoticed.
            for one in prompts:
                ids = tokenizer(one)
                row = ids[0] if hasattr(ids, "__len__") and len(ids) else ids
                nonzero = int((row != 0).sum().item()) if hasattr(row, "sum") else 0
                lengths.append(nonzero)
                if nonzero >= limit:
                    truncated += 1
        token_ids = tokenizer(prompts).to(device)
        embeddings = model.encode_text(token_ids)
        embeddings = F.normalize(embeddings, dim=-1)
        prototype = embeddings.mean(dim=0)
        prototype = prototype / prototype.norm()
        prototypes.append(prototype)

    global LAST_TEXT_TRUNCATION
    LAST_TEXT_TRUNCATION = {
        "token_limit": limit,
        "prompts_total": len(lengths),
        "prompts_truncated": int(truncated),
        "longest_prompt_tokens": int(max(lengths)) if lengths else 0,
    }
    if truncated:
        print(
            f"[vlm] WARNING: {truncated}/{len(lengths)} prompts reached this "
            f"tower's {limit}-token limit and lost their tail. This checkpoint "
            "was chosen because llm_morphology fits inside it, so a hit here "
            "means the descriptions grew -- check them before trusting the row."
        )
    return torch.stack(prototypes, dim=0)


@torch.inference_mode()
def encode_text_prototypes(
    model,
    class_names: Sequence[str],
    class_prompts: Sequence[Sequence[str]],
    device: torch.device,
    vlm_name: Optional[str] = None,
) -> torch.Tensor:
    """One 512-d prototype per class, the official CONCH ensemble
    (`downstream/zeroshot_path.py::zero_shot_classifier`, verified line-by-line):
    for each class, encode every (prompt) string, L2-normalize each embedding
    INDIVIDUALLY, mean over all of them, then L2-normalize the mean.

    This is NOT "mean the raw embeddings then normalize once" -- that
    over-weights whichever prompt happened to produce a larger raw norm before
    normalization. `class_prompts[i]` is expected to already be the full
    cross product of (classname x template) for class `i`; this function does
    not care whether the caller built that from one manual description, an
    LLM-written one, or the official 22-template x 4-5-classname ensemble --
    it treats every style identically once expanded to a flat prompt list.

    Returns a `(C, 512)` tensor, L2-normalized per row, ready to compare
    against `PROJ_SPACE` image embeddings.
    """
    import torch.nn.functional as F

    if len(class_names) != len(class_prompts):
        raise ValueError(
            f"{len(class_names)} class names but {len(class_prompts)} prompt lists"
        )

    # QuiltNet (open_clip CustomTextCLIP) also has `encode_text`, so the two
    # families cannot be told apart by that. The VISION tower is what differs
    # unambiguously, and it is the same attribute `extract_vlm_image_features`
    # dispatches on, so the two functions cannot disagree about what a model is.
    #
    # Note the test for the open_clip branch is POSITIVE (`visual.trunk` +
    # `visual.head`) rather than "not CONCH-shaped". A negative test would
    # route anything unrecognised -- including a CONCH model that failed to
    # load -- into the open_clip path, where it would raise about a missing
    # `vlm_name` instead of about the real problem.
    visual = getattr(model, "visual", None)
    if hasattr(visual, "trunk") and hasattr(visual, "head"):
        if not vlm_name:
            # open_clip keys its tokenizer on the hub name, and the wrong
            # tokenizer produces valid-looking ids for a different vocabulary
            # -- prototypes that are silently meaningless. Refuse rather than
            # guess a default.
            raise ValueError(
                "encode_text_prototypes needs `vlm_name` for a non-CONCH "
                "checkpoint: open_clip resolves the tokenizer from the hub "
                "name, and the wrong vocabulary yields prototypes that look "
                "fine and mean nothing."
            )
        return _encode_text_prototypes_open_clip(
            model, vlm_name, class_prompts, device
        )

    from conch.open_clip_custom import get_tokenizer, tokenize

    tokenizer = get_tokenizer()
    # CONCH's own `tokenize` calls `tokenizer.batch_encode_plus`, which
    # transformers 5 removed along with `PreTrainedTokenizerFast` (replaced by
    # `TokenizersBackend`). CONCH has not followed, so on 5.x this raises
    # `AttributeError: TokenizersBackend has no attribute batch_encode_plus`
    # from inside the conch package -- after the 802 MB checkpoint has already
    # downloaded, and pointing at a file this project does not own. Say what is
    # actually wrong instead.
    if not hasattr(tokenizer, "batch_encode_plus"):
        import transformers

        raise RuntimeError(
            f"CONCH's tokenizer needs `batch_encode_plus`, which transformers "
            f"{transformers.__version__} does not provide (it was removed in "
            "transformers 5). Install transformers<5 -- requirements.txt pins "
            "this, so a 5.x here means the environment was built without it."
        )
    prototypes = []
    for prompts in class_prompts:
        if not prompts:
            raise ValueError("a class's prompt list must not be empty")
        token_ids = tokenize(tokenizer, list(prompts)).to(device)
        embeddings = model.encode_text(token_ids)  # (num_prompts, 512), normalize=True default
        embeddings = F.normalize(embeddings, dim=-1)
        prototype = embeddings.mean(dim=0)
        prototype = prototype / prototype.norm()
        prototypes.append(prototype)
    return torch.stack(prototypes, dim=0)


def zero_shot_logits(
    image_embeddings: np.ndarray,
    text_prototypes: np.ndarray,
    logit_scale: float,
) -> np.ndarray:
    """`softmax((image @ text.T) * logit_scale.exp())`, matching the official
    zero-shot code exactly (`downstream/zeroshot_path.py`, and the starter
    notebook's inline version).

    `logit_scale` here is already `model.logit_scale.exp().item()` -- a
    LEARNED parameter read off the loaded checkpoint, never a hard-coded
    temperature (verified against the checkpoint: init is `log(1/0.07)`, but
    training moves it, so 0.07 is not what a loaded model actually has).

    Pure numpy, no model needed: this is what lets the ensemble math and the
    class-order check be tested without `conch` installed.
    """
    image_embeddings = np.asarray(image_embeddings, dtype=np.float64)
    text_prototypes = np.asarray(text_prototypes, dtype=np.float64)
    logits = image_embeddings @ text_prototypes.T
    scaled = logits * float(logit_scale)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
