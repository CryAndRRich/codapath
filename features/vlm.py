"""CONCH (or another pathology VLM) as a second, optional image encoder.

Verified 2026-08-29 against `repos/CONCH` (clone of `mahmoodlab/CONCH`) and
`pdfs/CONCH_2307.12914.pdf` directly — every claim below is read from the
actual model code, not from the README or the paper's prose alone.

**Two embedding spaces, and they are not interchangeable** (`coca_model.py`,
`CoCa.encode_image`):

    encode_image(x, proj_contrast=False, normalize=False)   512-d, pre-projection
        -> linear probe, coverage kernel, everything scalpel's disagreement
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
    "extract_vlm_image_features",
    "get_or_extract_vlm_features",
    "load_official_conch_prompts",
    "assert_class_order_matches_prompts",
    "encode_text_prototypes",
    "zero_shot_logits",
]

# The two spaces by name, used consistently as dict keys and file-name
# fragments so "which space is this" is never inferred from context.
RAW_SPACE = "raw"    # proj_contrast=False, normalize=False -- for the probe
PROJ_SPACE = "proj"  # proj_contrast=True,  normalize=True  -- for text comparison


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
    hand-rolling an equivalent transform is the exact mistake §10.3 of
    PLAN_IMPLEMENT.md documents: it does not crash, it just silently shifts
    every embedding.
    """
    from conch.open_clip_custom import create_model_from_pretrained

    checkpoint = vlm_name if vlm_name.startswith("hf_hub:") else f"hf_hub:{vlm_name}"
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", checkpoint, hf_auth_token=hf_token or None, device=device,
    )
    model.eval()
    return model, preprocess


@torch.inference_mode()
def extract_vlm_image_features(
    dataloader,
    model,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """One forward pass, both embedding spaces.

    Returns `(raw, proj)`: `raw` is `proj_contrast=False, normalize=False`
    (768 for CONCH's captioning head width is NOT this -- it is the 512-d
    pre-projection vector `forward_no_head` returns, matching `embed_dim` in
    `conch_ViT-B-16.json`); `proj` is `proj_contrast=True, normalize=True`.

    Both are computed from the SAME forward pass through the vision tower --
    re-running the whole dataloader a second time for the other space would
    double a cost that is already the expensive part of this notebook (448x448
    is 4x the pixels of DINOv2's 224x224).
    """
    from tqdm import tqdm

    model = model.to(device)
    model.eval()

    raw_batches: List[np.ndarray] = []
    proj_batches: List[np.ndarray] = []
    for images, _ in tqdm(dataloader, desc="Extracting VLM features", leave=False):
        images = images.to(device, non_blocking=True)
        raw = model.encode_image(images, normalize=False, proj_contrast=False)
        proj = model.encode_image(images, normalize=True, proj_contrast=True)
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
        model, _ = load_conch(vlm_name, device, hf_token=hf_token)
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
# Official CONCH prompts -- pure data, no model needed to load or validate.
# --------------------------------------------------------------------------

def load_official_conch_prompts(path: str) -> Dict[str, object]:
    """Read `crc100k_prompts_all_per_class.json`.

    The content is nested one level deeper than it looks: everything is under
    a `"0"` key, i.e. `data["0"]["classnames"]` and `data["0"]["templates"]`,
    not `data["classnames"]`. Reading it at the top level raises `KeyError`
    immediately -- documented here once so nobody re-derives this by trial and
    error.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    inner = raw["0"]
    return {"classnames": inner["classnames"], "templates": inner["templates"]}


def assert_class_order_matches_prompts(class_names: Sequence[str], prompt_classnames: Dict[str, List[str]]) -> None:
    """The order of `class_names` (from `config.yaml`) must match the order of
    `prompt_classnames.keys()` (from the official CONCH prompt file) one-to-one.

    Verified once by hand for PathMNIST (`adipose->ADI, background->BACK, ...,
    colorectal_adenocarcinoma->TUM`), but this must be checked in code, not
    assumed: a silent mismatch does not crash, it just permutes the confusion
    matrix and reports a wrong number that looks plausible.

    This checks COUNT and POSITION only -- it does not (and cannot) check that
    `adipose` at position 0 semantically means the same thing as `ADI` at
    position 0. That correspondence has to be verified once, by a human reading
    both label sets, which is what the docstring above records.
    """
    prompt_keys = list(prompt_classnames.keys())
    if len(class_names) != len(prompt_keys):
        raise ValueError(
            f"{len(class_names)} dataset classes vs {len(prompt_keys)} prompt "
            f"classes -- this dataset does not have an official CONCH prompt "
            f"file for it (only PathMNIST/CRC100K does)."
        )


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
def encode_text_prototypes(
    model,
    class_names: Sequence[str],
    class_prompts: Sequence[Sequence[str]],
    device: torch.device,
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
    from conch.open_clip_custom import get_tokenizer, tokenize
    import torch.nn.functional as F

    if len(class_names) != len(class_prompts):
        raise ValueError(
            f"{len(class_names)} class names but {len(class_prompts)} prompt lists"
        )
    tokenizer = get_tokenizer()
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
