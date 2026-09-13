"""Re-score a finished run's SAVED WEIGHTS against the test set.

This module exists because reading `<run>_results.pt` proves nothing about the
weights: those numbers were written by the run that produced them, and an
archive whose probe is corrupt, truncated, or paired with the wrong feature
space carries exactly the same file. Loading `<run>_probe_budget_<b>.pt`
(plus `<run>_lora_budget_<b>.pt` when the run trained an adapter) and running
`evaluate_probe` over real test features is what actually exercises them.

Two feature spaces, and picking the wrong one is SILENT:

* a DINOv2 run trains on `<ds>_seed<N>_facebook_dinov2-base_test.npy` (768-d)
* a CONCH run trains on `<ds>_seed<N>_MahmoodLab_CONCH_test.npy` -- the
  RAW_SPACE array, NOT the `_proj_test.npy` beside it. Both CONCH arrays are
  512-d, so handing over the projected one loads, runs, and reports a wrong
  number with nothing to say so. `probe_feature_paths` refuses a manifest
  whose `space` is not RAW_SPACE for that reason.

A LoRA run needs more than a different array. Its probe lives in the feature
space of the ADAPTED encoder, and no cache holds that space -- the cache is
the frozen encoder. `rescore_run` therefore refuses to score a LoRA run from
a cache and requires `encode_test` to be supplied, a callable that re-encodes
the test images through the adapter. That refusal is the point: the same
mistake once measured 0.10-0.21 accuracy against a 0.071 floor, invisible
because the widths matched.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from evaluation.metrics import evaluate_probe
from training.checkpoint import load_probe

__all__ = [
    "BUDGET_FROM_NAME",
    "find_run_dir",
    "read_run_metadata",
    "probe_feature_paths",
    "load_test_features",
    "make_lora_encoder",
    "rescore_run",
    "compare_to_recorded",
]

RAW_SPACE = "raw"


def _budget_of(path: str) -> int:
    """`.../<run>_probe_budget_125.pt` -> 125."""
    stem = os.path.basename(path)[: -len(".pt")]
    return int(stem.rsplit("_", 1)[1])


BUDGET_FROM_NAME = _budget_of


def find_run_dir(results_root: str, dataset: str, seed: int, run_name: str) -> str:
    """Locate one run's directory under an extracted `PACT/` or `baselines/`.

    The archive suffix is not part of the run name and is not consistent
    across runs, so both spellings are accepted rather than reconstructed.
    """
    base = os.path.join(results_root, dataset, f"seed{seed}")
    stem = f"{dataset}_{run_name}_seed{seed}"
    for candidate in (stem, stem + "_conch"):
        path = os.path.join(base, candidate)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"no run directory for {run_name!r} at {base} (tried {stem!r} and "
        f"{stem + '_conch'!r})"
    )


def read_run_metadata(run_dir: str) -> Dict[str, object]:
    """Describe a run by reading ONE of its probe checkpoints.

    The weights are the only thing an archive of weights is allowed to carry,
    so the encoder, the dataset, the seed and the run name are read from
    `metadata` inside a probe rather than from a results file. `save_probe`
    writes that dict for exactly this reason: without it a directory of
    `*_probe_budget_*.pt` can only be interpreted through its filenames.

    `encoder` is absent on checkpoints written before that field existed, and
    those are all DINOv2 -- the CONCH path was added after it. Defaulting is
    therefore safe HERE and nowhere else, and the default is stated rather than
    inferred so a future third encoder cannot inherit it silently.
    """
    probes = sorted(glob.glob(os.path.join(run_dir, "*_probe_budget_*.pt")), key=_budget_of)
    if not probes:
        raise FileNotFoundError(f"{run_dir}: no *_probe_budget_*.pt to describe the run")
    payload = torch.load(probes[0], map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not metadata:
        raise ValueError(
            f"{probes[0]}: no metadata, so nothing says which encoder this probe "
            "was trained on. Scoring it would be a guess."
        )
    encoder = metadata.get("encoder")
    kind = metadata.get("encoder_kind")
    if encoder is None:
        if kind is not None and kind != "dinov2":
            raise ValueError(
                f"{probes[0]}: encoder_kind={kind!r} but no encoder name -- "
                "refusing to guess the backbone."
            )
        encoder, kind = "facebook/dinov2-base", "dinov2"
    has_lora = bool(glob.glob(os.path.join(run_dir, "*_lora_budget_*.pt")))
    return {
        "run_name": metadata["run_name"],
        "dataset": metadata["dataset"],
        "seed": metadata["seed"],
        "sampler": metadata.get("sampler"),
        "encoder": encoder,
        "encoder_kind": kind,
        "class_names": metadata.get("class_names"),
        "final_train_cfg": metadata.get("final_train_cfg"),
        "has_lora": has_lora,
    }


def probe_feature_paths(feature_root: str, dataset: str, seed: int, backbone: str) -> Dict[str, str]:
    """Return `{test, manifest}` for the cache a FROZEN run's probe was trained on.

    `backbone` is the run's own encoder, read from probe metadata, so the
    caller never names it by hand. The manifest is returned with the array
    because it is what proves the space, and a CONCH cache holds two arrays of
    the same width.
    """
    slug = backbone.replace("/", "_")
    base = os.path.join(feature_root, dataset, f"seed{seed}")
    test = os.path.join(base, f"{dataset}_seed{seed}_{slug}_test.npy")
    manifest = os.path.join(base, f"{dataset}_seed{seed}_{slug}_manifest.json")
    if not os.path.exists(test):
        raise FileNotFoundError(f"no test features at {test}")
    return {"test": test, "manifest": manifest}


def load_test_features(
    paths: Dict[str, str],
    expect_conch: bool,
    test_fingerprint: Optional[str] = None,
) -> np.ndarray:
    """Load the test array, refusing PROJ_SPACE and a foreign split.

    The space check reads the manifest rather than the filename: `_proj_test`
    is a naming convention, `space` is the recorded fact. The fingerprint check
    is the one that catches a cache built from a DIFFERENT split -- the array
    still has the right width and the right row count, so every shape check
    passes while every label lines up with the wrong patch.
    """
    manifest_path = paths.get("manifest")
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        space = manifest.get("space")
        if expect_conch and space is not None and space != RAW_SPACE:
            raise ValueError(
                f"{manifest_path}: space={space!r}, but a probe trains on "
                f"{RAW_SPACE!r}. PROJ_SPACE is the same width, so scoring "
                "against it would report a wrong number silently."
            )
        cached = manifest.get("test_fingerprint")
        if test_fingerprint and cached and cached != test_fingerprint:
            raise ValueError(
                f"{manifest_path}: test_fingerprint {cached[:12]}... does not "
                f"match the run's {test_fingerprint[:12]}... -- this cache is a "
                "different split, and its rows would be scored against the "
                "wrong labels with nothing in the shapes to say so."
            )
    return np.load(paths["test"], mmap_mode="r")


def rescore_run(
    run_dir: str,
    test_features: Optional[np.ndarray],
    test_labels: np.ndarray,
    device: torch.device,
    encode_test: Optional[Callable[[str, int], np.ndarray]] = None,
    budgets: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[int, Dict[str, float]]:
    """Load every saved probe in `run_dir` and score it on the test set.

    Returns `{budget: {acc, precision, recall, f1}}`, computed here and not
    read from any `_results.pt`.

    `encode_test(lora_path, budget) -> features` is required for a run that
    saved LoRA adapters and is what makes a LoRA run scorable at all: its
    probe indexes the adapted encoder's space, which no cache contains.
    """
    probes = sorted(glob.glob(os.path.join(run_dir, "*_probe_budget_*.pt")), key=_budget_of)
    if not probes:
        raise FileNotFoundError(f"{run_dir}: no *_probe_budget_*.pt")

    adapters = {
        _budget_of(path): path
        for path in glob.glob(os.path.join(run_dir, "*_lora_budget_*.pt"))
    }

    out: Dict[int, Dict[str, float]] = {}
    for probe_path in probes:
        budget = _budget_of(probe_path)
        if budgets is not None and budget not in budgets:
            continue

        adapter = adapters.get(budget)
        if adapter is not None:
            if encode_test is None:
                raise ValueError(
                    f"{run_dir} saved a LoRA adapter for budget {budget}, so its "
                    "probe lives in the ADAPTED encoder's feature space. Pass "
                    "encode_test to re-encode the test set through that adapter; "
                    "scoring it against a frozen cache is the wrong-space bug "
                    "that both spaces being 512-d makes invisible."
                )
            features = encode_test(adapter, budget)
        else:
            if test_features is None:
                raise ValueError(f"{run_dir}: frozen run needs test_features")
            features = test_features

        probe = load_probe(probe_path, device)
        # A memmap stays read-only through `asarray`/`ascontiguousarray`, and
        # torch warns on every batch it wraps one. `np.array` copies, which
        # costs one test matrix and silences a warning that would otherwise
        # bury a real one.
        features = np.array(features, dtype=np.float32, copy=True)
        if probe.fc.in_features != features.shape[1]:
            raise ValueError(
                f"{probe_path}: probe expects {probe.fc.in_features}-d features, "
                f"got {features.shape[1]}-d -- wrong cache or wrong encoder."
            )
        accuracy, precision, recall, f1 = evaluate_probe(
            probe, features, test_labels, device, verbose=verbose
        )
        out[budget] = {
            "acc": accuracy, "precision": precision, "recall": recall, "f1": f1,
        }
    return out


def compare_to_recorded(
    rescored: Dict[int, Dict[str, float]],
    recorded: Dict[int, Dict[str, float]],
    tolerance: float = 1e-6,
) -> List[str]:
    """Report budgets where a freshly computed metric disagrees with the archive.

    This is a DIAGNOSTIC, not the source of the numbers: the rescored value is
    what the weights actually produce. A disagreement on a LoRA run is expected
    when the adapter was retrained rather than recovered bit-for-bit; on a
    frozen run it means the probe and the recorded metric no longer match.
    """
    lines = []
    for budget in sorted(set(rescored) & set(recorded)):
        for metric, value in sorted(rescored[budget].items()):
            other = recorded[budget].get(metric)
            if other is None:
                continue
            if abs(value - other) > tolerance:
                lines.append(
                    f"budget {budget:>3} {metric:<9} rescored {value * 100:.4f} "
                    f"vs recorded {other * 100:.4f}  (delta {(value - other) * 100:+.4f})"
                )
    return lines


def make_lora_encoder(
    run_metadata: dict,
    test_dataset,
    device: torch.device,
    batch_size: int = 64,
):
    """Build the `encode_test` callable a LoRA run needs.

    Returns `f(lora_path, budget) -> features`, which loads that budget's
    adapter onto a fresh copy of the base encoder and re-encodes the whole test
    set through it. One forward pass over the test set per budget: this is the
    expensive path, and it is the only correct one for a run that trained the
    encoder.

    `lora_alpha` is read from the adapter file rather than assumed. Alpha
    scales the delta at forward time and leaves no trace in the weights, so
    rebuilding at the wrong alpha yields a DIFFERENT encoder with identical
    shapes and no error.
    """
    # Imported here: a frozen-only run should not pay for torch's vision stack
    # or require the `conch` package to be installed at all.
    from training.finetune import encode_dataset
    from training.lora import apply_lora_to_conch, apply_lora_to_dinov2

    backbone = run_metadata["encoder"]
    is_conch = run_metadata["encoder_kind"] == "conch"

    def encode(lora_path: str, budget: int) -> np.ndarray:
        state = torch.load(lora_path, map_location="cpu", weights_only=False)
        rank, alpha = state["lora_r"], float(state["lora_alpha"])

        if is_conch:
            from features.vlm import load_conch
            model, preprocess = load_conch(backbone, torch.device("cpu"))
            model = apply_lora_to_conch(model, r=rank, alpha=alpha)
            encoder_kind, conch_preprocess = "conch", preprocess
        else:
            from transformers import Dinov2Model
            model = Dinov2Model.from_pretrained(backbone)
            model = apply_lora_to_dinov2(model, r=rank, alpha=alpha)
            encoder_kind, conch_preprocess = "dinov2", None

        missing, unexpected = model.load_state_dict(state["state"], strict=False)
        assert not unexpected, f"{lora_path}: unexpected tensors {unexpected[:3]}"
        loaded = [name for name in state["state"]]
        assert loaded, f"{lora_path}: empty adapter"
        # `reset_lora_parameters` zeroes B, so a state captured before training
        # rebuilds the FROZEN encoder -- correct shapes, silently useless.
        b_norm = sum(float(state["state"][k].norm()) for k in state["state"] if "lora_B" in k)
        assert b_norm > 0, (
            f"{lora_path}: every lora_B is zero, so this adapter reconstructs "
            "the frozen encoder rather than the trained one."
        )

        features = encode_dataset(
            test_dataset, model, device,
            image_encoder=encoder_kind,
            conch_preprocess=conch_preprocess,
            batch_size=batch_size,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return features

    return encode
