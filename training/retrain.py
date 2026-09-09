"""Re-run the final-training pass over a FINISHED run's selection.

Nine T3/T4 runs were completed before `main.py` learned to persist the LoRA
adapter, so their encoders -- the feature space their probes were trained to
read -- are unrecoverable from the archives. The numbers survive; the weights
do not, which blocks visualisation, cross-scoring, and handing a reviewer
anything but "re-run it".

The expensive half of those runs does NOT need repeating. Selection is what
costs the AL loop its rounds, and it is already on disk: every archive holds
`*_selected_budget_*.pt` with the exact indices that were acquired. This module
reads those and runs only `finetune_and_evaluate`, which is the half that
touches the encoder.

**This is a re-training, not a recovery.** Adapter training draws from the
global RNG and shuffles its batches, so what comes out is a valid encoder for
this selection -- not, bit for bit, the one the original archive reported. The
metrics it returns are therefore the ones to quote for any figure built on the
adapter: they and the weights describe the same model, which is precisely what
the old archives cannot offer.

`train_fingerprint` is checked against the saved selection before anything
runs. Selected indices are positions into a split, and applying them to a
differently-ordered dataset silently trains on the wrong images.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from data.identity import sample_order_fingerprint
from data.loaders import get_data_loaders, get_sample_ids
from training.checkpoint import save_probe
from training.finetune import finetune_and_evaluate
from utils import clear_memory, set_seed

__all__ = ["load_selection", "retrain_run", "retrain_on_worker"]


def load_selection(run_dir: str, budget: int) -> Dict[str, Any]:
    """The saved selection payload for one budget, from either naming scheme.

    An archive names its files `<run_name>_selected_budget_<b>.pt`. The input
    dataset this notebook reads renames them to `budget_<b>.pt`, because
    Kaggle refuses any archive entry longer than 248 bytes and the full run
    name appeared TWICE in the original path (once as the directory, once as
    the filename) -- 254 bytes for the longest, which failed the upload. The
    payload inside is untouched, so both are read here rather than forcing the
    input to carry a name it does not need: `manifest.json` already records
    `run_name`, and it is the manifest, not the path, that names the output.
    """
    names = os.listdir(run_dir)
    matches = [n for n in names if n.endswith(f"_selected_budget_{budget}.pt")]
    if not matches:
        matches = [n for n in names if n == f"budget_{budget}.pt"]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{run_dir}: expected exactly one selection file for budget "
            f"{budget} (*_selected_budget_{budget}.pt or budget_{budget}.pt), "
            f"found {len(matches)}"
        )
    return torch.load(os.path.join(run_dir, matches[0]), map_location="cpu", weights_only=False)


def _dataset_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "lbl"):
        return dataset.lbl
    return np.array(dataset.dataset.targets)[dataset.indices]


def retrain_run(
    run_dir: str,
    data_path: str,
    save_dir: str,
    run_name: str,
    final_train_cfg: Dict[str, Any],
    num_classes: int,
    budgets: List[int],
    dataset_key: str,
    sampler_name: str,
    random_seed: int,
    device: torch.device,
    probe_epochs: int,
    probe_lr: float,
    image_encoder: str,
    visual_backbone: str,
    test_features: Optional[np.ndarray] = None,
    hf_token: Optional[str] = None,
    encoder_model=None,
    conch_preprocess=None,
    mmap_cache_dir: Optional[str] = None,
    shard_tag: Optional[str] = None,
    verbose: bool = True,
) -> Dict[int, Dict[str, float]]:
    """Re-train every budget of `run_dir`'s selection, saving probe + adapter.

    Returns `{budget: metrics}` for the freshly trained models. Writes
    `<run>_probe_budget_<b>.pt`, `<run>_lora_budget_<b>.pt` and
    `<run>_predictions_budget_<b>.pt` into `save_dir`, plus a `_results.pt`
    with the same shape `evaluation/results_io.py` already reads.

    The encoder is built ONCE and reset per budget, exactly as `main.run` does
    -- reloading it per budget costs more than the pass itself, and skipping
    the reset would let budget k+1 inherit budget k's adapter, which is a
    measured bug this project has already paid for once.

    `shard_tag` suffixes only the `_results.pt` filename, exactly as
    `main.run` does, so two workers can split the budgets across two GPUs and
    `main.merge_budget_shards` folds their tables afterwards. Everything else
    this writes is already named by budget, so nothing collides.
    """
    from training.lora import apply_lora_to_conch, apply_lora_to_dinov2, reset_lora_parameters

    use_lora = bool(final_train_cfg.get("use_lora"))
    if not use_lora:
        raise ValueError(
            "retrain_run is for LoRA runs: a frozen-encoder run has no adapter "
            "to recover, and its probe is already reproducible from the cache."
        )
    lora_r = int(final_train_cfg.get("lora_r", 8))
    lora_alpha = float(final_train_cfg.get("lora_alpha", 2 * lora_r))

    set_seed(random_seed)
    train_loader, test_loader, class_names = get_data_loaders(
        data_path, random_seed, verbose, mmap_cache_dir=mmap_cache_dir, num_workers=0,
    )
    train_dataset, test_dataset = train_loader.dataset, test_loader.dataset
    train_fingerprint = sample_order_fingerprint(get_sample_ids(train_dataset))
    train_labels = _dataset_labels(train_dataset)
    test_labels = _dataset_labels(test_dataset)

    # Build the encoder once, on the CPU. `finetune_and_evaluate` moves it to
    # the device itself; loading a 448px CONCH straight onto the card would
    # hold VRAM through everything that happens before the first batch.
    if encoder_model is not None:
        # Caller-supplied, already wrapped. Exists so a test can stand in a
        # small encoder -- a real DINOv2 forward over the test set is minutes
        # on CPU, which is what this pass costs on a GPU and what makes an
        # end-to-end CPU test of it impractical.
        encoder = encoder_model
    elif image_encoder == "dinov2":
        from transformers import Dinov2Model

        encoder = Dinov2Model.from_pretrained(visual_backbone)
        apply_lora_to_dinov2(encoder, r=lora_r, alpha=lora_alpha)
    else:
        from features.vlm import load_conch

        encoder, conch_preprocess = load_conch(
            visual_backbone, torch.device("cpu"), hf_token=hf_token
        )
        apply_lora_to_conch(encoder, r=lora_r, alpha=lora_alpha)

    os.makedirs(save_dir, exist_ok=True)
    results: Dict[int, Dict[str, float]] = {}
    for budget in budgets:
        payload = load_selection(run_dir, budget)
        saved_fp = payload.get("train_fingerprint")
        if saved_fp and saved_fp != train_fingerprint:
            raise ValueError(
                f"budget {budget}: the saved selection came from a split with "
                f"fingerprint {saved_fp[:12]}... but this dataset is "
                f"{train_fingerprint[:12]}... -- the indices would name "
                "different images"
            )
        selected = list(payload["selected_indices"])
        if len(selected) != budget:
            raise ValueError(
                f"budget {budget}: selection holds {len(selected)} indices"
            )

        n_reset = reset_lora_parameters(encoder, seed=random_seed)
        if n_reset == 0:
            raise RuntimeError(
                f"no LoRA adapters found to reset on the {image_encoder} "
                "encoder -- the wrap did not take effect"
            )
        if verbose:
            print(f"\n── {run_name} | budget={budget} (re-train from saved selection) ──")

        probe, metrics = finetune_and_evaluate(
            train_dataset=train_dataset,
            selected_indices=selected,
            labels=train_labels,
            test_features=test_features,
            test_labels=test_labels,
            num_classes=num_classes,
            device=device,
            probe_epochs=probe_epochs,
            probe_lr=probe_lr,
            image_encoder=image_encoder,
            use_lora=True,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_lr=final_train_cfg.get("lora_lr"),
            aux_loss=final_train_cfg.get("aux_loss", "none"),
            aux_weight=final_train_cfg.get("aux_weight", 0.5),
            augment=final_train_cfg.get("augment", "none"),
            encoder_model=encoder,
            conch_preprocess=conch_preprocess,
            test_dataset=test_dataset,
        )
        results[budget] = {
            key: metrics[key] for key in ("acc", "precision", "recall", "f1")
        }
        if verbose:
            print(f"   acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}")

        save_probe(
            probe, os.path.join(save_dir, f"{run_name}_probe_budget_{budget}.pt"),
            metadata={
                "run_name": run_name, "budget": budget, "seed": random_seed,
                "dataset": dataset_key, "sampler": sampler_name,
                "class_names": list(class_names),
                "metrics": results[budget], "encoder": visual_backbone,
                "encoder_kind": image_encoder,
                "final_train_cfg": final_train_cfg,
                "retrained_from_selection": True,
            },
        )
        if "lora_state" not in metrics:
            raise RuntimeError(
                "finetune_and_evaluate returned no adapter for a use_lora run "
                "-- recovering the adapter is the entire point of this pass"
            )
        torch.save(
            {
                "run_name": run_name, "budget": budget, "seed": random_seed,
                "encoder": visual_backbone, "encoder_kind": image_encoder,
                "lora_r": lora_r, "lora_alpha": lora_alpha,
                "state": metrics["lora_state"],
                "retrained_from_selection": True,
            },
            os.path.join(save_dir, f"{run_name}_lora_budget_{budget}.pt"),
        )
        torch.save(
            {
                "run_name": run_name, "budget": budget,
                "probabilities": metrics["probabilities"],
                "labels": np.asarray(test_labels),
                "retrained_from_selection": True,
            },
            os.path.join(save_dir, f"{run_name}_predictions_budget_{budget}.pt"),
        )
        clear_memory()

    shard_suffix = f"_{shard_tag}" if shard_tag else ""
    torch.save(
        {
            # `dataset` and `sampler` are not decoration: `results_io`'s
            # `discover_runs` filters on them, so an archive missing either is
            # invisible to every downstream table.
            "dataset": dataset_key,
            "sampler": sampler_name,
            "class_names": list(class_names),
            "run_name": run_name, "seed": random_seed,
            "visual_backbone": visual_backbone,
            "final_train_cfg": final_train_cfg,
            "train_fingerprint": train_fingerprint,
            "test_fingerprint": sample_order_fingerprint(get_sample_ids(test_dataset)),
            "num_classes": num_classes,
            "budgets": sorted(results),
            "linear": results,
            "retrained_from_selection": True,
        },
        os.path.join(save_dir, f"{run_name}{shard_suffix}_results.pt"),
    )
    del encoder
    clear_memory()
    return results


def retrain_on_worker(**kwargs) -> None:
    """`retrain_run` with the device resolved inside the process that uses it.

    `utils.parallel` hands each worker its own `device_string` (`cuda:<index>`),
    which is why this takes the string and builds the `torch.device` here: a
    device object made in the parent does not survive pickling meaningfully,
    and the worker -- not the caller -- is the only party that knows which card
    it owns.
    """
    device_string = kwargs.pop("device_string", "cuda:0")
    retrain_run(device=torch.device(device_string), **kwargs)
