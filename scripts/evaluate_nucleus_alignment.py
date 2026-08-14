"""Diagnose whether nucleus features help the final linear learner.

This script does not rerun acquisition. It loads existing selected-index
checkpoints and evaluates several final-learner representations on the
unselected part of the training pool. The pool result is a fast diagnostic,
not a replacement for evaluation on the official test split.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from identity import sample_order_fingerprint
from load_data import get_data_loaders, get_sample_ids
from model import _feature_cache_paths
from nucleus.cache import load_nucleus_cache
from nucleus.alignment import (
    concat_blocks,
    fit_pca_block,
    normalized_auc,
    residualize_cell_block,
    standardize_l2,
)
from nucleus.ragged import pool_ragged_features, pool_ragged_moments
from set_up import clear_memory, set_seed
from trainer import train_linear


DEFAULT_BUDGETS = [25, 50, 75, 100, 125, 150, 175, 200]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--feature_cache_dir", default=None)
    parser.add_argument("--nucleus_cache_dir", default=None)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument(
        "--selection_run", action="append", default=None,
        help="Checkpoint prefix; repeat for several acquisition runs. If omitted, discover nucleus_* runs.",
    )
    parser.add_argument(
        "--all_runs", action="store_true",
        help="Evaluate every discovered nucleus run; default uses disagreement when available.",
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument(
        "--cell_source", choices=["cellvit_embedding", "crop_dino"],
        default="cellvit_embedding",
    )
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--pca_fit_samples", type=int, default=20000)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--probe_repeats", type=int, default=3)
    parser.add_argument("--probe_epochs", type=int, default=None)
    parser.add_argument("--probe_lr", type=float, default=None)
    parser.add_argument("--probe_weight_decay", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _labels(dataset) -> np.ndarray:
    if hasattr(dataset, "lbl"):
        return np.asarray(dataset.lbl, dtype=np.int64)
    return np.asarray(dataset.dataset.targets, dtype=np.int64)[dataset.indices]


def _load_dino_cache(
    cache_dir: str,
    dataset: str,
    seed: int,
    backbone: str,
    train_fingerprint: str,
    expected_rows: int,
) -> np.ndarray:
    train_path, _, manifest_path = _feature_cache_paths(
        cache_dir, dataset, seed, backbone
    )
    if not os.path.exists(train_path) or not os.path.exists(manifest_path):
        raise FileNotFoundError(
            "Aligned DINO train cache is missing. Run extract_features.ipynb "
            f"first: {train_path}"
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    expected = {
        "dataset": dataset,
        "seed": seed,
        "backbone": backbone,
        "train_fingerprint": train_fingerprint,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"DINO cache metadata/order mismatch: {mismatches}")
    features = np.load(train_path, mmap_mode="r")
    if features.ndim != 2 or len(features) != expected_rows:
        raise ValueError(
            f"DINO cache shape mismatch: {features.shape}, expected rows={expected_rows}"
        )
    return features


def _discover_runs(checkpoint_dir: str, budgets: Sequence[int]) -> List[str]:
    suffix = f"_selected_budget_{budgets[0]}.pt"
    paths = glob.glob(os.path.join(checkpoint_dir, f"nucleus_*{suffix}"))
    return sorted(os.path.basename(path)[:-len(suffix)] for path in paths)


def _load_selections(
    checkpoint_dir: str,
    runs: Sequence[str],
    budgets: Sequence[int],
    labels: np.ndarray,
    train_fingerprint: str,
) -> Dict[str, Dict[int, np.ndarray]]:
    selections: Dict[str, Dict[int, np.ndarray]] = {}
    for run in runs:
        by_budget: Dict[int, np.ndarray] = {}
        for budget in budgets:
            path = os.path.join(
                checkpoint_dir, f"{run}_selected_budget_{budget}.pt"
            )
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing selected-index checkpoint: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            saved_fingerprint = payload.get("train_fingerprint")
            if saved_fingerprint and saved_fingerprint != train_fingerprint:
                raise ValueError(f"Selection sample order mismatch: {path}")
            indices = np.asarray(payload["selected_indices"], dtype=np.int64)
            if len(indices) != budget or len(np.unique(indices)) != budget:
                raise ValueError(f"Invalid selection size/duplicates in {path}")
            if np.any(indices < 0) or np.any(indices >= len(labels)):
                raise ValueError(f"Out-of-range selected index in {path}")
            saved_labels = payload.get("selected_labels")
            if saved_labels is not None:
                np.testing.assert_array_equal(
                    labels[indices], np.asarray(saved_labels, dtype=np.int64),
                    err_msg=f"Selected labels do not align in {path}",
                )
            by_budget[budget] = indices
        selections[run] = by_budget
    return selections


def _evaluate_representation(
    features: np.ndarray,
    labels: np.ndarray,
    selections: Dict[str, Dict[int, np.ndarray]],
    num_classes: int,
    probe_epochs: int,
    probe_lr: float,
    probe_weight_decay: float,
    probe_repeats: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    output: Dict[str, Dict[str, Dict[str, float]]] = {}
    for run, by_budget in selections.items():
        output[run] = {}
        for budget, selected in by_budget.items():
            eval_mask = np.ones(len(labels), dtype=bool)
            eval_mask[selected] = False
            accuracies: List[float] = []
            macro_f1s: List[float] = []
            for repeat in range(probe_repeats):
                set_seed(seed + 1009 * repeat)
                probe = train_linear(
                    np.asarray(features[selected], dtype=np.float32),
                    labels[selected],
                    num_classes,
                    probe_epochs,
                    probe_lr,
                    device,
                    weight_decay=probe_weight_decay,
                )
                predictions = np.argmax(probe.predict_logits(features, device), axis=1)
                accuracies.append(float(accuracy_score(labels[eval_mask], predictions[eval_mask])))
                macro_f1s.append(float(f1_score(
                    labels[eval_mask], predictions[eval_mask],
                    average="macro", zero_division=0,
                )))
                del probe, predictions
                clear_memory()
            output[run][str(budget)] = {
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies, ddof=0)),
                "macro_f1_mean": float(np.mean(macro_f1s)),
                "macro_f1_std": float(np.std(macro_f1s, ddof=0)),
                "eval_size": int(eval_mask.sum()),
            }
            print(
                f"[{run} b={budget}] acc={np.mean(accuracies):.4f}±{np.std(accuracies):.4f} "
                f"f1={np.mean(macro_f1s):.4f}±{np.std(macro_f1s):.4f}"
            )
    return output


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dataset = args.dataset or config["dataset"]
    seed = config.get("random_seed", 42) if args.seed is None else args.seed
    if dataset not in config["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset}")
    dataset_cfg = config["datasets"][dataset]
    data_path = args.data_path or dataset_cfg["path"]
    feature_cache_dir = args.feature_cache_dir or config.get(
        "feature_cache_dir", "features"
    )
    nucleus_cache_dir = args.nucleus_cache_dir or config.get(
        "nucleus_cache_dir", "nucleus_features"
    )
    device = torch.device(args.device or config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.pca_dim < 1 or args.pca_fit_samples < 2:
        raise ValueError("pca_dim must be positive and pca_fit_samples >= 2")
    if args.probe_repeats < 1:
        raise ValueError("probe_repeats must be positive")
    if args.ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")
    if len(args.budgets) < 2 or any(
        right <= left for left, right in zip(args.budgets[:-1], args.budgets[1:])
    ):
        raise ValueError("budgets must be strictly increasing")

    train_loader, _, class_names = get_data_loaders(data_path, seed, verbose=True)
    train_ids = get_sample_ids(train_loader.dataset)
    train_fingerprint = sample_order_fingerprint(train_ids)
    labels = _labels(train_loader.dataset)
    backbone = config.get("models", {}).get("vit", "facebook/dinov2-base")
    dino = _load_dino_cache(
        feature_cache_dir, dataset, seed, backbone,
        train_fingerprint, len(labels),
    )
    nucleus_path = os.path.join(nucleus_cache_dir, f"{dataset}_seed{seed}")
    cache = load_nucleus_cache(nucleus_path, expected_sample_ids=train_ids)
    if cache.manifest.get("dataset") != dataset or cache.manifest.get("seed") != seed:
        raise ValueError("Nucleus cache dataset/seed mismatch")
    cell_features = cache.features(args.cell_source)
    mean_view = pool_ragged_features(
        cell_features, cache.offsets, cache.confidence,
    )
    moment_view = pool_ragged_moments(
        cell_features, cache.offsets, cache.confidence,
    )
    valid = mean_view.valid
    print(
        f"[alignment] valid nucleus patches={valid.mean():.3%}, "
        f"mean cells={mean_view.cell_counts.mean():.2f}"
    )

    runs = args.selection_run or _discover_runs(args.checkpoint_dir, args.budgets)
    if not runs:
        raise FileNotFoundError(
            f"No nucleus selected-index checkpoints found in {args.checkpoint_dir}"
        )
    if args.selection_run is None and not args.all_runs:
        preferred = "nucleus_cellvit_embedding_disagreement"
        runs = [preferred] if preferred in runs else [runs[0]]
        print(
            f"[alignment] Defaulting to selection run {runs[0]!r}; "
            "pass --all_runs to evaluate every nucleus acquisition run."
        )
    selections = _load_selections(
        args.checkpoint_dir, runs, args.budgets, labels, train_fingerprint,
    )
    training_cfg = config.get("training", {})
    probe_epochs = args.probe_epochs or training_cfg.get("probe_epochs", 100)
    probe_lr = args.probe_lr or training_cfg.get("probe_lr", 0.001)

    dino_normalized = standardize_l2(dino)
    cell_mean = standardize_l2(mean_view.patch_features, valid=valid)
    cell_moments = fit_pca_block(
        moment_view.patch_features, valid, args.pca_dim,
        args.pca_fit_samples, seed,
    )
    cell_residual = residualize_cell_block(
        dino, cell_moments, valid, args.pca_fit_samples,
        args.ridge_alpha, seed,
    )

    representations: Iterable[Tuple[str, np.ndarray]] = (
        ("dino_original", dino),
        ("dino_normalized", dino_normalized),
        ("cell_mean", cell_mean),
        ("dino_cell_mean", concat_blocks(dino_normalized, cell_mean)),
        ("dino_cell_moments", concat_blocks(dino_normalized, cell_moments)),
        ("dino_cell_residual", concat_blocks(dino_normalized, cell_residual)),
    )
    all_results: Dict[str, Dict] = {}
    for name, features in representations:
        print(f"\n=== representation={name} shape={features.shape} ===")
        all_results[name] = _evaluate_representation(
            features, labels, selections, len(class_names),
            probe_epochs, probe_lr, args.probe_weight_decay,
            args.probe_repeats, seed, device,
        )

    summary: Dict[str, Dict[str, float]] = {}
    for representation, by_run in all_results.items():
        summary[representation] = {}
        for run, by_budget in by_run.items():
            values = [by_budget[str(b)]["accuracy_mean"] for b in args.budgets]
            summary[representation][run] = normalized_auc(args.budgets, values)

    output_path = args.output or os.path.join(
        "alignment_results", f"nucleus_alignment_{dataset}_seed{seed}.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {
        "protocol": "unselected_train_pool_diagnostic_not_official_test",
        "dataset": dataset,
        "seed": seed,
        "train_fingerprint": train_fingerprint,
        "cell_source": args.cell_source,
        "budgets": list(args.budgets),
        "selection_runs": list(runs),
        "probe_repeats": args.probe_repeats,
        "probe_epochs": probe_epochs,
        "probe_lr": probe_lr,
        "probe_weight_decay": args.probe_weight_decay,
        "pca_dim": args.pca_dim,
        "pca_fit_samples": args.pca_fit_samples,
        "ridge_alpha": args.ridge_alpha,
        "valid_nucleus_rate": float(valid.mean()),
        "mean_cells_per_patch": float(mean_view.cell_counts.mean()),
        "normalized_auc": summary,
        "results": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print("\n=== normalized accuracy AUC (25-200) ===")
    for representation, by_run in summary.items():
        for run, value in by_run.items():
            print(f"{representation:24s} | {run:48s} | {value:.4f}")
    print("\n=== gain over normalized-DINO learner ===")
    for run in runs:
        baseline = summary["dino_normalized"][run]
        for representation in (
            "cell_mean", "dino_cell_mean", "dino_cell_moments",
            "dino_cell_residual",
        ):
            delta = summary[representation][run] - baseline
            print(f"{representation:24s} | {run:48s} | {delta:+.4f}")
    print(f"[alignment] Saved → {output_path}")


if __name__ == "__main__":
    main()
