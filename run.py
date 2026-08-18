from typing import Dict, List
import os
import yaml
import argparse

import numpy as np
import torch

from set_up import set_seed, clear_memory
from load_data import get_data_loaders, get_sample_ids, sample_order_fingerprint
from model import extract_image_features, get_or_extract_features
from trainer import train_linear, save_model
from sampling import get_sampler
from evaluate import evaluate_model, palm_evaluate, format_palm_report


# SLICEABLE: one greedy order at max_budget; smaller budgets are exact prefixes.
SLICEABLE_SAMPLERS = {"random", "coreset", "codapath", "tcm"}

# PER_BUDGET: selection depends on the budget (budget-scaled phase / #clusters /
# centroids), so it must be re-run for every budget.
# `refine` is here (not SLICEABLE) because its stage-2 head is uncertainty_herding,
# whose internal phase-switch point scales with `max_budget` — slicing a single
# max-budget run would freeze every smaller budget inside phase-1 pure coverage,
# silently never applying the uncertainty-weighted stage-2 objective.
PER_BUDGET_SAMPLERS = {"typiclust", "activeft", "dropquery", "uncertainty_herding", "refine"}

# ITERATIVE: re-run per budget with internal probe-refinement rounds.
ITERATIVE_SAMPLERS = {"entropy", "margin", "badge", "scalpel", "nucleus_al", "nucleus_coverage", "graph_deuce"}


def _save(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


def _safe_run_name(name: str) -> str:
    if not name or name in {".", ".."}:
        raise ValueError("run_name must be a non-empty filename-safe identifier")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in name):
        raise ValueError("run_name may contain only letters, numbers, '_' and '-'")
    return name


def main(
    data_path: str,
    sampler_name: str,
    num_classes: int,
    cumulative_budget: List[int],
    data_descriptions: Dict[str, str],
    prompt_templates: List[str],
    sampler_cfg: Dict,
    probe_epochs: int,
    probe_lr: float,
    device: torch.device,
    random_seed: int,
    save_dir: str,
    verbose: bool,
    model_cfg: Dict,
    feature_cache_dir: str = "features",
    nucleus_cache_dir: str = "nucleus_features",
    run_name: str = None,
) -> None:

    print(f"Device: {device}")
    set_seed(random_seed)

    dataset_key = os.path.basename(save_dir)
    train_loader, test_loader, class_names = get_data_loaders(data_path, random_seed, verbose)
    train_dataset = train_loader.dataset
    test_dataset  = test_loader.dataset
    train_sample_ids = get_sample_ids(train_dataset)
    test_sample_ids = get_sample_ids(test_dataset)
    train_fingerprint = sample_order_fingerprint(train_sample_ids)
    test_fingerprint = sample_order_fingerprint(test_sample_ids)
    if run_name is None:
        if sampler_name == "nucleus_al":
            source = sampler_cfg.get("cell_source", "cellvit_embedding")
            uncertainty = sampler_cfg.get("uncertainty_mode", "disagreement")
            run_name = f"nucleus_{source}_{uncertainty}"
        elif sampler_name == "nucleus_coverage":
            coverage_source = sampler_cfg.get("coverage_source", "dino")
            run_name = f"nucleus_coverage_{coverage_source}"
            missing_impute = sampler_cfg.get("missing_impute", "mean")
            if missing_impute != "mean":
                # Keep the ablation from overwriting the main run's checkpoints.
                run_name = f"{run_name}_{missing_impute}"
        elif sampler_name == "graph_deuce":
            acquisition_variant = sampler_cfg.get("acquisition_variant", "laplace_margin")
            run_name = f"graph_deuce_{acquisition_variant}"
            if sampler_cfg.get("per_point", False):
                # Keep per_point=True runs from overwriting the round-based
                # (per_point=False) checkpoints of the same acquisition_variant.
                run_name = f"{run_name}_perpoint"
            embedding_reduction = sampler_cfg.get("embedding_reduction", "vae")
            if embedding_reduction != "vae":
                # Otherwise a "pca" sweep silently overwrites the "vae" run's
                # checkpoints for the same acquisition_variant (2026-08-18).
                run_name = f"{run_name}_{embedding_reduction}"
            k = sampler_cfg.get("k", 20)
            if k != 20:
                # Same collision risk when comparing k values (e.g. k=20 vs
                # k=100) for the same acquisition_variant in one session.
                run_name = f"{run_name}_k{k}"
    output_name = _safe_run_name(run_name or sampler_name)

    train_labels = (
        train_dataset.lbl
        if hasattr(train_dataset, "lbl")
        else np.array(train_dataset.dataset.targets)[train_dataset.indices]
    )
    test_labels = (
        test_dataset.lbl
        if hasattr(test_dataset, "lbl")
        else np.array(test_dataset.dataset.targets)[test_dataset.indices]
    )

    vit_name = model_cfg.get("vit", "facebook/dinov2-base")
    train_features, test_features = get_or_extract_features(
        train_loader, test_loader, dataset_key, random_seed, vit_name,
        device, cache_dir=feature_cache_dir,
        train_fingerprint=train_fingerprint,
        test_fingerprint=test_fingerprint,
    )
    clear_memory()

    train_vlm_features   = None
    text_embeddings      = None
    train_stain_features = None
    nucleus_embeddings    = None
    nucleus_reliability   = None
    nucleus_manifest      = None

    if sampler_name == "scalpel":
        from sampling.scalpel import extract_stain_features
        train_stain_features = extract_stain_features(train_loader, device)

    if sampler_name == "codapath":
        from sampling.codapath import DualVLMExtractor, extract_text_features
        vlm = DualVLMExtractor(
            plip_model=model_cfg.get("vlm_secondary", "vinid/plip"),
            biomedclip_model=model_cfg.get("vlm_primary",
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
        ).to(device)
        train_vlm_features = extract_image_features(train_loader, vlm, device)
        del vlm
        clear_memory()
        text_embeddings = extract_text_features(
            data_descriptions, prompt_templates, class_names, device,
            plip_model=model_cfg.get("vlm_secondary", "vinid/plip"),
            biomedbert_model=model_cfg.get("biomedbert",
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
        )

    if sampler_name in {"nucleus_al", "nucleus_coverage", "graph_deuce"}:
        from nucleus.cache import load_nucleus_cache
        from nucleus.ragged import pool_ragged_features

        cache_path = os.path.join(
            nucleus_cache_dir, f"{dataset_key}_seed{random_seed}"
        )
        nucleus_cache = load_nucleus_cache(
            cache_path, expected_sample_ids=train_sample_ids
        )
        cell_source = sampler_cfg.get("cell_source", "cellvit_embedding")
        if nucleus_cache.manifest.get("dataset") != dataset_key:
            raise ValueError("Nucleus cache dataset does not match the current run")
        if nucleus_cache.manifest.get("seed") != random_seed:
            raise ValueError("Nucleus cache seed does not match the current split")
        if (
            cell_source == "crop_dino"
            and nucleus_cache.manifest.get("dino_backbone") != vit_name
        ):
            raise ValueError(
                "Crop-DINO cache backbone does not match the run backbone"
            )
        cell_features = nucleus_cache.features(cell_source)
        nucleus_view = pool_ragged_features(
            cell_features,
            nucleus_cache.offsets,
            nucleus_cache.confidence,
            reliability_mode=sampler_cfg.get("reliability_mode", "valid"),
        )
        nucleus_embeddings = nucleus_view.patch_features
        nucleus_reliability = nucleus_view.reliability
        nucleus_manifest = nucleus_cache.manifest
        print(
            f"[nucleus] Loaded {cache_path}: "
            f"patches={nucleus_cache.num_patches}, cells={nucleus_cache.num_cells}, "
            f"source={cell_source}"
        )

    master_selected = None
    if sampler_name in SLICEABLE_SAMPLERS:
        samp_features = train_vlm_features if sampler_name == "codapath" else train_features
        kwargs = {
            "image_embeddings": samp_features,
            "oracle_labels":    train_labels,
            "num_classes":      num_classes,
            "max_budget":       max(cumulative_budget),
            "device":           device,
            **sampler_cfg,
        }
        if sampler_name == "codapath":
            kwargs["text_embeddings"] = text_embeddings
        master_selected = get_sampler(name=sampler_name, **kwargs)

    palm_acc: Dict[int, float] = {}
    results: Dict[int, Dict[str, float]] = {}

    for budget in cumulative_budget:
        set_seed(random_seed)

        if sampler_name in ITERATIVE_SAMPLERS:
            iterative_kwargs = dict(
                image_embeddings=train_features,
                stain_features=train_stain_features,          # None unless scalpel
                oracle_labels=train_labels,
                max_budget=budget,
                num_classes=num_classes,
                device=device,
                **sampler_cfg,
            )
            if sampler_name in {"nucleus_al", "nucleus_coverage", "graph_deuce"}:
                iterative_kwargs.update(
                    nucleus_embeddings=nucleus_embeddings,
                    nucleus_reliability=nucleus_reliability,
                )
            selected_indices = get_sampler(
                name=sampler_name, **iterative_kwargs
            )
        elif sampler_name in PER_BUDGET_SAMPLERS:
            selected_indices = get_sampler(
                name=sampler_name,
                image_embeddings=train_features,
                oracle_labels=train_labels,
                num_classes=num_classes,
                max_budget=budget,
                device=device,
                **sampler_cfg,
            )
        elif sampler_name == "tcm" and budget < 2 * num_classes:
            # tcm's greedy order is only prefix-exact once budget >= 2*num_classes
            # (below that, transition_budget = min(2*num_classes, budget) shrinks
            # the phase-1 cluster count itself, producing a DIFFERENT clustering,
            # not a subset of the master run) — slicing master_selected here would
            # silently report a non-prefix result. Recompute directly instead.
            selected_indices = get_sampler(
                name=sampler_name,
                image_embeddings=train_features,
                oracle_labels=train_labels,
                num_classes=num_classes,
                max_budget=budget,
                device=device,
                **sampler_cfg,
            )
        else:
            selected_indices = master_selected[:budget]

        labeled_features = train_features[selected_indices]
        labeled_labels   = train_labels[selected_indices]

        _save(
            os.path.join(save_dir, f"{output_name}_selected_budget_{budget}.pt"),
            {"selected_indices": list(selected_indices),
             "selected_labels":  labeled_labels.tolist(),
             "sampler": sampler_name,
             "run_name": output_name,
             "sampler_config": sampler_cfg,
             "train_fingerprint": train_fingerprint,
             "nucleus_manifest": nucleus_manifest},
        )

        if verbose:
            print(f"\n── {output_name.upper()} | budget={budget} ──")

        probe = train_linear(
            labeled_features, labeled_labels, num_classes,
            probe_epochs, probe_lr, device,
        )
        acc, pre, rec, f1 = evaluate_model(probe, test_features, test_labels, device)
        palm_acc[budget] = acc
        results[budget]  = {"acc": acc, "precision": pre, "recall": rec, "f1": f1}

        save_model(probe, os.path.join(save_dir, f"{output_name}_probe_budget_{budget}.pt"))
        del probe
        clear_memory()

    _save(
        os.path.join(save_dir, f"{output_name}_results.pt"),
        {"sampler": sampler_name, "run_name": output_name,
         "sampler_config": sampler_cfg, "budgets": cumulative_budget,
         "linear": results, "train_fingerprint": train_fingerprint,
         "test_fingerprint": test_fingerprint,
         "nucleus_manifest": nucleus_manifest},
    )

    dataset_label = os.path.basename(save_dir)

    if len(palm_acc) < 4:
        print(f"[PALM] Skipped: need ≥ 4 budget points, got {len(palm_acc)}.")
    else:
        try:
            params = palm_evaluate(
                budgets=list(palm_acc.keys()),
                accuracies=list(palm_acc.values()),
            )
            if verbose:
                print(format_palm_report(params, output_name, dataset_label))
            palm_path = os.path.join(save_dir, f"{output_name}_palm.pt")
            _save(palm_path, params)
            print(f"[PALM] Saved → {palm_path}")
        except Exception as e:
            print(f"[PALM] Fitting failed: {e}")



if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config/config.yaml")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    if not os.path.exists(pre_args.config):
        raise FileNotFoundError(f"Config file not found: {pre_args.config}")

    with open(pre_args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    training_cfg = config.get("training", {})
    model_cfg    = config.get("models", {})

    parser = argparse.ArgumentParser(description="Active Learning for Pathology")
    parser.add_argument("--config",       type=str,  default=pre_args.config)
    parser.add_argument("--verbose",      type=bool, default=False)
    parser.add_argument("--dataset",      type=str,  default=config.get("dataset", "pathmnist"),
                        choices=list(config["datasets"].keys()))
    parser.add_argument("--sampler_name", type=str,  default=config.get("sampler_name", "codapath"))
    parser.add_argument("--seed",         type=int,  default=config.get("random_seed", 42))
    parser.add_argument("--device",       type=str,  default=config.get("device", "cuda"))
    parser.add_argument("--probe_epochs", type=int,  default=training_cfg.get("probe_epochs", 100))
    parser.add_argument("--probe_lr",     type=float,default=training_cfg.get("probe_lr", 1e-3))
    parser.add_argument("--feature_cache_dir", type=str, default=config.get("feature_cache_dir", "features"))
    parser.add_argument("--nucleus_cache_dir", type=str, default=config.get("nucleus_cache_dir", "nucleus_features"))
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--cell_source", choices=["cellvit_embedding", "crop_dino"],
        default=None, help="Override nucleus_al.cell_source.",
    )
    parser.add_argument(
        "--uncertainty_mode",
        choices=["cell_margin", "disagreement", "fusion_concat", "fusion_add"],
        default=None, help="Override nucleus_al.uncertainty_mode.",
    )
    parser.add_argument(
        "--coverage_source", choices=["dino", "cellvit", "concat"],
        default=None, help="Override nucleus_coverage.coverage_source.",
    )
    parser.add_argument(
        "--missing_impute", choices=["mean", "zero"],
        default=None, help="Override nucleus_coverage.missing_impute.",
    )
    parser.add_argument(
        "--acquisition_variant",
        choices=[
            "laplace_margin", "uherding_swap_uncertainty", "uherding_swap_coverage",
            "laplace_plus_ppr", "deuce_native",
        ],
        default=None, help="Override graph_deuce.acquisition_variant.",
    )
    parser.add_argument(
        "--embedding_reduction", choices=["vae", "pca"],
        default=None, help="Override graph_deuce.embedding_reduction.",
    )

    args = parser.parse_args(remaining_argv)

    dataset_key  = args.dataset
    dataset_info = config["datasets"][dataset_key]
    sampler_cfg  = config.get("samplers", {}).get(args.sampler_name, {})
    sampler_cfg = dict(sampler_cfg)
    if args.cell_source is not None or args.uncertainty_mode is not None:
        if args.sampler_name != "nucleus_al":
            raise ValueError(
                "--cell_source/--uncertainty_mode are valid only for nucleus_al"
            )
        if args.cell_source is not None:
            sampler_cfg["cell_source"] = args.cell_source
        if args.uncertainty_mode is not None:
            sampler_cfg["uncertainty_mode"] = args.uncertainty_mode
    if args.coverage_source is not None or args.missing_impute is not None:
        if args.sampler_name != "nucleus_coverage":
            raise ValueError(
                "--coverage_source/--missing_impute are valid only for nucleus_coverage"
            )
        if args.coverage_source is not None:
            sampler_cfg["coverage_source"] = args.coverage_source
        if args.missing_impute is not None:
            sampler_cfg["missing_impute"] = args.missing_impute
    if args.acquisition_variant is not None:
        if args.sampler_name != "graph_deuce":
            raise ValueError("--acquisition_variant is valid only for graph_deuce")
        sampler_cfg["acquisition_variant"] = args.acquisition_variant
    if args.embedding_reduction is not None:
        if args.sampler_name != "graph_deuce":
            raise ValueError("--embedding_reduction is valid only for graph_deuce")
        sampler_cfg["embedding_reduction"] = args.embedding_reduction

    print("=" * 60)
    print(f"Dataset      : {dataset_key.upper()} ({dataset_info['num_classes']} classes)")
    print(f"Sampler      : {args.sampler_name.upper()}")
    print(f"Training     : linear probe")
    print(f"Budget       : {config['cumulative_budget']}")
    print(f"ViT backbone : {model_cfg.get('vit', 'facebook/dinov2-base')}")
    print(f"Probe LR: {args.probe_lr} | Epochs: {args.probe_epochs}")
    print("=" * 60)

    main(
        data_path=dataset_info["path"],
        sampler_name=args.sampler_name,
        num_classes=dataset_info["num_classes"],
        cumulative_budget=config["cumulative_budget"],
        data_descriptions=dataset_info["descriptions"],
        prompt_templates=config["prompt_templates"],
        sampler_cfg=sampler_cfg,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        device=torch.device(args.device),
        random_seed=args.seed,
        save_dir=os.path.join("checkpoints", dataset_key),
        verbose=args.verbose,
        model_cfg=model_cfg,
        feature_cache_dir=args.feature_cache_dir,
        nucleus_cache_dir=args.nucleus_cache_dir,
        run_name=args.run_name,
    )
