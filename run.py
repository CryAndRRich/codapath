from typing import Dict, List
import os
import yaml
import argparse

import numpy as np
import torch

from set_up import set_seed, clear_memory
from load_data import get_data_loaders
from model import extract_image_features, get_or_extract_features
from trainer import train_linear, save_model
from sampling import get_sampler
from evaluate import evaluate_model, palm_evaluate, format_palm_report


# SLICEABLE: one greedy order at max_budget; smaller budgets are exact prefixes.
SLICEABLE_SAMPLERS = {"random", "coreset", "codapath", "tcm", "refine"}

# PER_BUDGET: selection depends on the budget (budget-scaled phase / #clusters /
# centroids), so it must be re-run for every budget.
PER_BUDGET_SAMPLERS = {"typiclust", "activeft", "dropquery", "uncertainty_herding"}

# ITERATIVE: re-run per budget with internal probe-refinement rounds.
ITERATIVE_SAMPLERS = {"entropy", "margin", "badge", "scalpel", "scalpel_multiscale"}


def _save(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


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
) -> None:

    print(f"Device: {device}")
    set_seed(random_seed)

    dataset_key = os.path.basename(save_dir)
    train_loader, test_loader, class_names = get_data_loaders(data_path, random_seed, verbose)
    train_dataset = train_loader.dataset
    test_dataset  = test_loader.dataset

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
    )
    clear_memory()

    train_vlm_features        = None
    text_embeddings           = None
    train_stain_features      = None
    train_multiscale_features = None

    if sampler_name == "scalpel":
        from sampling.scalpel import extract_stain_features
        train_stain_features = extract_stain_features(train_loader, device)

    if sampler_name == "scalpel_multiscale":
        from sampling.scalpel_multiscale import get_or_extract_multiscale_features
        scale_factors = sampler_cfg.get("scale_factors", [2, 4])
        train_multiscale_features = get_or_extract_multiscale_features(
            train_loader, dataset_key, random_seed, vit_name, scale_factors,
            device, cache_dir=feature_cache_dir,
        )

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
            selected_indices = get_sampler(
                name=sampler_name,
                image_embeddings=train_features,
                stain_features=train_stain_features,                    # None unless scalpel
                multiscale_features=train_multiscale_features,          # None unless scalpel_multiscale
                oracle_labels=train_labels,
                max_budget=budget,
                num_classes=num_classes,
                device=device,
                **sampler_cfg,
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
        else:  
            selected_indices = master_selected[:budget]

        labeled_features = train_features[selected_indices]
        labeled_labels   = train_labels[selected_indices]

        _save(
            os.path.join(save_dir, f"{sampler_name}_selected_budget_{budget}.pt"),
            {"selected_indices": list(selected_indices),
             "selected_labels":  labeled_labels.tolist()},
        )

        if verbose:
            print(f"\n── {sampler_name.upper()} | budget={budget} ──")

        probe = train_linear(
            labeled_features, labeled_labels, num_classes,
            probe_epochs, probe_lr, device,
        )
        acc, pre, rec, f1 = evaluate_model(probe, test_features, test_labels, device)
        palm_acc[budget] = acc
        results[budget]  = {"acc": acc, "precision": pre, "recall": rec, "f1": f1}

        save_model(probe, os.path.join(save_dir, f"{sampler_name}_probe_budget_{budget}.pt"))
        del probe
        clear_memory()

    _save(
        os.path.join(save_dir, f"{sampler_name}_results.pt"),
        {"sampler": sampler_name, "budgets": cumulative_budget, "linear": results},
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
                print(format_palm_report(params, sampler_name, dataset_label))
            palm_path = os.path.join(save_dir, f"{sampler_name}_palm.pt")
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

    args = parser.parse_args(remaining_argv)

    dataset_key  = args.dataset
    dataset_info = config["datasets"][dataset_key]
    sampler_cfg  = config.get("samplers", {}).get(args.sampler_name, {})

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
    )