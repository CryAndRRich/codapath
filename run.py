from typing import List, Dict
import os
import yaml
import argparse

import numpy as np
import torch

from set_up import set_seed, clear_memory
from load_data import get_data_loaders
from model import DINOv2Extractor, extract_image_features
from trainer import train_linear, train_knn_linear, save_model
from sampling import get_sampler


# ── Sampler categories ────────────────────────────────────────────────────────
# SLICEABLE: run ONCE at max_budget; [:budget] gives the result for any smaller budget.
# Greedy methods (coreset, codapath, uncertainty_herding, dcom, tcm, refine) and
# random all satisfy this property.
SLICEABLE_SAMPLERS = {"random", "coreset", "codapath",
                      "uncertainty_herding", "dcom", "tcm", "refine"}

# PER_BUDGET: must re-run for each budget (clustering depends on K=budget,
# or K-means output not naturally ordered).
PER_BUDGET_SAMPLERS = {"typiclust", "activeft", "dropquery"}

# ITERATIVE: use an internal LinearProbe that re-trains per round.
ITERATIVE_SAMPLERS = {"entropy", "margin", "badge"}


def main(data_path: str,
         sampler_name: str,
         num_classes: int,
         cumulative_budget: List[int],
         data_descriptions: Dict[str, str],
         prompt_templates: List[str],
         sampler_cfg: Dict,
         training_mode: str,
         probe_epochs: int,
         probe_lr: float,
         knn_k: int,
         knn_threshold: float,
         device: torch.device,
         random_seed: int,
         save_dir: str,
         verbose: bool,
         model_cfg: Dict) -> None:

    print(f"Device: {device}")
    set_seed(random_seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader, class_names = get_data_loaders(data_path, random_seed, verbose)
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

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

    # ── DINOv2 features (all methods) ────────────────────────────────────────
    vit_name = model_cfg.get("vit", "facebook/dinov2-base")
    dinov2 = DINOv2Extractor(model_name=vit_name).to(device)
    train_features = extract_image_features(train_loader, dinov2, device)   # (N, 768)
    test_features = extract_image_features(test_loader, dinov2, device)     # (M, 768)
    del dinov2
    clear_memory()

    # ── CODAPath-specific features (dual VLM + text; only when needed) ────────
    train_vlm_features = None
    text_embeddings = None
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

    # ── Sliceable: run once at max_budget ─────────────────────────────────────
    master_selected = None
    if sampler_name in SLICEABLE_SAMPLERS:
        samp_features = train_vlm_features if sampler_name == "codapath" else train_features
        kwargs = {
            "image_embeddings": samp_features,
            "oracle_labels": train_labels,        # passed to all; ignored if unused
            "num_classes": num_classes,
            "max_budget": max(cumulative_budget),
            "device": device,
            **sampler_cfg,
        }
        if sampler_name == "codapath":
            kwargs["text_embeddings"] = text_embeddings
        master_selected = get_sampler(name=sampler_name, **kwargs)

    # ── Per-budget training loop ───────────────────────────────────────────────
    for budget in cumulative_budget:
        set_seed(random_seed)

        if sampler_name in ITERATIVE_SAMPLERS:
            selected_indices = get_sampler(
                name=sampler_name,
                image_embeddings=train_features,
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
        else:  # SLICEABLE
            selected_indices = master_selected[:budget]

        labeled_features = train_features[selected_indices]
        labeled_labels = train_labels[selected_indices]

        if training_mode == "linear":
            probe = train_linear(
                labeled_features, labeled_labels, num_classes,
                probe_epochs, probe_lr, device,
            )
        else:  # knn_linear
            probe = train_knn_linear(
                train_features, selected_indices, labeled_labels, num_classes,
                knn_k, knn_threshold, probe_epochs, probe_lr, device,
            )

        if verbose:
            from evaluate import evaluate_model
            print(f"\n── {sampler_name.upper()} | budget={budget} ──")
            evaluate_model(probe, test_features, test_labels, device)

        # ── Save selected indices for this budget ─────────────────────────────
        data_save_path = os.path.join(save_dir, f"{sampler_name}_selected_budget_{budget}.pt")
        os.makedirs(os.path.dirname(data_save_path), exist_ok=True)
        torch.save(
            {"selected_indices": list(selected_indices),
             "selected_labels": labeled_labels.tolist()},
            data_save_path,
        )

        # ── Save linear probe weights (fc layer only) ─────────────────────────
        model_save_path = os.path.join(save_dir, f"{sampler_name}_probe_budget_{budget}.pt")
        save_model(probe, model_save_path)

        del probe
        clear_memory()


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config/config.yaml")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    if not os.path.exists(pre_args.config):
        raise FileNotFoundError(f"Config file not found: {pre_args.config}")

    with open(pre_args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    training_cfg = config.get("training", {})
    model_cfg = config.get("models", {})

    parser = argparse.ArgumentParser(description="Active Learning for Pathology")
    parser.add_argument("--config", type=str, default=pre_args.config)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--dataset", type=str, default=config.get("dataset", "pathmnist"),
                        choices=list(config["datasets"].keys()))
    parser.add_argument("--sampler_name", type=str,
                        default=config.get("sampler_name", "codapath"))
    parser.add_argument("--seed", type=int, default=config.get("random_seed", 42))
    parser.add_argument("--device", type=str, default=config.get("device", "cuda"))
    parser.add_argument("--training_mode", type=str,
                        default=training_cfg.get("mode", "linear"),
                        choices=["linear", "knn_linear"])
    parser.add_argument("--probe_epochs", type=int,
                        default=training_cfg.get("probe_epochs", 100))
    parser.add_argument("--probe_lr", type=float,
                        default=training_cfg.get("probe_lr", 1e-3))
    parser.add_argument("--knn_k", type=int, default=training_cfg.get("knn_k", 5))
    parser.add_argument("--knn_threshold", type=float,
                        default=training_cfg.get("knn_threshold", 0.9))

    args = parser.parse_args(remaining_argv)

    dataset_key = args.dataset
    if dataset_key not in config["datasets"]:
        raise ValueError(
            f"Dataset '{dataset_key}' not in config. "
            f"Available: {list(config['datasets'].keys())}"
        )
    dataset_info = config["datasets"][dataset_key]

    sampler_cfg = config.get("samplers", {}).get(args.sampler_name, {})

    print("=" * 60)
    print(f"Dataset      : {dataset_key.upper()} ({dataset_info['num_classes']} classes)")
    print(f"Sampler      : {args.sampler_name.upper()}")
    print(f"Training mode: {args.training_mode}")
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
        training_mode=args.training_mode,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        knn_k=args.knn_k,
        knn_threshold=args.knn_threshold,
        device=torch.device(args.device),
        random_seed=args.seed,
        save_dir=os.path.join("checkpoints", dataset_key),
        verbose=args.verbose,
        model_cfg=model_cfg,
    )
