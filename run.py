from typing import Dict, List
import os
import yaml
import argparse

import numpy as np
import torch

from set_up import set_seed, clear_memory
from load_data import get_data_loaders
from model import DINOv2Extractor, extract_image_features
from trainer import train_linear, train_knn_linear
from sampling import get_sampler
from evaluate import evaluate_model, palm_evaluate, format_palm_report


SLICEABLE_SAMPLERS = {"random", "coreset", "codapath",
                      "uncertainty_herding", "tcm", "refine"}

PER_BUDGET_SAMPLERS = {"typiclust", "activeft", "dropquery"}

ITERATIVE_SAMPLERS = {"entropy", "margin", "badge", "scalpel"}


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
    knn_k: int,
    knn_threshold: float,
    device: torch.device,
    random_seed: int,
    save_dir: str,
    verbose: bool,
    model_cfg: Dict,
) -> None:

    print(f"Device: {device}")
    set_seed(random_seed)

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

    vit_name      = model_cfg.get("vit", "facebook/dinov2-base")
    dinov2        = DINOv2Extractor(model_name=vit_name).to(device)
    train_features = extract_image_features(train_loader, dinov2, device)  
    test_features  = extract_image_features(test_loader,  dinov2, device)  
    del dinov2
    clear_memory()

    train_vlm_features = None
    text_embeddings    = None

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

    if sampler_name == "scalpel":
        from sampling.scalpel import PLIPExtractor, extract_plip_text_features
        plip_name = model_cfg.get("vlm_secondary", "vinid/plip")
        plip = PLIPExtractor(model_name=plip_name).to(device)
        train_vlm_features = extract_image_features(train_loader, plip, device)
        del plip
        clear_memory()
        text_embeddings = extract_plip_text_features(
            data_descriptions, prompt_templates, class_names, device,
            plip_model=plip_name,
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

    palm_acc: Dict[str, Dict[int, float]] = {"linear": {}, "knn": {}}

    results: Dict[str, Dict[int, Dict[str, float]]] = {"linear": {}, "knn": {}}

    for budget in cumulative_budget:
        set_seed(random_seed)

        if sampler_name in ITERATIVE_SAMPLERS:
            selected_indices = get_sampler(
                name=sampler_name,
                image_embeddings=train_features,
                vlm_image_embeddings=train_vlm_features,   # None unless scalpel
                text_embeddings=text_embeddings,           # None unless scalpel
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

        probe_linear = train_linear(
            labeled_features, labeled_labels, num_classes,
            probe_epochs, probe_lr, device,
        )
        if verbose:
            print("  [linear]")
        acc_l, pre_l, rec_l, f1_l = evaluate_model(probe_linear, test_features, test_labels, device)
        palm_acc["linear"][budget] = acc_l
        results["linear"][budget]  = {"acc": acc_l, "precision": pre_l, "recall": rec_l, "f1": f1_l}

        _save(
            os.path.join(save_dir, f"{sampler_name}_probe_linear_budget_{budget}.pt"),
            probe_linear,
        )
        del probe_linear
        clear_memory()

        probe_knn = train_knn_linear(
            train_features, selected_indices, labeled_labels, num_classes,
            knn_k, knn_threshold, probe_epochs, probe_lr, device,
        )
        if verbose:
            print("  [knn_linear]")
        acc_k, pre_k, rec_k, f1_k = evaluate_model(probe_knn, test_features, test_labels, device)
        palm_acc["knn"][budget] = acc_k
        results["knn"][budget]  = {"acc": acc_k, "precision": pre_k, "recall": rec_k, "f1": f1_k}

        _save(
            os.path.join(save_dir, f"{sampler_name}_probe_knn_budget_{budget}.pt"),
            probe_knn,
        )
        del probe_knn
        clear_memory()

    _save(
        os.path.join(save_dir, f"{sampler_name}_results.pt"),
        {"sampler": sampler_name, "budgets": cumulative_budget,
         "linear": results["linear"], "knn": results["knn"]},
    )

    dataset_label = os.path.basename(save_dir)

    for mode, acc_dict in palm_acc.items():
        if len(acc_dict) < 4:
            print(f"[PALM/{mode}] Skipped: need ≥ 4 budget points, got {len(acc_dict)}.")
            continue
        try:
            params = palm_evaluate(
                budgets=list(acc_dict.keys()),
                accuracies=list(acc_dict.values()),
            )
            if verbose:
                print(format_palm_report(params, sampler_name, dataset_label))
            palm_path = os.path.join(save_dir, f"{sampler_name}_palm_{mode}.pt")
            _save(palm_path, params)
            print(f"[PALM/{mode}] Saved → {palm_path}")
        except Exception as e:
            print(f"[PALM/{mode}] Fitting failed: {e}")



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
    parser.add_argument("--knn_k",        type=int,  default=training_cfg.get("knn_k", 5))
    parser.add_argument("--knn_threshold",type=float,default=training_cfg.get("knn_threshold", 0.9))

    args = parser.parse_args(remaining_argv)

    dataset_key  = args.dataset
    dataset_info = config["datasets"][dataset_key]
    sampler_cfg  = config.get("samplers", {}).get(args.sampler_name, {})

    print("=" * 60)
    print(f"Dataset      : {dataset_key.upper()} ({dataset_info['num_classes']} classes)")
    print(f"Sampler      : {args.sampler_name.upper()}")
    print(f"Training     : linear + knn_linear (both modes)")
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
        knn_k=args.knn_k,
        knn_threshold=args.knn_threshold,
        device=torch.device(args.device),
        random_seed=args.seed,
        save_dir=os.path.join("checkpoints", dataset_key),
        verbose=args.verbose,
        model_cfg=model_cfg,
    )