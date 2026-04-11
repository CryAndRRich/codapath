from typing import List, Dict
import os
import yaml
import argparse

import numpy as np
import torch
import seaborn as sns

from set_up import set_seed, clear_memory
from load_data import get_data_loaders
from model import CODAModel, extract_text_embeddings, train_model

def main(data_path: str,
         sampler_name: str,
         num_classes: int,
         cumulative_budget: List[int],
         data_descriptions: Dict[str, str],
         prompt_templates: Dict[str, str],
         rank_lora: int,
         num_epochs: int,
         learn_rate: float,
         alpha: float,
         device: torch.device,
         random_seed: int,
         save_dir: str,
         verbose: bool) -> None:
    
    print(f"Device: {device}")
    seed_worker_fn = set_seed(random_seed)
    g_seed = torch.Generator()
    g_seed.manual_seed(random_seed)

    train_loader, test_loader, class_names = get_data_loaders(data_path, random_seed, verbose)

    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    train_labels = train_dataset.lbl if hasattr(train_dataset, "lbl") else np.array(train_dataset.dataset.targets)[train_dataset.indices]
    test_labels = test_dataset.lbl if hasattr(test_dataset, "lbl") else np.array(test_dataset.dataset.targets)[test_dataset.indices]

    unique_classes = sorted(class_names)
    palette_colors = sns.color_palette("husl", len(unique_classes))
    COLOR_MAP = dict(zip(unique_classes, palette_colors))

    text_embeddings = extract_text_embeddings(
        class_descriptions=data_descriptions,
        prompt_templates=prompt_templates,
        class_names=class_names,
        device=device
    )

    model = CODAModel(
        num_classes=num_classes, 
        r=rank_lora, 
        lora_alpha=rank_lora * 2
    ).to(device)

    train_model(
        model=model,
        sampler_name=sampler_name,
        train_dataset=train_dataset,
        oracle_labels=train_labels,        
        test_loader=test_loader,        
        test_labels=test_labels,        
        class_names=class_names,  
        text_embeddings=text_embeddings,
        color_map=COLOR_MAP,
        cumulative_budget=cumulative_budget,
        num_epochs=num_epochs,
        learn_rate=learn_rate,
        alpha=alpha,
        r=rank_lora,
        device=device,
        seed_worker_fn=seed_worker_fn,
        g_seed=g_seed,
        seed=random_seed,
        save_dir=save_dir,
        verbose=verbose
    )

    del model, train_loader, test_loader
    clear_memory()

if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config/config.yaml")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    if not os.path.exists(pre_args.config):
        raise FileNotFoundError(f"Config file not found at {pre_args.config}. Please check again")
        
    with open(pre_args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    hyper = config.get("hyperparameters", {})

    parser = argparse.ArgumentParser(description="CODAPath: Active Learning for Pathology")
    
    parser.add_argument("--config", type=str, default=pre_args.config, help="Path to the YAML configuration file")
    parser.add_argument("--verbose", type=bool, default=False, help="Enable verbose output during training and evaluation")

    parser.add_argument("--dataset", type=str, default=config.get("dataset", "pathmnist"), choices=["pathmnist", "histoset", "skintissue"], help="Dataset name to run")
    parser.add_argument("--sampler_name", type=str, default=config.get("sampler_name", "codapath"), help="Sampling algorithm name (e.g., codapath, random, entropy...)")
    parser.add_argument("--seed", type=int, default=config.get("random_seed", 42), help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=config.get("device", "cuda"), help="Device to run on")
    
    parser.add_argument("--alpha", type=float, default=hyper.get("alpha", 0.5), help="Balance coefficient between Coverage and Uncertainty")
    parser.add_argument("--num_epochs", type=int, default=hyper.get("num_epochs", 25), help="Number of epochs to fine-tune LoRA")
    parser.add_argument("--rank_lora", type=int, default=hyper.get("rank_lora", 8), help="Rank of the LoRA matrix")
    parser.add_argument("--lr", type=float, default=hyper.get("learning_rate", 1e-4), help="Learning rate")
    
    args = parser.parse_args(remaining_argv)

    dataset_key = args.dataset
    if dataset_key not in config["datasets"]:
        raise ValueError(f"Dataset '{dataset_key}' not found in {pre_args.config}). Available datasets: {list(config['datasets'].keys())}")
        
    dataset_info = config["datasets"][dataset_key]
    
    data_path = dataset_info["path"]
    num_classes = dataset_info["num_classes"]
    data_descriptions = dataset_info["descriptions"]
    
    prompt_templates = config["prompt_templates"]
    cumulative_budget = config["cumulative_budget"]

    device = torch.device(args.device)

    print(f"============================================================")
    print(f"Dataset: {dataset_key.upper()} ({num_classes} classes)")
    print(f"Sampler: {args.sampler_name.upper()}")
    print(f"Budget: {cumulative_budget}")
    print(f"Alpha: {args.alpha} | LR: {args.lr} | Epochs: {args.num_epochs} | LoRA Rank: {args.rank_lora}")
    print(f"============================================================")
    
    main(
        data_path=data_path,
        sampler_name=args.sampler_name,
        num_classes=num_classes,
        cumulative_budget=cumulative_budget,
        data_descriptions=data_descriptions,
        prompt_templates=prompt_templates,
        rank_lora=args.rank_lora,
        num_epochs=args.num_epochs,
        learn_rate=args.lr,
        alpha=args.alpha,
        device=device,
        random_seed=args.seed,
        save_dir=f"checkpoints/{dataset_key}",
        verbose=args.verbose
    )