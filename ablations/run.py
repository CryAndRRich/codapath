import argparse
import os
from typing import Dict, List, Union

import seaborn as sns
import torch
import yaml

from ablations import VALID_ABLATIONS
from ablations.model import AblationCODAModel, extract_text_embeddings, train_model
from load_data import get_data_loaders
from set_up import clear_memory, set_seed


def main(
    data_path: str,
    sampler_name: str,
    num_classes: int,
    cumulative_budget: Union[List[int], str],
    data_descriptions: Dict[str, str],
    prompt_templates: List[str],
    rank_lora: int,
    num_epochs: int,
    learn_rate: float,
    alpha: float,
    device: torch.device,
    random_seed: int,
    save_dir: str,
    ablation_approach: str,
    verbose: bool,
) -> None:
    if ablation_approach not in VALID_ABLATIONS:
        raise ValueError(
            f"Unsupported ablation '{ablation_approach}'. Expected one of: {sorted(VALID_ABLATIONS)}"
        )

    print(f"Device: {device}")
    print(f"Ablation: {ablation_approach}")
    seed_worker_fn = set_seed(random_seed)
    g_seed = torch.Generator()
    g_seed.manual_seed(random_seed)

    train_loader, test_loader, class_names = get_data_loaders(data_path, random_seed, verbose)
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    train_labels = (
        train_dataset.lbl
        if hasattr(train_dataset, "lbl")
        else torch.tensor(train_dataset.dataset.targets)[train_dataset.indices].numpy()
    )
    test_labels = (
        test_dataset.lbl
        if hasattr(test_dataset, "lbl")
        else torch.tensor(test_dataset.dataset.targets)[test_dataset.indices].numpy()
    )

    unique_classes = sorted(class_names)
    palette_colors = sns.color_palette("husl", len(unique_classes))
    color_map = dict(zip(unique_classes, palette_colors))

    text_embeddings = extract_text_embeddings(
        class_descriptions=data_descriptions,
        prompt_templates=prompt_templates,
        class_names=class_names,
        device=device,
        ablation_approach=ablation_approach,
    )

    model = AblationCODAModel(
        num_classes=num_classes,
        r=rank_lora,
        lora_alpha=rank_lora * 2,
        ablation_approach=ablation_approach,
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
        color_map=color_map,
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
        ablation_approach=ablation_approach,
        verbose=verbose,
    )

    del model, train_loader, test_loader
    clear_memory()


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config/config.yaml")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    if not os.path.exists(pre_args.config):
        raise FileNotFoundError(f"Config file not found at {pre_args.config}. Please check again")

    with open(pre_args.config, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)

    hyper = config.get("hyperparameters", {})

    parser = argparse.ArgumentParser(description="CODAPath ablation runner")
    parser.add_argument("--config", type=str, default=pre_args.config, help="Path to the YAML configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output during training and evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default=config.get("dataset", "pathmnist"),
        choices=["pathmnist", "histoset", "skintissue"],
        help="Dataset name to run",
    )
    parser.add_argument(
        "--sampler_name",
        type=str,
        default=config.get("sampler_name", "codapath"),
        help="Sampling algorithm name. CODAPath ablations should normally use 'codapath'.",
    )
    parser.add_argument("--seed", type=int, default=config.get("random_seed", 42), help="Random seed")
    parser.add_argument("--device", type=str, default=config.get("device", "cuda"), help="Device to run on")
    parser.add_argument("--alpha", type=float, default=hyper.get("alpha", 0.5), help="Coverage/uncertainty tradeoff")
    parser.add_argument("--num_epochs", type=int, default=hyper.get("num_epochs", 25), help="Number of fine-tuning epochs")
    parser.add_argument("--rank_lora", type=int, default=hyper.get("rank_lora", 8), help="Rank of the LoRA matrix")
    parser.add_argument("--lr", type=float, default=hyper.get("learning_rate", 1e-4), help="Learning rate")
    parser.add_argument(
        "--ablation_approach",
        type=str,
        required=True,
        choices=sorted(VALID_ABLATIONS),
        help="Which ablation to run",
    )

    args = parser.parse_args(remaining_argv)

    dataset_key = args.dataset
    if dataset_key not in config["datasets"]:
        raise ValueError(
            f"Dataset '{dataset_key}' not found in {pre_args.config}. "
            f"Available datasets: {list(config['datasets'].keys())}"
        )

    dataset_info = config["datasets"][dataset_key]

    print("============================================================")
    print(f"Dataset: {dataset_key.upper()} ({dataset_info['num_classes']} classes)")
    print(f"Sampler: {args.sampler_name.upper()}")
    print(f"Ablation: {args.ablation_approach}")
    print(f"Budget: {config['cumulative_budget']}")
    print(
        f"Alpha: {args.alpha} | LR: {args.lr} | Epochs: {args.num_epochs} | "
        f"LoRA Rank: {args.rank_lora}"
    )
    print("============================================================")

    main(
        data_path=dataset_info["path"],
        sampler_name=args.sampler_name,
        num_classes=dataset_info["num_classes"],
        cumulative_budget=config["cumulative_budget"],
        data_descriptions=dataset_info["descriptions"],
        prompt_templates=config["prompt_templates"],
        rank_lora=args.rank_lora,
        num_epochs=args.num_epochs,
        learn_rate=args.lr,
        alpha=args.alpha,
        device=torch.device(args.device),
        random_seed=args.seed,
        save_dir=os.path.join("ablations", "checkpoints", dataset_key),
        ablation_approach=args.ablation_approach,
        verbose=args.verbose,
    )

