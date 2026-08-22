"""Single entry point: run one sampler over a budget sweep and score it.

Usage:
    python main.py --dataset pathmnist --sampler scalpel

Every sampler is compared under the same protocol — same frozen DINOv2
features, same linear probe, same test metrics — so a difference in accuracy
can only come from which samples were selected. How a sampler is swept over the
budget list is decided by `sampling.specs.SAMPLER_SPECS`, not by anything here.

For each budget the run writes, under `<save_dir>`:
    <run>_selected_budget_<B>.pt   selected indices and their labels
    <run>_probe_budget_<B>.pt      the probe's linear weights
    <run>_results.pt               metrics for every budget
    <run>_palm.pt                  fitted PALM curve parameters
    <run>.log                      everything printed during the run
"""

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from data.identity import sample_order_fingerprint
from data.loaders import get_data_loaders, get_sample_ids
from evaluation.metrics import evaluate_probe
from evaluation.palm import format_palm_report, palm_evaluate
from features.visual import get_or_extract_features
from sampling import get_sampler
from sampling.specs import spec_for
from training.checkpoint import save_probe
from training.probe import train_probe
from utils import clear_memory, set_seed, tee_stdout

MIN_PALM_POINTS = 4


def _save(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def _safe_run_name(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if not name or name in {".", ".."}:
        raise ValueError("run_name must be a non-empty filename-safe identifier")
    if any(character not in allowed for character in name):
        raise ValueError("run_name may contain only letters, numbers, '_' and '-'")
    return name


def _default_run_name(sampler_name: str, sampler_cfg: Dict) -> str:
    """Encode the config axes that would otherwise overwrite each other."""
    if sampler_name != "scalpel":
        return sampler_name
    parts = [sampler_name, sampler_cfg.get("uncertainty_mode", "disagreement")]
    pooling = sampler_cfg.get("cell_pooling", "mean")
    if pooling != "mean":
        parts.append(pooling)
    if sampler_cfg.get("missing_impute", "mean") != "mean":
        parts.append(sampler_cfg["missing_impute"])
    if float(sampler_cfg.get("consistency_weight", 0.0)) > 0.0:
        parts.append(f"cons{sampler_cfg['consistency_weight']}".replace(".", "p"))
    return "_".join(parts)


def _load_cell_view(
    cache_dir: str,
    dataset_key: str,
    random_seed: int,
    sample_ids: List[str],
    sampler_cfg: Dict,
    visual_backbone: str,
):
    """Load the CellViT cache and pool it into one vector per patch."""
    from features.cellvit.cache import load_cellvit_cache
    from features.cellvit.pooling import pool_cells_mean, pool_cells_moments, pool_cells_rff

    cache_path = os.path.join(cache_dir, f"{dataset_key}_seed{random_seed}")
    cache = load_cellvit_cache(cache_path, expected_sample_ids=sample_ids)
    if cache.manifest.get("dataset") != dataset_key:
        raise ValueError("CellViT cache dataset does not match the current run")
    if cache.manifest.get("seed") != random_seed:
        raise ValueError("CellViT cache seed does not match the current split")

    cell_source = sampler_cfg.get("cell_source", "cellvit_embedding")
    if cell_source == "crop_dino" and cache.manifest.get("dino_backbone") != visual_backbone:
        raise ValueError("Crop-DINO cache backbone does not match the run backbone")

    pooling = sampler_cfg.get("cell_pooling", "mean")
    reliability_mode = sampler_cfg.get("reliability_mode", "valid")
    features = cache.features(cell_source)
    if pooling == "mean":
        view = pool_cells_mean(features, cache.offsets, cache.confidence, reliability_mode)
    elif pooling == "rff":
        view = pool_cells_rff(
            features, cache.offsets, cache.confidence, reliability_mode,
            output_dim=int(sampler_cfg.get("rff_dim", 64)),
            bandwidth=sampler_cfg.get("rff_bandwidth"),
            bandwidth_sample_size=int(sampler_cfg.get("rff_bandwidth_sample_size", 2048)),
            transform_batch_size=int(sampler_cfg.get("rff_transform_batch_size", 32768)),
        )
    elif pooling == "moments":
        view = pool_cells_moments(features, cache.offsets, cache.confidence, reliability_mode)
    else:
        raise ValueError(f"Unknown cell_pooling={pooling!r}; expected mean, rff or moments")

    print(
        f"[cellvit] {cache_path}: patches={cache.num_patches} cells={cache.num_cells} "
        f"source={cell_source} pooling={pooling}"
    )
    return view, cache.manifest


def run(
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
    cellvit_cache_dir: str = "cellvit_features",
    run_name: Optional[str] = None,
) -> None:
    spec = spec_for(sampler_name)
    output_name = _safe_run_name(run_name or _default_run_name(sampler_name, sampler_cfg))

    with tee_stdout(os.path.join(save_dir, f"{output_name}.log")):
        print(f"Device: {device}")
        set_seed(random_seed)

        dataset_key = os.path.basename(save_dir)
        train_loader, test_loader, class_names = get_data_loaders(
            data_path, random_seed, verbose
        )
        train_dataset, test_dataset = train_loader.dataset, test_loader.dataset
        train_sample_ids = get_sample_ids(train_dataset)
        train_fingerprint = sample_order_fingerprint(train_sample_ids)
        test_fingerprint = sample_order_fingerprint(get_sample_ids(test_dataset))

        train_labels = _dataset_labels(train_dataset)
        test_labels = _dataset_labels(test_dataset)

        visual_backbone = model_cfg.get("vit", "facebook/dinov2-base")
        train_features, test_features = get_or_extract_features(
            train_loader, test_loader, dataset_key, random_seed, visual_backbone,
            device, cache_dir=feature_cache_dir,
            train_fingerprint=train_fingerprint, test_fingerprint=test_fingerprint,
        )
        clear_memory()

        # The probe always trains on DINOv2 features. A sampler may still SELECT
        # in a different space -- CODAPath scores against VLM text prototypes,
        # so it needs that VLM's own image tower.
        selection_features = train_features
        sampler_inputs: Dict[str, object] = {}
        cellvit_manifest = None
        if "text_embeddings" in spec.needs:
            selection_features, sampler_inputs["text_embeddings"] = _load_vlm_inputs(
                train_loader, data_descriptions, prompt_templates, class_names,
                device, model_cfg,
            )
        if "cell_embeddings" in spec.needs:
            view, cellvit_manifest = _load_cell_view(
                cellvit_cache_dir, dataset_key, random_seed, train_sample_ids,
                sampler_cfg, visual_backbone,
            )
            sampler_inputs["cell_embeddings"] = view.patch_features
            sampler_inputs["cell_reliability"] = view.reliability

        common = dict(
            oracle_labels=train_labels,
            num_classes=num_classes,
            device=device,
            **sampler_inputs,
            **sampler_cfg,
        )

        # One shared run only when the spec says a prefix is faithful. Every
        # budget below `prefix_exact_min_class_multiple * num_classes` still
        # gets its own run, so a clamped threshold can never be reported as if
        # it were a prefix.
        master_selected: Optional[List[int]] = None
        if spec.prefix_exact:
            master_selected = get_sampler(
                name=sampler_name,
                image_embeddings=selection_features,
                max_budget=max(cumulative_budget),
                **common,
            )

        accuracy_by_budget: Dict[int, float] = {}
        results: Dict[int, Dict[str, float]] = {}

        for budget in cumulative_budget:
            set_seed(random_seed)
            if master_selected is not None and spec.is_prefix_exact(budget, num_classes):
                selected_indices = master_selected[:budget]
            else:
                selected_indices = get_sampler(
                    name=sampler_name,
                    image_embeddings=selection_features,
                    max_budget=budget,
                    **common,
                )

            labeled_features = train_features[selected_indices]
            labeled_labels = train_labels[selected_indices]

            _save(
                os.path.join(save_dir, f"{output_name}_selected_budget_{budget}.pt"),
                {
                    "selected_indices": list(selected_indices),
                    "selected_labels": labeled_labels.tolist(),
                    "sampler": sampler_name,
                    "run_name": output_name,
                    "sampler_config": sampler_cfg,
                    "train_fingerprint": train_fingerprint,
                    "cellvit_manifest": cellvit_manifest,
                },
            )

            if verbose:
                print(f"\n── {output_name} | budget={budget} ──")
            probe = train_probe(
                labeled_features, labeled_labels, num_classes, probe_epochs, probe_lr, device
            )
            accuracy, precision, recall, f1 = evaluate_probe(
                probe, test_features, test_labels, device
            )
            accuracy_by_budget[budget] = accuracy
            results[budget] = {
                "acc": accuracy, "precision": precision, "recall": recall, "f1": f1
            }
            save_probe(
                probe, os.path.join(save_dir, f"{output_name}_probe_budget_{budget}.pt")
            )
            del probe
            clear_memory()

        _save(
            os.path.join(save_dir, f"{output_name}_results.pt"),
            {
                "sampler": sampler_name,
                "run_name": output_name,
                "sampler_config": sampler_cfg,
                "budgets": cumulative_budget,
                "linear": results,
                "train_fingerprint": train_fingerprint,
                "test_fingerprint": test_fingerprint,
                "cellvit_manifest": cellvit_manifest,
            },
        )
        _fit_palm(accuracy_by_budget, save_dir, output_name, dataset_key, verbose)


def _dataset_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "lbl"):
        return dataset.lbl
    return np.array(dataset.dataset.targets)[dataset.indices]


def _load_vlm_inputs(
    train_loader,
    data_descriptions: Dict[str, str],
    prompt_templates: List[str],
    class_names: List[str],
    device: torch.device,
    model_cfg: Dict,
):
    """Return `(pool image features, class text prototypes)` in the VLM space.

    CODAPath compares images against text prototypes, so both sides have to
    come from the same dual-VLM encoder; DINOv2 features do not live in that
    space and cannot be substituted here.
    """
    from features.visual import extract_image_features
    from sampling.baselines.codapath import DualVLMExtractor, extract_text_features

    plip = model_cfg.get("vlm_secondary", "vinid/plip")
    biomedclip = model_cfg.get(
        "vlm_primary", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    extractor = DualVLMExtractor(plip_model=plip, biomedclip_model=biomedclip).to(device)
    image_features = extract_image_features(train_loader, extractor, device)
    del extractor
    clear_memory()

    text_embeddings = extract_text_features(
        data_descriptions, prompt_templates, class_names, device,
        plip_model=plip,
        biomedbert_model=model_cfg.get(
            "biomedbert",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        ),
    )
    return image_features, text_embeddings


def _fit_palm(
    accuracy_by_budget: Dict[int, float],
    save_dir: str,
    output_name: str,
    dataset_key: str,
    verbose: bool,
) -> None:
    if len(accuracy_by_budget) < MIN_PALM_POINTS:
        print(f"[PALM] Skipped: need >= {MIN_PALM_POINTS} budgets, got {len(accuracy_by_budget)}.")
        return
    try:
        params = palm_evaluate(
            budgets=list(accuracy_by_budget), accuracies=list(accuracy_by_budget.values())
        )
    except (RuntimeError, ValueError) as error:
        print(f"[PALM] Fitting failed: {error}")
        return
    if verbose:
        print(format_palm_report(params, output_name, dataset_key))
    path = os.path.join(save_dir, f"{output_name}_palm.pt")
    _save(path, params)
    print(f"[PALM] Saved -> {path}")


def _parse_overrides(pairs: List[str]) -> Dict[str, object]:
    """Turn `key=value` pairs into a typed sampler-config override dict.

    Values are read as YAML so `true`, `0.5`, `null` and `[1, 2]` all arrive as
    the right Python type instead of strings.
    """
    overrides: Dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override {pair!r} must look like key=value")
        key, raw = pair.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(raw)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one active-learning sampler over a budget sweep."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sampler", required=True, help="a name from sampling.specs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--probe_epochs", type=int, default=None)
    parser.add_argument("--probe_lr", type=float, default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--feature_cache_dir", default=None)
    parser.add_argument("--cellvit_cache_dir", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="override sampler config, e.g. --set uncertainty_mode=visual_margin",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.dataset not in config["datasets"]:
        raise ValueError(
            f"Unknown dataset {args.dataset!r}; config has {sorted(config['datasets'])}"
        )
    dataset_info = config["datasets"][args.dataset]
    training_cfg = config.get("training", {})
    sampler_cfg = dict(config.get("samplers", {}).get(args.sampler, {}))
    sampler_cfg.update(_parse_overrides(args.set))

    seed = args.seed if args.seed is not None else config.get("random_seed", 42)
    save_dir = args.save_dir or os.path.join(config.get("output_dir", "checkpoints"), args.dataset)

    spec = spec_for(args.sampler)
    print("=" * 68)
    print(f"Dataset  : {args.dataset} ({dataset_info['num_classes']} classes)")
    print(f"Sampler  : {args.sampler} [{spec.passes} pass, "
          f"prefix-exact={spec.prefix_exact}] {spec.why}")
    print(f"Budgets  : {config['cumulative_budget']}")
    print(f"Backbone : {config.get('models', {}).get('vit', 'facebook/dinov2-base')}")
    print(f"Config   : {sampler_cfg}")
    print("=" * 68)

    run(
        data_path=dataset_info["path"],
        sampler_name=args.sampler,
        num_classes=dataset_info["num_classes"],
        cumulative_budget=config["cumulative_budget"],
        data_descriptions=dataset_info.get("descriptions", {}),
        prompt_templates=config.get("prompt_templates", []),
        sampler_cfg=sampler_cfg,
        probe_epochs=args.probe_epochs or training_cfg["probe_epochs"],
        probe_lr=args.probe_lr or training_cfg["probe_lr"],
        device=torch.device(args.device or config.get("device", "cuda")),
        random_seed=seed,
        save_dir=save_dir,
        verbose=not args.quiet,
        model_cfg=config.get("models", {}),
        feature_cache_dir=args.feature_cache_dir or config.get("feature_cache_dir", "features"),
        cellvit_cache_dir=args.cellvit_cache_dir or config.get("cellvit_cache_dir", "cellvit_features"),
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
