"""Single entry point: run one sampler over a budget sweep and score it.

Usage:
    python main.py --dataset pathmnist --sampler scalpel

Every sampler is compared under the same protocol — same frozen DINOv2
features, same linear probe, same test metrics — so a difference in accuracy
can only come from which samples were selected. How a sampler is swept over the
budget list is decided by `sampling.specs.SAMPLER_SPECS`, not by anything here.

For each budget the run writes, under `<save_dir>`:
    <run>_selected_budget_<B>.pt      indices, sample ids, labels, per-step
                                      trace (round/rank/score) and sanity report
    <run>_probe_budget_<B>.pt         the probe's linear weights plus metadata
    <run>_predictions_budget_<B>.pt   test-set probabilities and labels
    <run>_results.pt                  metrics and timings for every budget
    <run>.log                         everything printed during the run

Everything a later plot might need is written during the run, because none of
it can be recovered afterwards: the per-step acquisition score exists only
inside the greedy loop, and the test predictions would cost a full backbone
pass to rebuild.

This run reports accuracy, precision, recall and macro-F1 only. Curve-level
evaluation — PALM and anything else derived from the whole budget sweep — is
`notebooks/evaluate_al_sampler.ipynb`'s job, deliberately: it is a pure
function of the per-budget outputs written here, it needs every budget to have
finished (which a resumed or GPU-split run cannot guarantee mid-sweep), and
re-fitting it there costs seconds against re-running the sweep.
"""

import argparse
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from data.identity import sample_order_fingerprint
from data.loaders import get_data_loaders, get_sample_ids
from evaluation.metrics import evaluate_probe
from evaluation.sanity import (
    SEVERITY_ORDER,
    check_selection,
    format_report as format_sanity_report,
)
from features.visual import get_or_extract_features
from sampling import get_sampler
from sampling.specs import spec_for
from training.checkpoint import save_probe
from training.probe import train_probe
from utils import clear_memory, set_seed, tee_stdout
from utils.progress import Stopwatch, format_duration
from utils.trace import SelectionTrace


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


def _prefix_trace(master: Optional[SelectionTrace], budget: int):
    """The part of a shared prefix-exact run that produced this budget.

    A prefix-exact sampler runs once at the maximum budget; the first `budget`
    steps are exactly what a direct run at `budget` would have done, so they are
    this budget's trace. Rounds are kept whole — they describe the state the
    picks were made under, and a round is either entered or it is not.
    """
    if master is None:
        return None
    payload = master.to_payload()
    steps = [step for step in payload["steps"] if step["rank"] < budget]
    rounds_used = {step["round_index"] for step in steps}
    clipped = SelectionTrace(master.sampler, budget, master.pool_size)
    clipped.steps = [step for step in master.steps if step.rank < budget]
    clipped.rounds = [
        record for record in master.rounds if record.round_index in rounds_used
    ]
    return clipped


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
    mmap_cache_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    shard_tag: Optional[str] = None,
) -> None:
    spec = spec_for(sampler_name)
    output_name = _safe_run_name(run_name or _default_run_name(sampler_name, sampler_cfg))

    # A budget-sharded run splits `cumulative_budget` across processes. The
    # per-budget files are already named by budget and so never collide, but
    # the results table and the log are per-RUN and would overwrite each
    # other. `shard_tag` gives those two a distinct name per shard; the shards
    # are merged afterwards by `merge_budget_shards`. Only meaningful for a
    # sampler that is NOT prefix-exact, since a prefix-exact one shares a
    # single selection pass that must not be repeated per shard.
    shard_suffix = "" if shard_tag is None else f"_{_safe_run_name(shard_tag)}"
    if shard_tag is not None and spec.prefix_exact:
        raise ValueError(
            f"{sampler_name!r} is prefix-exact: its whole sweep comes from ONE "
            "selection pass at the maximum budget, so splitting its budgets "
            "across processes would repeat that pass per shard and cost more "
            "than it saves. Run it unsharded."
        )

    run_started = time.time()
    with tee_stdout(os.path.join(save_dir, f"{output_name}{shard_suffix}.log")):
        print(f"Device: {device}")
        set_seed(random_seed)

        dataset_key = os.path.basename(save_dir)
        # `mmap_cache_dir` matters here even though this function never looks at
        # a pixel: it opens the dataset for labels, sample IDs and the cache
        # fingerprint, and NPZDataset reads a .npz eagerly. PathMNIST-224 is
        # ~15 GiB of uint8, so two AL workers sharing a Kaggle session hold
        # ~30 GiB between them and one is killed before the sampler starts.
        # num_workers=0: an AL run never reads a pixel. The loaders exist for
        # labels, sample IDs and the cache fingerprint, and on a cache hit
        # `get_or_extract_features` only asks them for `len(dataset)`. Workers
        # would be per-loader processes held alive by persistent_workers for the
        # whole run — 8 of them across two GPU workers and two loaders — paying
        # startup and memory for batches nobody iterates. The extraction
        # notebooks, which DO decode images, choose their own count.
        train_loader, test_loader, class_names = get_data_loaders(
            data_path, random_seed, verbose,
            mmap_cache_dir=mmap_cache_dir, num_workers=0,
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

        # The probe always trains on visual_backbone features, and every
        # sampler currently selects in that same space -- no sampler declares
        # `text_embeddings` in `spec.needs` at the moment. A future cold-start
        # text prior is a config-driven axis of `scalpel` itself, not a static
        # per-sampler requirement, and will be wired in separately.
        selection_features = train_features
        sampler_inputs: Dict[str, object] = {}
        cellvit_manifest = None
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
        master_trace: Optional[SelectionTrace] = None
        if spec.prefix_exact:
            print(f"\n{'=' * 68}\nShared run at max budget {max(cumulative_budget)}")
            master_trace = SelectionTrace(
                sampler_name, max(cumulative_budget), len(selection_features)
            )
            master_selected = get_sampler(
                name=sampler_name,
                image_embeddings=selection_features,
                max_budget=max(cumulative_budget),
                trace=master_trace,
                **common,
            )

        results: Dict[int, Dict[str, float]] = {}
        sweep_watch = Stopwatch(len(cumulative_budget), f"{output_name} budgets")

        for budget in cumulative_budget:
            set_seed(random_seed)
            budget_started = time.time()
            trace: Optional[SelectionTrace] = None
            if master_selected is not None and spec.is_prefix_exact(budget, num_classes):
                selected_indices = master_selected[:budget]
                # The shared run's steps beyond this budget did not influence
                # this prefix, so only the first `budget` of them are its trace.
                trace = _prefix_trace(master_trace, budget)
            else:
                trace = SelectionTrace(sampler_name, budget, len(selection_features))
                selected_indices = get_sampler(
                    name=sampler_name,
                    image_embeddings=selection_features,
                    max_budget=budget,
                    trace=trace,
                    **common,
                )

            labeled_features = train_features[selected_indices]
            labeled_labels = train_labels[selected_indices]
            selection_seconds = time.time() - budget_started

            # Samplers that do not use the shared coverage greedy record no
            # steps of their own. Backfill the pick ORDER for them, so the
            # selection sequence — which every sampler has, and which the
            # order-degeneracy check reads — is present for all of them. The
            # per-step score stays absent because it genuinely does not exist.
            if trace is not None and not trace.steps:
                for index in selected_indices:
                    trace.add_step(int(index))
            trace_payload = None if trace is None else trace.to_payload()
            sanity = check_selection(
                selected_indices, len(selection_features),
                labels=train_labels, num_classes=num_classes, trace=trace_payload,
            )
            print(format_sanity_report(sanity, output_name, budget))

            _save(
                os.path.join(save_dir, f"{output_name}_selected_budget_{budget}.pt"),
                {
                    "selected_indices": list(selected_indices),
                    "selected_labels": labeled_labels.tolist(),
                    "selected_sample_ids": [train_sample_ids[i] for i in selected_indices],
                    "sampler": sampler_name,
                    "run_name": output_name,
                    "sampler_config": sampler_cfg,
                    "budget": budget,
                    "seed": random_seed,
                    "dataset": dataset_key,
                    "num_classes": num_classes,
                    "pool_size": len(selection_features),
                    "class_names": list(class_names),
                    "label_counts": np.bincount(
                        labeled_labels.astype(np.int64), minlength=num_classes
                    ).tolist(),
                    "selection_seconds": selection_seconds,
                    "trace": trace_payload,
                    "sanity": sanity,
                    "spec": {
                        "passes": spec.passes,
                        "prefix_exact": spec.prefix_exact,
                        "prefix_used": bool(
                            master_selected is not None
                            and spec.is_prefix_exact(budget, num_classes)
                        ),
                    },
                    "visual_backbone": visual_backbone,
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
            results[budget] = {
                "acc": accuracy, "precision": precision, "recall": recall, "f1": f1,
                "selection_seconds": selection_seconds,
                "sanity_severity": sanity["severity"],
            }
            save_probe(
                probe,
                os.path.join(save_dir, f"{output_name}_probe_budget_{budget}.pt"),
                metadata={
                    "run_name": output_name,
                    "sampler": sampler_name,
                    "budget": budget,
                    "seed": random_seed,
                    "dataset": dataset_key,
                    "class_names": list(class_names),
                    "metrics": results[budget],
                },
            )
            # Test-set predictions are what a confusion matrix or a per-class
            # error plot needs, and re-deriving them means re-running the
            # backbone over the whole test set.
            _save(
                os.path.join(save_dir, f"{output_name}_predictions_budget_{budget}.pt"),
                {
                    "run_name": output_name,
                    "budget": budget,
                    "probabilities": probe.predict_proba(test_features, device),
                    "test_labels": np.asarray(test_labels),
                    "class_names": list(class_names),
                    "test_fingerprint": test_fingerprint,
                },
            )
            del probe
            clear_memory()
            sweep_watch.advance()
            print(f"[sweep] {sweep_watch.line()}")

        _save(
            os.path.join(save_dir, f"{output_name}{shard_suffix}_results.pt"),
            {
                "sampler": sampler_name,
                "run_name": output_name,
                "sampler_config": sampler_cfg,
                "budgets": cumulative_budget,
                "linear": results,
                "seed": random_seed,
                "dataset": dataset_key,
                "num_classes": num_classes,
                "class_names": list(class_names),
                "visual_backbone": visual_backbone,
                "probe_epochs": probe_epochs,
                "probe_lr": probe_lr,
                "spec": {"passes": spec.passes, "prefix_exact": spec.prefix_exact},
                "total_seconds": time.time() - run_started,
                "train_fingerprint": train_fingerprint,
                "test_fingerprint": test_fingerprint,
                "cellvit_manifest": cellvit_manifest,
            },
        )
        worst = max(
            (results[budget]["sanity_severity"] for budget in results),
            key=lambda level: SEVERITY_ORDER.index(level),
            default="ok",
        )
        print(
            f"\n{'=' * 68}\n{output_name}: {len(results)} budgets in "
            f"{format_duration(time.time() - run_started)} | worst sanity: "
            f"{worst.upper()}\n{'=' * 68}"
        )


def merge_budget_shards(
    save_dir: str, output_name: str, shard_tags: List[str]
) -> Dict[int, Dict[str, float]]:
    """Fold per-shard results files into the single `<run>_results.pt`.

    A budget-sharded run writes one results table per shard. Everything else it
    writes is already named by budget and needs no merging. The merged file is
    byte-for-byte the shape an unsharded run produces, so nothing downstream —
    `evaluate_al_sampler.ipynb` included — has to know a run was sharded.

    Shard files are left in place rather than deleted: they carry each shard's
    own timings, and a partially-finished sweep is worth being able to inspect.
    """
    merged: Dict[int, Dict[str, float]] = {}
    budgets: List[int] = []
    base: Optional[Dict] = None
    missing: List[str] = []

    for tag in shard_tags:
        path = os.path.join(save_dir, f"{output_name}_{tag}_results.pt")
        if not os.path.isfile(path):
            missing.append(tag)
            continue
        payload = torch.load(path, weights_only=False)
        if base is None:
            base = payload
        overlap = set(payload["linear"]) & set(merged)
        if overlap:
            raise ValueError(
                f"Budget shards overlap on {sorted(overlap)}: two shards ran the "
                "same budget, so one overwrote the other's per-budget files."
            )
        merged.update(payload["linear"])
        budgets.extend(payload["budgets"])

    if base is None:
        raise FileNotFoundError(
            f"No shard results found for {output_name!r} in {save_dir!r}"
        )
    if missing:
        raise FileNotFoundError(
            f"Shards {missing} produced no results file; the merged table would "
            "silently be missing their budgets. Re-run them before merging."
        )

    payload = dict(base)
    payload["linear"] = {budget: merged[budget] for budget in sorted(merged)}
    payload["budgets"] = sorted(budgets)
    payload["run_name"] = output_name
    payload["sharded_over"] = list(shard_tags)
    # Per-shard wall clock does not add up to anything meaningful once the
    # shards ran concurrently, so it is dropped rather than summed into a
    # number that would read as the run's duration.
    payload.pop("total_seconds", None)
    _save(os.path.join(save_dir, f"{output_name}_results.pt"), payload)
    return payload["linear"]


def run_on_worker(**kwargs) -> None:
    """`run` with the device resolved inside the process that will use it.

    `utils.parallel` pins one GPU per worker through `CUDA_VISIBLE_DEVICES`
    before torch initialises, so inside a worker the pinned card is always
    `cuda:0`. A `torch.device` built in the parent would also not survive
    pickling meaningfully, hence the string here.
    """
    device_string = kwargs.pop("device_string", "cuda:0")
    run(device=torch.device(device_string), **kwargs)


def _dataset_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "lbl"):
        return dataset.lbl
    return np.array(dataset.dataset.targets)[dataset.indices]


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
