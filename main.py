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
import json
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
from features.vlm import RAW_SPACE, vlm_feature_cache_paths
from features.visual import get_or_extract_features
from sampling import get_sampler
from sampling.specs import spec_for
from training.checkpoint import save_probe
from training.finetune import finetune_and_evaluate, needs_pixels
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


def _default_run_name(
    sampler_name: str,
    sampler_cfg: Dict,
    *,
    encoder: str = "dinov2",
    use_text: bool = False,
    final_train_cfg: Optional[Dict] = None,
) -> str:
    """Encode the config axes that would otherwise overwrite each other.

    `encoder` and `use_text` default to the values every run had before
    `features/vlm.py` existed, so a caller that does not pass them gets
    EXACTLY the old name back (`test_default_run_name_is_unchanged`) --
    every already-published baseline/scalpel run stays resumable and
    unorphaned.

    Without these two params in the signature at all, a DINOv2 run and a
    CONCH run of the same sampler+config could not be told apart by name:
    `_default_run_name('uncertainty_herding', {})` returns
    `'uncertainty_herding'` regardless of which encoder produced it, so the
    second protocol to run would overwrite every file the first wrote --
    `_results.pt`, `_selected_budget_*.pt`, `_probe_budget_*.pt`,
    `_predictions_budget_*.pt`, its log -- and the notebook's resume check
    (`<name>_results.pt exists`) would then silently SKIP the second
    protocol's run entirely, mistaking the first protocol's leftover file
    for a finished run of the second (PLAN_IMPLEMENT.md §6.2).

    `use_text` is included even though no sampler reads a text prior yet
    (that is `run_al_main.ipynb`'s job, step 10): the round-1 cold-start
    text prior is a config-driven axis of a run, not of the sampler
    function's own kwargs, so it has to be nameable the same way `encoder`
    is, from the same call site, before that call site exists.
    """
    parts = [sampler_name]
    if sampler_name == "scalpel":
        parts.append(sampler_cfg.get("uncertainty_mode", "disagreement"))
        pooling = sampler_cfg.get("cell_pooling", "mean")
        if pooling != "mean":
            parts.append(pooling)
        if sampler_cfg.get("missing_impute", "mean") != "mean":
            parts.append(sampler_cfg["missing_impute"])
        if float(sampler_cfg.get("consistency_weight", 0.0)) > 0.0:
            parts.append(f"cons{sampler_cfg['consistency_weight']}".replace(".", "p"))
    if encoder != "dinov2":
        parts.append(encoder)
    if use_text:
        parts.append("text")

    # The §6.4 final-training axes. Without these in the name, a 48-combination
    # sweep over (cell_pooling x use_lora x aux_loss x augment) collapses onto
    # just 3 names -- 16 runs per name, each overwriting the previous one's
    # `_results.pt`, `_probe_budget_*.pt` and log. Worse than losing them: the
    # notebook's resume check is "does `<name>_results.pt` exist", so runs 2..16
    # of each group would be SKIPPED entirely and the results table would look
    # complete while holding 3 configurations instead of 48.
    #
    # Every part is appended only when it differs from the default, so a run
    # that sets none of them keeps exactly the name it had before this
    # parameter existed (`test_default_run_name_is_unchanged`).
    final_train_cfg = final_train_cfg or {}
    if bool(final_train_cfg.get("use_lora", False)):
        rank = int(final_train_cfg.get("lora_r", 8))
        alpha = float(final_train_cfg.get("lora_alpha", 16.0))
        part = f"lora{rank}"
        if alpha != 16.0:
            part = f"{part}a{alpha}".replace(".", "p")
        parts.append(part)
    aux_loss = final_train_cfg.get("aux_loss", "none")
    if aux_loss != "none":
        part = f"aux{aux_loss}"
        weight = float(final_train_cfg.get("aux_weight", 0.5))
        if weight != 0.5:
            part = f"{part}{weight}".replace(".", "p")
        parts.append(part)
    augment = final_train_cfg.get("augment", "none")
    if augment != "none":
        parts.append(f"aug{augment}")
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
        # `crop_dino` embeddings are produced by a DINOv2 forward pass over
        # each nucleus crop (nucleus/crop_dino.py); the manifest records
        # exactly which DINOv2 checkpoint. Any run whose own visual_backbone
        # is not that same checkpoint -- including every CONCH run, whose
        # `visual_backbone` is a CONCH model id -- would be mixing a CONCH
        # image space with a DINOv2 cell space, which `sampling/scalpel`
        # assumes are the same space (PLAN_IMPLEMENT.md §6.3).
        raise ValueError(
            f"cell_source='crop_dino' was extracted with DINOv2 backbone "
            f"{cache.manifest.get('dino_backbone')!r}, but this run's "
            f"visual_backbone is {visual_backbone!r} -- mixing a different "
            "image encoder (e.g. CONCH) with crop_dino cell embeddings mixes "
            "two incompatible feature spaces. Use cell_source='cellvit_embedding' "
            "instead, which is encoder-independent."
        )

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


def _load_vlm_features(
    vlm_cache_dir: str,
    dataset_key: str,
    random_seed: int,
    vlm_name: str,
    n_train: int,
    n_test: int,
    train_fingerprint: str,
    test_fingerprint: str,
):
    """Read a VLM's RAW_SPACE image features from an already-built cache.

    Unlike DINOv2, an AL run never extracts CONCH features itself: doing so
    would need the `conch` package, an HF token and a 448x448 forward pass
    inside a function that a 2-GPU AL sweep already calls per worker, all to
    duplicate what `extract_vlm_features.ipynb` already does. So this is a
    READ only -- `extract_vlm_features.ipynb` must have published the cache
    first, and a missing or mismatched cache raises rather than falling back
    to extraction.

    Only RAW_SPACE (`proj_contrast=False, normalize=False`) is read -- the
    space every probe and coverage kernel in this project trains on.
    PROJ_SPACE is never read here; using it would not crash (both are 512-d
    for CONCH) but would silently train on features meant only for comparing
    an image against text.
    """
    paths = vlm_feature_cache_paths(vlm_cache_dir, dataset_key, random_seed, vlm_name)
    if not (os.path.exists(paths["train"]) and os.path.exists(paths["test"])
             and os.path.exists(paths["manifest"])):
        raise FileNotFoundError(
            f"No CONCH feature cache for {dataset_key}_seed{random_seed}_{vlm_name} "
            f"under {vlm_cache_dir!r}. Run extract_vlm_features.ipynb with "
            f"DATASET={dataset_key!r}, SEED={random_seed}, VLM={vlm_name!r} first -- "
            "an AL run reads this cache, it does not build it."
        )

    with open(paths["manifest"], "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("dataset") != dataset_key or manifest.get("seed") != random_seed:
        raise ValueError(
            f"CONCH cache manifest at {paths['manifest']} does not match this run "
            f"(dataset={dataset_key!r}, seed={random_seed}): got {manifest}"
        )
    if manifest.get("space") != RAW_SPACE:
        raise ValueError(
            f"{paths['manifest']} declares space={manifest.get('space')!r}, expected "
            f"{RAW_SPACE!r} -- an AL run must never train on PROJ_SPACE features"
        )
    if manifest.get("train_fingerprint") != train_fingerprint or \
            manifest.get("test_fingerprint") != test_fingerprint:
        raise ValueError(
            f"{paths['manifest']} was built from a different train/test split "
            "(sample-order fingerprint mismatch) than this run's dataset -- "
            "re-run extract_vlm_features.ipynb against the current data."
        )

    train_features = np.load(paths["train"])
    test_features = np.load(paths["test"])
    if train_features.shape[0] != n_train or test_features.shape[0] != n_test:
        raise ValueError(
            f"CONCH cache row count does not match this dataset: cache has "
            f"{train_features.shape[0]} train / {test_features.shape[0]} test rows, "
            f"expected {n_train} / {n_test}"
        )
    print(f"[vlm] Loaded RAW_SPACE cache -> {paths['train']}")
    return train_features.astype(np.float32), test_features.astype(np.float32)


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
    image_encoder: str = "dinov2",
    vlm_cache_dir: str = "vlm_features",
    hf_token: Optional[str] = None,
    final_train_cfg: Optional[Dict] = None,
) -> None:
    """
    `image_encoder` ("dinov2" | "conch") decides the ONE feature space every
    stage of this run uses -- the coverage kernel, the disagreement probes
    inside `scalpel`, and the final evaluation probe -- because a run mixing
    encoders across stages is not a comparable protocol
    (PLAN_IMPLEMENT.md §6.2). `"conch"` reads `RAW_SPACE` (proj_contrast=False,
    normalize=False) from `features/vlm.py::get_or_extract_vlm_features`, never
    `PROJ_SPACE` -- that space is only for comparing an image against text, and
    using it here would not crash (both are 512-d) but would silently train
    every probe on features it was never meant for.

    `final_train_cfg`, if given, is a dict with any of `use_lora` (bool),
    `lora_r` (int), `lora_alpha` (float), `aux_loss`
    ("none"|"center"|"supcon"|"triplet"), `aux_weight` (float), `augment`
    ("none"|"flip_rotate") -- the PLAN_IMPLEMENT.md §6.4 final-training pass,
    run AFTER each budget's points are selected, never during selection
    itself. `None` (the default) and a dict where `training.finetune.needs_pixels`
    is False are identical: both keep the exact fast path this project has
    always run -- `training.probe.train_probe` on the frozen embedding
    cache, no pixel loaded. Confirmed choice (PLAN_IMPLEMENT.md §6.4): the
    final-training pass runs at EVERY budget in the sweep, not only the
    largest -- a full learning curve, not a single point, because the
    research question is whether LoRA helps more at low or high budgets.
    """
    if image_encoder not in ("dinov2", "conch"):
        raise ValueError(f"image_encoder must be 'dinov2' or 'conch', got {image_encoder!r}")

    # A token left unset falls back to the ambient environment, which is how
    # `huggingface_hub.login()` (extract_vlm_features.ipynb) and a Kaggle
    # Secret both make themselves visible. Resolved here rather than at the
    # `load_conch` call so the two share one rule.
    hf_token = hf_token or os.environ.get("HF_TOKEN") or None

    final_train_cfg = dict(final_train_cfg or {})
    ft_use_lora = bool(final_train_cfg.get("use_lora", False))
    ft_lora_r = int(final_train_cfg.get("lora_r", 8))
    ft_lora_alpha = float(final_train_cfg.get("lora_alpha", 16.0))
    ft_aux_loss = final_train_cfg.get("aux_loss", "none")
    ft_aux_weight = float(final_train_cfg.get("aux_weight", 0.5))
    ft_augment = final_train_cfg.get("augment", "none")
    run_final_training = needs_pixels(ft_use_lora, ft_augment)

    spec = spec_for(sampler_name)
    output_name = _safe_run_name(
        run_name or _default_run_name(sampler_name, sampler_cfg, encoder=image_encoder)
    )

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

        # `image_encoder` decides which cache this run reads -- never both:
        # the coverage kernel, the disagreement probes, and the evaluation
        # probe all live in whichever ONE space `visual_backbone` names below.
        # The loaders above still use the default DINOv2 transform regardless
        # of `image_encoder`, because nothing here decodes a pixel through
        # them either way (see the num_workers=0 comment above) -- they exist
        # only for labels, sample IDs and the fingerprint, all of which come
        # from `sample_id`, not pixels (tests/test_loaders_transform.py pins
        # this).
        if image_encoder == "conch":
            visual_backbone = model_cfg.get("vlm", "MahmoodLab/CONCH")
            train_features, test_features = _load_vlm_features(
                vlm_cache_dir, dataset_key, random_seed, visual_backbone,
                n_train=len(train_dataset), n_test=len(test_dataset),
                train_fingerprint=train_fingerprint, test_fingerprint=test_fingerprint,
            )
        else:
            visual_backbone = model_cfg.get("vit", "facebook/dinov2-base")
            train_features, test_features = get_or_extract_features(
                train_loader, test_loader, dataset_key, random_seed, visual_backbone,
                device, cache_dir=feature_cache_dir,
                train_fingerprint=train_fingerprint, test_fingerprint=test_fingerprint,
            )
        clear_memory()

        # Final-training encoder (§6.4) -- loaded ONCE here, reused across
        # every budget's final-training pass, never per-budget: reloading a
        # checkpoint 8 times (once per budget in the sweep) would dominate
        # this pass's cost far more than the pass itself does. `None` when
        # this run does not need pixels at all -- the frozen `train_probe`
        # path below never touches it.
        ft_encoder_model = None
        ft_conch_preprocess = None
        if run_final_training:
            if image_encoder == "dinov2":
                from transformers import Dinov2Model

                ft_encoder_model = Dinov2Model.from_pretrained(visual_backbone)
                if ft_use_lora:
                    from training.lora import apply_lora_to_dinov2

                    apply_lora_to_dinov2(ft_encoder_model, r=ft_lora_r, alpha=ft_lora_alpha)
                else:
                    ft_encoder_model.requires_grad_(False)
            else:
                from features.vlm import load_conch

                ft_encoder_model, ft_conch_preprocess = load_conch(
                    visual_backbone, device, hf_token=hf_token
                )
                if ft_use_lora:
                    from training.lora import apply_lora_to_conch

                    apply_lora_to_conch(ft_encoder_model, r=ft_lora_r, alpha=ft_lora_alpha)
                else:
                    ft_encoder_model.requires_grad_(False)
            print(
                f"[final-train] encoder={visual_backbone} use_lora={ft_use_lora} "
                f"aux_loss={ft_aux_loss} augment={ft_augment}"
            )

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

        # AUGMENT reaches into the AL loop too, not just the final-training
        # pass: with `augment != "none"` the per-round uncertainty probe
        # trains on freshly augmented pixels of the LABELED set instead of
        # frozen cache rows, so "augmented" means the same thing in both
        # halves of a run. Only the probe's training rows change -- the
        # coverage kernel, sigma adaptation and the pool-wide uncertainty
        # scoring all still read the frozen cache, because augmenting ~90k
        # pool rows every round costs ~27x what augmenting <=200 labeled rows
        # does. `USE_LORA` deliberately does NOT reach in here: a LoRA-adapted
        # encoder must never select the points it is later trained on.
        selection_augment_provider = None
        selection_encoder_model = None
        if ft_augment != "none":
            from training.finetune import make_augmented_feature_provider

            # A SEPARATE, always-frozen encoder -- never `ft_encoder_model`.
            # That one is LoRA-wrapped and `finetune_and_evaluate` trains it
            # in place, once per budget, so reusing it here would let budget
            # k's fine-tuned weights choose budget k+1's points: training
            # leaking into selection, which no amount of frozen-cache
            # coverage elsewhere would undo. Loading a second copy costs one
            # checkpoint read per run and keeps the two roles honest.
            if image_encoder == "dinov2":
                from transformers import Dinov2Model

                selection_encoder_model = Dinov2Model.from_pretrained(visual_backbone)
                selection_encoder_model.requires_grad_(False)
                selection_preprocess = None
            else:
                from features.vlm import load_conch

                selection_encoder_model, selection_preprocess = load_conch(
                    visual_backbone, device, hf_token=hf_token
                )
                selection_encoder_model.requires_grad_(False)

            selection_augment_provider = make_augmented_feature_provider(
                train_dataset, selection_encoder_model, device,
                image_encoder=image_encoder, augment=ft_augment,
                conch_preprocess=selection_preprocess,
            )
            print(f"[selection] uncertainty probe trains on augment={ft_augment!r} pixels "
                  "(frozen encoder, separate from the final-training one)")

        common = dict(
            oracle_labels=train_labels,
            num_classes=num_classes,
            device=device,
            **sampler_inputs,
            **sampler_cfg,
        )
        if selection_augment_provider is not None:
            common["augmented_feature_provider"] = selection_augment_provider

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
            if run_final_training:
                # §6.4: every budget gets its own final-training pass (the
                # confirmed full-curve choice), reusing the ONE encoder
                # loaded above rather than reloading it per budget.
                probe, ft_metrics = finetune_and_evaluate(
                    train_dataset=train_dataset,
                    selected_indices=selected_indices,
                    labels=train_labels,
                    test_features=test_features,
                    test_labels=test_labels,
                    num_classes=num_classes,
                    device=device,
                    probe_epochs=probe_epochs,
                    probe_lr=probe_lr,
                    image_encoder=image_encoder,
                    use_lora=ft_use_lora,
                    lora_r=ft_lora_r,
                    lora_alpha=ft_lora_alpha,
                    aux_loss=ft_aux_loss,
                    aux_weight=ft_aux_weight,
                    augment=ft_augment,
                    encoder_model=ft_encoder_model,
                    conch_preprocess=ft_conch_preprocess,
                )
                accuracy, precision, recall, f1 = (
                    ft_metrics["acc"], ft_metrics["precision"],
                    ft_metrics["recall"], ft_metrics["f1"],
                )
            else:
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
                    # Which feature space this probe was trained on. Without
                    # this, evaluate_al_sampler.ipynb has no way to tell a
                    # DINOv2 run (768-d) from a CONCH run (512-d, RAW_SPACE)
                    # apart — a dimension mismatch crashes, but if two spaces
                    # ever coincide in width it would silently score a probe
                    # against the wrong test features (PLAN_IMPLEMENT.md §2.1).
                    # `encoder_kind` is explicit rather than inferred from the
                    # `encoder` string (e.g. guessing "dinov2" is in the name)
                    # so a reader never has to pattern-match a HF repo id to
                    # know which cache-loading path applies.
                    "encoder": visual_backbone,
                    "encoder_kind": image_encoder,
                    # Only present when this probe went through the §6.4
                    # final-training pass -- absent (not a False-valued key)
                    # for the ordinary frozen-embedding path, so a reader can
                    # tell "this run never had a final-training axis" apart
                    # from "this run's final-training axes were all off".
                    **({"final_train_cfg": final_train_cfg} if run_final_training else {}),
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

        del ft_encoder_model
        del selection_encoder_model
        clear_memory()

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
                # Present only when the final-training pass ran, same
                # reasoning as the probe metadata's own final_train_cfg key.
                **({"final_train_cfg": final_train_cfg} if run_final_training else {}),
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
    parser.add_argument(
        "--image_encoder", default="dinov2", choices=["dinov2", "conch"],
        help="which frozen image encoder this run's whole pipeline uses",
    )
    parser.add_argument(
        "--vlm_cache_dir", default=None,
        help="directory extract_vlm_features.ipynb published its cache into "
             "(only read when --image_encoder=conch)",
    )
    parser.add_argument(
        "--hf_token", default=None,
        help="Hugging Face token for the gated CONCH checkpoint. Only needed "
             "when --image_encoder=conch AND the final-training pass loads the "
             "model (--use_lora / --augment); reading the feature cache needs "
             "no token. Defaults to the HF_TOKEN environment variable, which "
             "is preferred -- a token on the command line lands in shell "
             "history and in `ps` output.",
    )
    parser.add_argument(
        "--use_lora", action="store_true",
        help="fine-tune image_encoder's attention via LoRA in the final-"
             "training pass (§6.4); needs raw pixels, runs after selection",
    )
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument(
        "--aux_loss", default=None, choices=["none", "center", "supcon", "triplet"],
    )
    parser.add_argument("--aux_weight", type=float, default=None)
    parser.add_argument(
        "--augment", default=None, choices=["none", "flip_rotate"],
    )
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

    # CLI flags override config.yaml's final_training defaults, same
    # precedence as every other --flag/config pair here (probe_epochs,
    # feature_cache_dir, ...).
    final_train_defaults = config.get("final_training", {})
    final_train_cfg = {
        "use_lora": args.use_lora or bool(final_train_defaults.get("use_lora", False)),
        "lora_r": args.lora_r if args.lora_r is not None else final_train_defaults.get("lora_r", 8),
        "lora_alpha": args.lora_alpha if args.lora_alpha is not None
            else final_train_defaults.get("lora_alpha", 16.0),
        "aux_loss": args.aux_loss if args.aux_loss is not None
            else final_train_defaults.get("aux_loss", "none"),
        "aux_weight": args.aux_weight if args.aux_weight is not None
            else final_train_defaults.get("aux_weight", 0.5),
        "augment": args.augment if args.augment is not None
            else final_train_defaults.get("augment", "none"),
    }

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
        image_encoder=args.image_encoder,
        vlm_cache_dir=args.vlm_cache_dir or config.get("vlm_cache_dir", "vlm_features"),
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or None,
        final_train_cfg=final_train_cfg,
    )


if __name__ == "__main__":
    main()
