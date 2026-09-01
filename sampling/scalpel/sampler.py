"""SCALPEL — Uncertainty Herding driven by visual/cell disagreement.

One objective, evaluated the same way in every round, is Uncertainty Herding's
weighted facility location on the DINOv2 space that the evaluation probe also
lives in:

    x* = argmax_i  sum_n  U_n * max(k_sigma(x_n, x_i) - K_n, 0)

Everything Uncertainty Herding specifies is kept as specified: the Gaussian
kernel, the running-max update, the radius adaptation `sigma = min pairwise
distance over the labeled set` recomputed per round (its Proposition 4, the
reason one method works across budgets), and temperature-calibrated logits.

The single substitution is `U`:

* round 1 has no labels, so `U = 1` and the objective is exactly MaxHerding —
  broad coverage, no cold-start guessing. This is the hook where a label-free
  semantic prior would go if one is added later.
* later rounds set `U` from how much a DINOv2 probe and a CellViT cell probe
  disagree on the same patch (see `.uncertainty`), falling back to the visual
  margin wherever no nucleus was detected and the cell probe therefore has no
  opinion. Both terms are RANK-normalized before being combined: raw, the
  disagreement (~0.02) and the margin (~0.7) are two orders of magnitude
  apart, and the fallback branch swamped the objective -- see `.uncertainty`.

`uncertainty_mode="visual_margin"` runs the identical loop with Uncertainty
Herding's own margin weight, which is the controlled ablation for measuring
what the cell view contributes.
"""

import time
from typing import List

import numpy as np
import torch

from utils.progress import format_duration
from utils.runtime import clear_memory
from ..kernels import (
    greedy_weighted_coverage,
    labeled_min_sigma,
    bootstrap_sigma,
    running_max_coverage,
)
from ..registry import register_sampler
from .uncertainty import MAX_TEMPERATURE, UNCERTAINTY_MODES, round_weights
from .views import build_cell_view, normalize_rows


@register_sampler("scalpel")
def scalpel_sampling(**kwargs) -> List[int]:
    visual_raw = np.asarray(kwargs["image_embeddings"], dtype=np.float32)
    cell_raw = np.asarray(kwargs["cell_embeddings"], dtype=np.float32)
    reliability = np.clip(np.asarray(kwargs["cell_reliability"], dtype=np.float32), 0.0, 1.0)
    labels = np.asarray(kwargs["oracle_labels"])
    num_classes = int(kwargs["num_classes"])
    max_budget = int(kwargs["max_budget"])
    device = kwargs["device"]

    uncertainty_mode = kwargs.get("uncertainty_mode", "disagreement")
    missing_impute = kwargs.get("missing_impute", "mean")
    num_rounds = int(kwargs.get("num_rounds", 5))
    chunk_size = int(kwargs.get("chunk_size", 2000))
    n_sigma = int(kwargs.get("n_sigma", 2000))
    # Mirrors config/config.yaml. A stale default here is not a runtime bug
    # (config.yaml always supplies the value via main.py) but it misleads
    # anyone reading this file for what the method actually runs.
    sigma_floor_ratio = float(kwargs.get("sigma_floor_ratio", 0.10))
    # 100 in config/config.yaml, matching training.probe_epochs so selection
    # and the evaluation probe share one budget. Kept in sync here for the
    # same reason as sigma_floor_ratio above.
    probe_epochs = int(kwargs.get("probe_epochs", 100))
    probe_lr = float(kwargs.get("probe_lr", 1e-3))
    probe_weight_decay = float(kwargs.get("probe_weight_decay", 1e-4))
    consistency_weight = float(kwargs.get("consistency_weight", 0.0))
    consistency_mode = kwargs.get("consistency_mode", "symmetric_js")
    # Overridable so the cap can be A/B tested from the notebook without
    # editing code (OVERRIDES={"max_temperature": 19.9} reproduces the
    # uncapped behaviour). Mirrors config/config.yaml.
    max_temperature = float(kwargs.get("max_temperature", MAX_TEMPERATURE))
    diag = bool(kwargs.get("diag", True))
    # Set by main.py only when AUGMENT != "none": a callable mapping pool
    # indices to freshly augmented features for the visual probe's TRAINING
    # rows. None keeps the frozen-cache behaviour exactly.
    augmented_feature_provider = kwargs.get("augmented_feature_provider")

    if uncertainty_mode not in UNCERTAINTY_MODES:
        raise ValueError(f"uncertainty_mode must be one of {list(UNCERTAINTY_MODES)}")
    if not len(visual_raw) == len(cell_raw) == len(reliability) == len(labels):
        raise ValueError("visual, cell, reliability and label arrays must align by patch")
    if visual_raw.ndim != 2 or cell_raw.ndim != 2:
        raise ValueError("image_embeddings and cell_embeddings must be 2-D")

    num_samples = len(visual_raw)
    budget = min(max_budget, num_samples)
    rounds = max(1, min(num_rounds, budget))
    base, remainder = divmod(budget, rounds)
    round_sizes = [base + (1 if r < remainder else 0) for r in range(rounds)]

    coverage_features = normalize_rows(visual_raw, device)
    cell_view, missing_fraction = build_cell_view(
        cell_raw, reliability, missing_impute, device
    )
    visual_np = coverage_features.cpu().numpy()
    cell_np = cell_view.cpu().numpy()
    del cell_view
    clear_memory()

    # Round-1 bandwidth, and the floor for later rounds: near-duplicate tissue
    # tiles can drive the labeled min-pairwise distance to ~0, which collapses
    # the kernel into an indicator of duplicates.
    pool_sigma = bootstrap_sigma(coverage_features, n_ref=n_sigma)
    sigma_floor = sigma_floor_ratio * pool_sigma
    sigma = pool_sigma

    selected: List[int] = []
    selected_set: set = set()
    trace = kwargs.get("trace")

    for round_index in range(rounds):
        n_select = round_sizes[round_index]
        if n_select <= 0:
            continue
        started = time.time()
        if trace is not None:
            trace.start_round(round_index)

        if round_index == 0:
            weights_np = None
            # Same keys round_weights returns, so the trace schema is
            # identical for round 0 (which never calls it) and every later
            # round.
            diagnostics = {
                "tau_visual": 1.0, "tau_cell": 1.0,
                "mean_disagreement": float("nan"), "tau_at_cap": 0.0,
            }
        else:
            sigma = max(labeled_min_sigma(coverage_features, selected, sigma), sigma_floor)
            weights_np, diagnostics = round_weights(
                visual_np, cell_np, reliability, labels, selected, num_classes, device,
                uncertainty_mode=uncertainty_mode,
                probe_epochs=probe_epochs,
                probe_lr=probe_lr,
                probe_weight_decay=probe_weight_decay,
                consistency_weight=consistency_weight,
                consistency_mode=consistency_mode,
                augmented_feature_provider=augmented_feature_provider,
                max_temperature=max_temperature,
            )

        if weights_np is None:
            weights = torch.ones(num_samples, device=device, dtype=torch.float32)
        else:
            weights = torch.as_tensor(weights_np, device=device, dtype=torch.float32)
            if float(weights.max()) <= 0.0:
                # An all-zero weight vector makes every marginal gain zero, and
                # argmax would then silently return index order.
                weights.fill_(1.0)

        # sigma changed, so the previous round's running max was computed at a
        # different bandwidth and cannot be carried forward.
        coverage = running_max_coverage(coverage_features, selected, sigma, chunk_size)
        picks = greedy_weighted_coverage(
            coverage_features, weights, coverage, sigma, n_select, selected_set, chunk_size,
            trace=trace, desc=f"scalpel b={budget} r={round_index}",
        )
        selected.extend(picks)
        elapsed = time.time() - started

        if trace is not None:
            trace.add_round(
                num_selected=len(picks),
                seconds=elapsed,
                sigma=sigma,
                weights=weights.detach().cpu().numpy(),
                uncertainty_mode=uncertainty_mode,
                missing_fraction=missing_fraction,
                # Round 1 has no labels, so U=1 and the objective is exactly
                # MaxHerding. Flag it so a checker does not read the constant
                # weight vector as a degenerate one.
                weight_uniform_by_design=(weights_np is None),
                **{
                    key: float(value)
                    for key, value in diagnostics.items()
                    if isinstance(value, (int, float))
                },
            )

        if diag:
            disagreement = diagnostics["mean_disagreement"]
            print(
                f"[scalpel b={budget} r={round_index}] picked={len(picks)} "
                f"in {format_duration(elapsed)} | mode={uncertainty_mode} "
                f"sigma={sigma:.4f} missing={missing_fraction:.3f} "
                f"tau={diagnostics['tau_visual']:.1f}/{diagnostics['tau_cell']:.1f} "
                f"js={'n/a' if np.isnan(disagreement) else f'{disagreement:.4f}'}"
            )

        del weights, coverage
        clear_memory()
        if len(picks) < n_select:
            break

    del coverage_features
    clear_memory()
    return selected
