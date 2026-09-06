"""PACT's per-round acquisition weight.

Uncertainty Herding weights each pool point by how close the classifier is to
a decision boundary there (`1 - margin`). PACT replaces that weight with how
much two views of the same patch DISAGREE: a probe on full-image DINOv2
features and a probe on pooled CellViT cell embeddings. A patch both heads read
the same way is not informative even when neither is confident; a patch where
tissue-level and cell-level evidence point at different classes is exactly
where a label buys the most.

Both probes are temperature-calibrated before comparison, otherwise the
divergence mostly reflects the two heads' different over-confidence rather
than genuine disagreement.

**The weight is built from RANKS, not raw values.** It combines two
quantities on very different scales -- Jensen-Shannon disagreement between
the probes, and the visual margin used wherever no cell view exists -- and
mixing them raw let the margin branch dominate completely. Each term is
rank-normalized over the set of points it is defined for before they are
combined; see the comment on the mixing line for what that cost when it was
missing, and `sampling.uncertainty.rank_normalize` for why ranks rather than
min-max.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from utils.runtime import clear_memory
from ..calibration import calibrate_temperature
from ..uncertainty import (
    js_disagreement_from_logits,
    margin_uncertainty_from_logits,
    rank_normalize,
)

UNCERTAINTY_MODES = ("disagreement", "visual_margin")

# Ceiling for the temperature search, tighter than the paper grid's 19.9.
#
# Uncertainty Herding calibrates ONE probe, where a large temperature merely
# flattens a single confidence. PACT compares TWO, and dividing both by a
# large T pushes each softmax toward uniform -- and two near-uniform
# distributions cannot disagree. Measured on 14-class logits from two
# genuinely different probes, the mean Jensen-Shannon divergence retained is:
#
#     T=1.0  100%      T=3.0  25.1%      T=7.0   5.2%
#     T=2.0   47.6%    T=5.0   9.9%      T=19.9  0.7%
#
# A real histoset run logged `tau=19.9/19.9 js=0.0001`: the search hit the
# grid ceiling and destroyed the disagreement signal, which is the entire
# contribution this method makes over Uncertainty Herding. 5.0 keeps ~14x
# more of that signal than 19.9 while still permitting genuine calibration
# (the useful range at these budgets is small T; a ceiling value is anyway
# fitted on a validation split that can be ~7 points across 14 classes).
MAX_TEMPERATURE = 5.0


def _has_two_classes(labels: np.ndarray) -> bool:
    return len(labels) >= 2 and len(np.unique(labels)) >= 2


def round_weights(
    visual_features: np.ndarray,
    cell_features: np.ndarray,
    reliability: np.ndarray,
    labels: np.ndarray,
    selected: Sequence[int],
    num_classes: int,
    device: torch.device,
    uncertainty_mode: str = "disagreement",
    probe_epochs: int = 50,
    probe_lr: float = 1e-3,
    probe_weight_decay: float = 1e-4,
    pool_consistency_weight: float = 0.0,
    pool_confidence_quantile: float = 0.5,
    augmented_feature_provider=None,
    max_temperature: float = MAX_TEMPERATURE,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Return `(per-pool weights in [0, 1], diagnostics)`.

    `None` weights mean "not enough labels to fit anything yet" — the caller
    falls back to uniform weights, which reduces the objective to plain
    MaxHerding coverage.

    `uncertainty_mode="visual_margin"` is the controlled ablation: identical
    machinery, but the weight is Uncertainty Herding's own calibrated margin.
    Comparing the two isolates what the cell view actually contributes.

    `augmented_feature_provider`, when given, is
    `training.finetune.make_augmented_feature_provider`'s closure: it maps a
    list of pool indices to freshly AUGMENTED features for exactly those
    rows. It replaces the frozen `visual_features[selected_index]` rows used
    to TRAIN the visual probe (and to calibrate its temperature), so a run
    with `AUGMENT != "none"` augments during selection too, not only in the
    final-training pass. Everything else still reads the frozen cache: the
    pool-wide `predict_logits` below, and the cell probe (CellViT embeddings
    have no pixels behind them to augment). This module stays free of any
    pixel/transform/encoder knowledge -- it only ever calls the opaque
    callable.
    """
    from training.probe import train_dual_probe, train_probe

    if uncertainty_mode not in UNCERTAINTY_MODES:
        raise ValueError(f"uncertainty_mode must be one of {list(UNCERTAINTY_MODES)}")

    selected_index = np.asarray(selected, dtype=np.int64)
    valid = reliability > 0.0
    cell_index = selected_index[valid[selected_index]]
    # Every key present on every path: the sampler copies this dict straight
    # into the per-round trace, so a key that appears only on some rounds
    # gives the saved trace an inconsistent schema and any reader comparing
    # rounds hits a KeyError.
    diagnostics: Dict[str, float] = {
        "tau_visual": 1.0,
        "tau_cell": 1.0,
        "mean_disagreement": float("nan"),
        "tau_at_cap": 0.0,
        # NaN, not 0.0: "the term did not run" and "it ran and masked nothing"
        # are different facts, and 0.0 would conflate them.
        "pool_mask_fraction": float("nan"),
    }

    if not _has_two_classes(labels[selected_index]):
        return None, diagnostics

    # The visual probe's TRAINING rows -- augmented when a provider is given,
    # frozen otherwise. `visual_features` itself is never rebuilt: the
    # pool-wide `predict_logits` further down must stay on the frozen cache
    # (augmenting 90k rows per round is ~27x the cost of augmenting only the
    # labeled set, which is the whole reason this is affordable).
    if augmented_feature_provider is not None:
        visual_train = augmented_feature_provider(selected_index.tolist())
        if visual_train.shape[0] != len(selected_index):
            raise ValueError(
                f"augmented_feature_provider returned {visual_train.shape[0]} rows "
                f"for {len(selected_index)} selected indices"
            )
        if visual_train.shape[1] != visual_features.shape[1]:
            raise ValueError(
                f"augmented_feature_provider returned width {visual_train.shape[1]}, "
                f"but the frozen pool cache is {visual_features.shape[1]}-d -- the "
                "probe trained on one cannot score the other"
            )
    else:
        visual_train = visual_features[selected_index]

    tau_visual = calibrate_temperature(
        visual_train, labels[selected_index], list(range(len(selected_index))), num_classes,
        probe_epochs, probe_lr, device, max_temperature=max_temperature,
    )
    cell_trainable = _has_two_classes(labels[cell_index])
    tau_cell = (
        calibrate_temperature(
            cell_features, labels, cell_index.tolist(), num_classes,
            probe_epochs, probe_lr, device, max_temperature=max_temperature,
        )
        if cell_trainable else 1.0
    )
    diagnostics["tau_visual"] = float(tau_visual)
    diagnostics["tau_cell"] = float(tau_cell)
    # A temperature sitting exactly on the cap means the ECE search wanted to
    # go further and was stopped. That is worth seeing: it says the probes are
    # so over-confident (or the validation split so small) that calibration
    # is fighting the signal rather than sharpening it.
    diagnostics["tau_at_cap"] = float(
        max_temperature is not None
        and (tau_visual >= max_temperature - 1e-9 or tau_cell >= max_temperature - 1e-9)
    )

    want_cell = cell_trainable and uncertainty_mode == "disagreement"
    # The dual path is what trains both heads in ONE optimizer, which is the
    # only place the coupling term can be computed at all.
    if want_cell and pool_consistency_weight > 0.0:
        visual_probe, cell_probe, dual_diag = train_dual_probe(
            visual_train,
            cell_features[selected_index],
            labels[selected_index],
            num_classes,
            probe_epochs,
            probe_lr,
            device,
            cell_valid=valid[selected_index],
            weight_decay=probe_weight_decay,
            # The UNLABELED pool, in the same two views the probes read. Rows
            # are passed whole; `train_dual_probe` does the subsampling, so
            # the draw is deterministic per round rather than depending on
            # whatever RNG state selection happened to leave behind.
            pool_visual_features=visual_features if pool_consistency_weight > 0.0 else None,
            pool_cell_features=cell_features if pool_consistency_weight > 0.0 else None,
            pool_consistency_weight=pool_consistency_weight,
            pool_confidence_quantile=pool_confidence_quantile,
        )
        # How much of the pool actually cleared the confidence gate. A run
        # where this stays 0 configured the term but never applied it -- an
        # absolute threshold is dataset-dependent, so this must be visible in
        # the trace rather than inferred from the score.
        diagnostics["pool_mask_fraction"] = float(dual_diag["pool_mask_fraction"])
    else:
        visual_probe = train_probe(
            visual_train, labels[selected_index], num_classes,
            probe_epochs, probe_lr, device, weight_decay=probe_weight_decay,
        )
        cell_probe = (
            train_probe(
                cell_features[cell_index], labels[cell_index], num_classes,
                probe_epochs, probe_lr, device, weight_decay=probe_weight_decay,
            )
            if want_cell else None
        )

    visual_logits = visual_probe.predict_logits(visual_features, device) / tau_visual
    visual_margin = margin_uncertainty_from_logits(visual_logits)
    del visual_probe

    if cell_probe is None:
        # Rank-normalized for the same reason, and just as importantly for
        # COMPARABILITY: `uncertainty_mode="visual_margin"` is the controlled
        # ablation that isolates what the cell view contributes, so it has to
        # differ from the disagreement mode in the weight's MEANING only, not
        # in how that weight is scaled. A monotone transform of U does not
        # preserve the greedy argmax -- U weights each pool ROW inside the
        # sum, so rescaling one row relative to another changes which
        # candidate wins (measured: 6 of 20 picks changed between raw and
        # ranked margin on the same features). Leaving one branch raw would
        # have made the ablation partly a test of normalization.
        weights = rank_normalize(visual_margin)
    else:
        cell_logits = cell_probe.predict_logits(cell_features, device) / tau_cell
        disagreement = js_disagreement_from_logits(visual_logits, cell_logits)

        # Patches with no nucleus have no cell opinion to disagree with, so
        # they fall back to the visual margin in proportion to how unreliable
        # their cell view is. rho=1 is pure disagreement, rho=0 is pure
        # margin. That fallback is deliberate and stays: a patch the cell
        # probe cannot speak for still needs SOME uncertainty signal, and the
        # visual margin is the only one left. Giving it 0 instead would
        # permanently exclude ~9% of the pool from every round after the
        # first, however hard those patches actually are.
        #
        # **The two terms are rank-normalized WITHIN THEIR OWN GROUP before
        # being combined**, because raw they are not on the same scale at all
        # -- measured on a real 14-class histoset run, JS averaged 0.02 while
        # the margin averaged ~0.7. Mixing them raw made every one of the
        # top-200 weights a nucleus-free patch (a fair share is ~18 of 200),
        # and since nucleus-free patches are not spread evenly over classes
        # (ovarian tissue is sparser than lung or colon), rounds 1-4 of that
        # run put 150 of 160 picks into 5 of the 14 classes and never sampled
        # 4 classes at all. Round 0, which uses U=1 and no weight, covered
        # all 14 -- so the collapse was the weight, not the coverage term.
        #
        # Each term is ranked over the set of points that term is DEFINED
        # for, which is not the same set for both:
        #
        # * JS exists only where a cell probe had an opinion, so it is ranked
        #   within `valid`. Ranking it against patches whose JS is discarded
        #   would compare it to numbers that never enter the objective.
        # * The margin exists EVERYWHERE, and with
        #   `reliability_mode="mean_confidence"` reliability is continuous, so
        #   a patch with rho=0.3 draws 70% of its weight from the margin while
        #   still being `valid`. Ranking the margin only within the
        #   nucleus-free group would hand every such patch a margin term of 0
        #   -- silently zeroing most of the weight for every partially
        #   reliable point. Ranked pool-wide, it is defined for all of them.
        #   (Under the default `reliability_mode="valid"` rho is exactly 0 or
        #   1, so this distinction changes nothing there; it only keeps the
        #   continuous mode from breaking.)
        #
        # rank_normalize maps a constant input to a constant 0.5, not to
        # zeros, so a degenerate round yields a neutral weight instead of an
        # all-zero one that would collapse every marginal gain (CLAUDE.md's
        # min-max-on-a-constant-vector lesson).
        weights = (
            reliability * rank_normalize(disagreement, valid)
            + (1.0 - reliability) * rank_normalize(visual_margin)
        )
        diagnostics["mean_disagreement"] = float(
            disagreement[valid].mean() if valid.any() else 0.0
        )
        del cell_probe, cell_logits

    del visual_logits
    clear_memory()
    return np.clip(weights, 0.0, 1.0).astype(np.float32), diagnostics


def text_prior_weights(
    image_embeddings: np.ndarray,
    text_prototypes: np.ndarray,
    logit_scale: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Round-1 acquisition weight from image-text similarity alone.

    This is contribution #1: the cold start. Round 1 has no labels, so the
    probes that produce every later round's weight cannot exist, and the method
    falls back to pure MaxHerding (`U = 1`) -- coverage with no notion of which
    regions are hard. A VLM's text tower supplies exactly the missing piece
    without a single label: encode one prototype per class, and a patch whose
    top two classes are nearly tied is one the model cannot place.

    The weight is Uncertainty Herding's own margin, computed on zero-shot
    logits instead of a trained probe's:

        logits  = (image @ text.T) * logit_scale
        weight  = 1 - (p_top1 - p_top2)   after softmax

    so it is the SAME quantity every later round uses, only sourced
    differently. That is deliberate: round 1 then differs from rounds 2+ in
    where the uncertainty comes from, not in what "uncertain" means, and the
    greedy coverage objective underneath is untouched.

    Returned ranked, like every other term in this module. `rank_normalize`
    maps a constant input to 0.5 rather than to zeros, so degenerate
    prototypes yield uniform weights -- i.e. exactly the MaxHerding this
    replaces -- instead of an all-zero vector that would collapse every
    marginal gain and make `argmax` return index order.

    `logit_scale` does NOT change the ranking. `argmax`, and the ORDER of
    `1 - margin`, are invariant to a positive scale on the logits... but the
    margin's VALUE is not, and neither is its rank ordering across points,
    because softmax is not order-preserving in the gap between top-2
    probabilities under different temperatures. Pass the model's own learned
    `logit_scale.exp()` when it is known; the default 1.0 is the raw cosine.

    `image @ text.T` is read AS a cosine, the same contract
    `sampling/kernels.py` documents, so both sides must be unit rows. The two
    sides are handled DIFFERENTLY on purpose:

    * text prototypes are CHECKED and a non-unit row raises. They are 14-ish
      vectors written once by the extraction notebook; a wrong norm there is a
      bug in that notebook, and it measurably shifts the ranking (~1.5% of
      predictions move), so failing loudly is right.
    * image rows are NORMALIZED here rather than checked. A caller may hand in
      RAW_SPACE features, which are not unit by construction, and rescaling a
      whole image row is uniform across classes -- it cannot change that row's
      argmax or its top-2 gap. Rejecting them would force every caller to
      duplicate this one line.
    """
    image_embeddings = np.asarray(image_embeddings, dtype=np.float32)
    text_prototypes = np.asarray(text_prototypes, dtype=np.float32)
    if image_embeddings.ndim != 2 or text_prototypes.ndim != 2:
        raise ValueError("image_embeddings and text_prototypes must be 2-D")
    if image_embeddings.shape[1] != text_prototypes.shape[1]:
        raise ValueError(
            f"image features are {image_embeddings.shape[1]}-d but text "
            f"prototypes are {text_prototypes.shape[1]}-d; these are compared "
            "by a dot product and must share a space"
        )
    if len(text_prototypes) < 2:
        raise ValueError(
            "a margin needs at least two classes to take a top-2 difference"
        )
    if float(logit_scale) <= 0.0:
        raise ValueError("logit_scale must be positive")

    text_norms = np.linalg.norm(text_prototypes, axis=1)
    if not np.all(np.abs(text_norms - 1.0) < 1e-3):
        raise ValueError(
            "text prototypes must be L2-normalized per row; "
            f"norms range [{text_norms.min():.4f}, {text_norms.max():.4f}]"
        )
    image_norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    unit_images = image_embeddings / np.maximum(image_norms, 1e-12)

    logits = (unit_images @ text_prototypes.T) * float(logit_scale)
    margin = margin_uncertainty_from_logits(logits)

    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)

    diagnostics = {
        "text_mean_margin": float(margin.mean()),
        "text_mean_top1": float(probabilities.max(axis=1).mean()),
        # How many classes the zero-shot head actually uses. A prior that puts
        # every patch in one class carries no information about WHICH regions
        # are hard, even though its margins may vary -- worth seeing in the
        # trace rather than inferring from accuracy nobody computed here.
        "text_classes_predicted": float(len(np.unique(predictions))),
        "text_num_classes": float(len(text_prototypes)),
    }
    return rank_normalize(margin), diagnostics
