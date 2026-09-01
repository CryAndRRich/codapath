"""SCALPEL's per-round acquisition weight.

Uncertainty Herding weights each pool point by how close the classifier is to
a decision boundary there (`1 - margin`). SCALPEL replaces that weight with how
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
    consistency_weight: float = 0.0,
    consistency_mode: str = "symmetric_js",
    augmented_feature_provider=None,
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
    diagnostics: Dict[str, float] = {
        "tau_visual": 1.0,
        "tau_cell": 1.0,
        "mean_disagreement": float("nan"),
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
        probe_epochs, probe_lr, device,
    )
    cell_trainable = _has_two_classes(labels[cell_index])
    tau_cell = (
        calibrate_temperature(
            cell_features, labels, cell_index.tolist(), num_classes,
            probe_epochs, probe_lr, device,
        )
        if cell_trainable else 1.0
    )
    diagnostics["tau_visual"] = float(tau_visual)
    diagnostics["tau_cell"] = float(tau_cell)

    want_cell = cell_trainable and uncertainty_mode == "disagreement"
    if want_cell and consistency_weight > 0.0:
        visual_probe, cell_probe = train_dual_probe(
            visual_train,
            cell_features[selected_index],
            labels[selected_index],
            num_classes,
            probe_epochs,
            probe_lr,
            device,
            cell_valid=valid[selected_index],
            cell_reliability=reliability[selected_index],
            consistency_weight=consistency_weight,
            consistency_mode=consistency_mode,
            weight_decay=probe_weight_decay,
        )
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
