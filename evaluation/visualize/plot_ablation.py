"""Ablation (A0-A3) and training-framework (T1-T4) figures.

Two figures, one script, because both read the same 72-cell grid and differ only
in which rows they draw:

    plot_ablation.py --table ablation   -> ablation_seeds38_42_611.png
    plot_ablation.py --table framework  -> framework_seeds38_42_611.png
    plot_ablation.py --table backbones  -> backbones_seeds38_42_611.png

Rows are classified BY CONFIGURATION, never by run name. `CLAUDE.md` records
that this project once shipped a table whose A-row labels were rotated one
position while every individual number still verified against an archive, so
existence-checking a number proves nothing about which row it belongs to. The
classifier below reads `uncertainty_mode`, `pool_consistency_weight` and
`visual_backbone` out of each payload, and asserts that no two archives land in
the same (row, dataset, seed) cell -- the duplicate check is what catches a
classifier that silently collapses two variants of one row.

Three axes (text prior, LoRA+center, augment) are NOT in `_results.pt`; they
live in the run name only. They are read from the stem and covered by the same
duplicate assertion.

The backbone is read as three values, not as "CONCH or not": QuiltNet runs carry
`disagreement` + `pool_consistency_weight=5` exactly like the DINOv2 A3 row, so a
two-valued test silently files all 18 of them as A3 and the duplicate assertion
is what would fire.

Style matches `plot_accuracy.py`: same panel geometry, same legend strip, same
dashed grid, so the three figures read as one set.
"""
import argparse
import glob
import os
import statistics

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODAPATH = os.path.dirname(os.path.dirname(HERE))
PROJECT_ROOT = os.path.dirname(CODAPATH)
DATA = os.path.join(PROJECT_ROOT, "data_upload", "selected", "PACT")
ASSETS = os.path.join(CODAPATH, "assets", "img")

BUDGETS = [25, 50, 75, 100, 125, 150, 175, 200]
DATASETS = ["pathmnist", "histoset", "skintissue"]
DATASET_TITLES = {
    "pathmnist": "PathMNIST",
    "histoset": "HistoSet-5x14",
    "skintissue": "SkinTissue",
}

# Cumulative rows, in the order the paper builds them up.
#
# FRAMEWORK now shows the encoder axis only: the frozen DINOv2 baseline (A3 =
# PACT), the CONCH backbone, and the language-guided prior on top of it. The
# LoRA and center-loss rows were dropped from the paper, and with them the
# reason to omit A3 -- the DINOv2 -> CONCH jump IS the claim this figure makes,
# so the bridge row belongs on the same axes.
ABLATION = ["A0", "A1", "A2", "A3"]
FRAMEWORK = ["A3", "T1", "T2"]
# The two VLM backbones with and without the prior, on one set of axes. DINOv2
# is deliberately absent: it sits ~10 points below, so including it compresses
# the four rows this figure exists to separate into one band.
BACKBONES = ["T1", "T2", "Q1", "Q2"]

# The A0..T4 ids are internal run bookkeeping and never appear in the figures or
# the paper: a reader has no way to resolve them. Each row is named by the
# component it adds.
ROW_LABELS = {
    "A0": "Coverage only",
    "A1": "+ Margin",
    "A2": "+ Cell-tissue disagreement",
    "A3": "+ Pool consistency (PACT)",
    "T1": "CONCH backbone",
    "T2": "+ Language-guided prior",
    "T3": "+ LoRA + center loss",
    "T4": "+ Augmentation",
    "Q1": "QuiltNet backbone",
    "Q2": "QuiltNet + language-guided prior",
}

# The highlighted row of each figure keeps the palette's red, matching
# "PACT (Ours)" in the accuracy figure.
HIGHLIGHT = {"ablation": "A3", "framework": "T2", "backbones": "Q2"}

# These two figures carry three or four rows, not ten, so they do NOT share
# curves.py's palette: with so few curves the colours can all be bright and
# still be far apart, which is easier to read than the mixed-luminance set the
# ten-method figure needs. Chosen by maximising the smallest pairwise CIE76
# distance over a bright pool (saturation >= 0.60, CIE L in [46, 78]); the
# closest pair here is 104.7 against the ten-method figure's 46.1.
_PALETTE = ["#e8000b", "#00b0ff", "#00c853", "#aa00ff", "#ff9100", "#e8118f"]
_MARKERS = ["*", "o", "v", "^", "s", "D"]

# This figure asks one question per PAIR -- does the prior help CONCH, does it
# help QuiltNet -- so the two members of a pair are what must contrast, and the
# colours are chosen by maximising the smaller of the two WITHIN-pair distances
# (154.7 here) subject to every cross-pair distance clearing 50 (66.5 here).
# Optimising the overall minimum instead spreads all four evenly and leaves the
# two curves being compared closer than they need to be: the earlier red/amber
# pairing measured 46 within-pair, which is where "red and amber are still too
# close" came from. Line style still tracks the prior (dashed without, solid
# with) as a second cue, and every row has its own marker.
BACKBONE_STYLE = {
    "T1": {"color": "#d500f9", "marker": "v", "linestyle": "--"},
    "T2": {"color": "#00c853", "marker": "*", "linestyle": "-"},
    "Q1": {"color": "#2979ff", "marker": "^", "linestyle": "--"},
    "Q2": {"color": "#ffab00", "marker": "s", "linestyle": "-"},
}

PANEL_SIZE = (5.4, 5.4)
# Fraction of figure height per legend ROW -- see curves.py, which sizes its
# strip the same way. A fixed strip sized for a two-row legend leaves an empty
# band above a one-row one, which is what pushed this figure's legend far from
# its panels.
LEGEND_ROW_STRIP = 0.065
LEGEND_GAP = 0.05

# Bands are faint and share one zorder below every curve, so a row is never
# hidden by another row's spread. Grid lines sit at 1.5.
BAND_ALPHA = 0.12
HIGHLIGHT_BAND_ALPHA = 0.20
BAND_ZORDER = 1.6

LABEL_FONTSIZE = 15.0
TICK_FONTSIZE = 13.0
TITLE_FONTSIZE = 17.0
LEGEND_FONTSIZE = 15.0


def classify(payload, run_dir):
    """Row id from configuration, plus the name-only axes the payload omits."""
    config = payload.get("sampler_config", {})
    mode = config.get("uncertainty_mode")
    pool_consistency = float(config.get("pool_consistency_weight") or 0)
    backbone = str(payload.get("visual_backbone", ""))
    conch = "CONCH" in backbone
    quilt = "Quilt" in backbone
    stem = os.path.basename(run_dir)
    text = "text-llm_" in stem
    lora = "lora" in stem and "auxcenter" in stem
    augment = "augflip_rotate" in stem

    if quilt:
        assert mode == "disagreement" and pool_consistency == 5, (
            f"unclassified quilt run {stem}: mode={mode} pc={pool_consistency}")
        return "Q2" if text else "Q1"
    if not conch:
        if mode == "coverage":
            return "A0"
        if mode == "visual_margin":
            return "A1"
        if mode == "disagreement" and pool_consistency == 0:
            return "A2"
        if mode == "disagreement" and pool_consistency == 5:
            return "A3"
        raise ValueError(f"unclassified run {stem}: mode={mode} pc={pool_consistency}")
    if not text:
        return "T1"
    if not lora:
        return "T2"
    return "T4" if augment else "T3"


def load(seeds):
    cells = {}
    for dataset in DATASETS:
        for seed in seeds:
            pattern = os.path.join(DATA, dataset, f"seed{seed}", "*", "*_results.pt")
            for path in glob.glob(pattern):
                # A sharded run writes per-shard results beside the merged one,
                # each holding only that shard's budgets. They are partial views
                # of a run already loaded, not runs of their own.
                if "_shard" in os.path.basename(path):
                    continue
                run_dir = os.path.dirname(path)
                payload = torch.load(path, map_location="cpu", weights_only=False)
                assert payload["dataset"] == dataset, (path, payload["dataset"])
                assert payload["seed"] == seed, (path, payload["seed"])
                row = classify(payload, run_dir)
                key = (row, dataset, seed)
                assert key not in cells, (
                    f"two archives map to {key}: {run_dir} and {cells[key][0]}"
                )
                # Budget keys are ints in the older archives and strings in the
                # QuiltNet ones; index by whichever this payload uses rather
                # than assuming, since guessing wrong is a KeyError per row.
                linear = payload["linear"]
                metrics = {int(b): m for b, m in linear.items()}
                cells[key] = (run_dir,
                              [metrics[b]["acc"] * 100.0 for b in BUDGETS])
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["ablation", "framework", "backbones"],
                        default="ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[38, 42, 611])
    parser.add_argument("--highlight-band-only", action="store_true",
                        help="shade only the highlighted row instead of all")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    rows = {"ablation": ABLATION, "framework": FRAMEWORK,
            "backbones": BACKBONES}[arguments.table]
    highlight = HIGHLIGHT[arguments.table]
    cells = load(arguments.seeds)

    missing = [(r, d, s) for r in rows for d in DATASETS for s in arguments.seeds
               if (r, d, s) not in cells]
    assert not missing, f"missing cells: {missing}"

    # Colour by position with the highlight first, so the highlighted row keeps
    # the palette's red -- the same colour "PACT (Ours)" has in the accuracy
    # figure, so a reader carries one association across all three figures.
    if arguments.table == "backbones":
        style = {row: dict(BACKBONE_STYLE[row]) for row in rows}
    else:
        order = [highlight] + [r for r in rows if r != highlight]
        style = {row: {"color": _PALETTE[order.index(row) % len(_PALETTE)],
                       "marker": _MARKERS[order.index(row) % len(_MARKERS)],
                       "linestyle": "-"}
                 for row in rows}

    # A3 is the DINOv2 baseline in the framework figure and the full method in
    # the ablation, so it is named for the role it plays in each.
    labels = dict(ROW_LABELS)
    if arguments.table == "framework":
        labels["A3"] = "DINOv2 backbone (PACT)"
    if arguments.table == "backbones":
        labels["T2"] = "CONCH + language-guided prior"

    figure, axes = plt.subplots(
        nrows=1, ncols=len(DATASETS),
        figsize=(PANEL_SIZE[0] * len(DATASETS), PANEL_SIZE[1]),
    )

    band_all = not arguments.highlight_band_only
    for axis, dataset in zip(axes, DATASETS):
        for row in rows:
            per_seed = [cells[(row, dataset, s)][1] for s in arguments.seeds]
            mean = [statistics.mean(c[i] for c in per_seed)
                    for i in range(len(BUDGETS))]
            is_highlight = row == highlight
            # +-1 std between seeds, shaded on every row by default. Unlike the
            # ten-method accuracy figure, four rows is few enough that the bands
            # stay readable, and here they carry the point: on some datasets the
            # seed spread is comparable to the gaps between rows, which a single
            # highlighted band would hide for the other three.
            if len(arguments.seeds) > 1 and (band_all or is_highlight):
                spread = [statistics.stdev([c[i] for c in per_seed])
                          for i in range(len(BUDGETS))]
                lower = [m - h for m, h in zip(mean, spread)]
                upper = [m + h for m, h in zip(mean, spread)]
                # Every band goes under every curve. Drawing the highlight's
                # band at its own zorder minus one put it at 9, i.e. ON TOP of
                # the other three curves at zorder 2 -- the highlighted row's
                # spread was covering the rows it is meant to be compared with.
                axis.fill_between(
                    BUDGETS, lower, upper,
                    color=style[row]["color"],
                    alpha=HIGHLIGHT_BAND_ALPHA if is_highlight else BAND_ALPHA,
                    linewidth=0,
                    zorder=BAND_ZORDER,
                )
            axis.plot(
                BUDGETS, mean,
                label=labels[row],
                color=style[row]["color"], marker=style[row]["marker"],
                linestyle=style[row]["linestyle"],
                linewidth=1.2, markersize=4.0,
                zorder=10 if is_highlight else 2,
                alpha=1.0 if is_highlight else 0.85,
            )
        axis.set_xticks(BUDGETS)
        axis.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        axis.set_xlabel("Cumulative Budget", fontsize=LABEL_FONTSIZE)
        axis.set_ylabel("Accuracy (%)", fontsize=LABEL_FONTSIZE)
        axis.set_title(DATASET_TITLES[dataset], fontsize=TITLE_FONTSIZE)
        axis.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    legend_rows = -(-len(labels) // len(rows))
    strip = LEGEND_ROW_STRIP * legend_rows + LEGEND_GAP
    plt.tight_layout(rect=(0, 0, 1, 1.0 - strip))
    figure.legend(handles, labels, loc="upper center",
                  bbox_to_anchor=(0.5, 1.0),
                  ncol=len(rows), fontsize=LEGEND_FONTSIZE, frameon=True)

    os.makedirs(ASSETS, exist_ok=True)
    suffix = "_".join(str(s) for s in arguments.seeds)
    out = arguments.out or os.path.join(
        ASSETS, f"{arguments.table}_seeds{suffix}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"[fig] wrote {out}")
    print(f"[fig] rows {rows} over seeds {arguments.seeds}")


if __name__ == "__main__":
    main()
