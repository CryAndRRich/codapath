"""Ablation (A0-A3) and training-framework (T1-T4) figures.

Two figures, one script, because both read the same 72-cell grid and differ only
in which rows they draw:

    plot_ablation.py --table ablation   -> ablation_seeds38_42_611.png
    plot_ablation.py --table framework  -> framework_seeds38_42_611.png

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
}

# The highlighted row of each figure keeps the palette's red, matching
# "PACT (Ours)" in the accuracy figure.
HIGHLIGHT = {"ablation": "A3", "framework": "T2"}

_PALETTE = ["#d62728", "#7f7f7f", "#ff7f0e", "#2ca02c", "#9467bd", "#1f77b4"]
_MARKERS = ["*", "o", "v", "^", "s", "D"]

PANEL_SIZE = (5.4, 5.4)
LEGEND_STRIP = 0.13


def classify(payload, run_dir):
    """Row id from configuration, plus the name-only axes the payload omits."""
    config = payload.get("sampler_config", {})
    mode = config.get("uncertainty_mode")
    pool_consistency = float(config.get("pool_consistency_weight") or 0)
    conch = "CONCH" in str(payload.get("visual_backbone", ""))
    stem = os.path.basename(run_dir)
    text = "text-llm_" in stem
    lora = "lora" in stem and "auxcenter" in stem
    augment = "augflip_rotate" in stem

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
                run_dir = os.path.dirname(path)
                payload = torch.load(path, map_location="cpu", weights_only=False)
                assert payload["dataset"] == dataset, (path, payload["dataset"])
                assert payload["seed"] == seed, (path, payload["seed"])
                row = classify(payload, run_dir)
                key = (row, dataset, seed)
                assert key not in cells, (
                    f"two archives map to {key}: {run_dir} and {cells[key][0]}"
                )
                cells[key] = (run_dir,
                              [payload["linear"][b]["acc"] * 100.0 for b in BUDGETS])
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["ablation", "framework"],
                        default="ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[38, 42, 611])
    parser.add_argument("--highlight-band-only", action="store_true",
                        help="shade only the highlighted row instead of all")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    rows = ABLATION if arguments.table == "ablation" else FRAMEWORK
    highlight = HIGHLIGHT[arguments.table]
    cells = load(arguments.seeds)

    missing = [(r, d, s) for r in rows for d in DATASETS for s in arguments.seeds
               if (r, d, s) not in cells]
    assert not missing, f"missing cells: {missing}"

    # Colour by position with the highlight first, so the highlighted row keeps
    # the palette's red -- the same colour "PACT (Ours)" has in the accuracy
    # figure, so a reader carries one association across all three figures.
    order = [highlight] + [r for r in rows if r != highlight]
    style = {row: {"color": _PALETTE[order.index(row) % len(_PALETTE)],
                   "marker": _MARKERS[order.index(row) % len(_MARKERS)]}
             for row in rows}

    # A3 is the DINOv2 baseline in the framework figure and the full method in
    # the ablation, so it is named for the role it plays in each.
    labels = dict(ROW_LABELS)
    if arguments.table == "framework":
        labels["A3"] = "DINOv2 backbone (PACT)"

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
                axis.fill_between(
                    BUDGETS, lower, upper,
                    color=style[row]["color"],
                    alpha=0.30 if is_highlight else 0.15,
                    linewidth=0,
                    zorder=(10 if is_highlight else 2) - 1,
                )
            axis.plot(
                BUDGETS, mean,
                label=labels[row],
                color=style[row]["color"], marker=style[row]["marker"],
                linewidth=1.2, markersize=4.0,
                zorder=10 if is_highlight else 2,
                alpha=1.0 if is_highlight else 0.85,
            )
        axis.set_xticks(BUDGETS)
        axis.set_xlabel("Cumulative Budget", fontsize=12)
        axis.set_ylabel("Accuracy (%)", fontsize=12)
        axis.set_title(DATASET_TITLES[dataset], fontsize=13)
        axis.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    plt.tight_layout(rect=(0, 0, 1, 1.0 - LEGEND_STRIP))
    figure.legend(handles, labels, loc="upper center",
                  bbox_to_anchor=(0.5, 1.0 - LEGEND_STRIP * 0.08),
                  ncol=len(rows), fontsize=10, frameon=True)

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
