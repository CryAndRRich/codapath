"""Per-class distribution of what each sampler SELECTED, with its entropy.

The left half of Fig. 3 in the CODAPath paper, standalone: one bar panel per
sampler over a shared x scale, class names down the side, and the Shannon
entropy of that selection printed underneath.

Entropy is **log2 and unnormalized** (bits), matching the published figure and
the retired `evaluation/plots.py::plot_label_diversity`. The ceiling is
log2(C) -- 3.807 for HistoSet's 14 classes -- so a number is only meaningful
against that, and only comparable across datasets after dividing by it.

Reading it: the bar SHAPE is the point. A column with a bar in every row spread
the budget over the whole label space; a column with tall bars and blank rows
spent it on a few classes and never saw the rest. Entropy compresses that to one
number but cannot distinguish "few classes, evenly" from "all classes, unevenly"
-- so the count of covered classes is printed with it.

Run:  python3.11 codapath/evaluation/visualize/plot_selection_entropy.py \
          --budget 25 --dataset histoset \
          --selection "Random=<...>.pt" --selection "PACT=<...>.pt"
"""
import argparse
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODAPATH = os.path.dirname(os.path.dirname(HERE))
ASSETS = os.path.join(CODAPATH, "assets", "img")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAR_COLOR = "#C44E52"


def shannon_bits(counts):
    """Entropy in BITS, unnormalized -- the quantity the paper prints."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--selection", action="append", required=True,
                        metavar="LABEL=PATH", help="one bar column, in order")
    parser.add_argument("--dataset", default="histoset")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sort", action="store_true",
                        help="order columns by entropy instead of as given")
    args = parser.parse_args()

    columns = []
    for item in args.selection:
        if "=" not in item:
            raise SystemExit(f"--selection wants LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["budget"]) != args.budget:
            raise SystemExit(
                f"{path} is budget {payload['budget']}, expected {args.budget}"
            )
        counts = np.bincount(
            np.asarray(payload["selected_labels"], dtype=int),
            minlength=payload["num_classes"],
        ).astype(float)
        columns.append((label, payload, counts))

    # One split, one class order: a run from a different split has different
    # label indices, so its bars would be a different dataset's rows.
    fingerprints = {p["train_fingerprint"] for _, p, _ in columns}
    if len(fingerprints) != 1:
        raise SystemExit(f"selections span {len(fingerprints)} different splits")
    class_names = [str(c) for c in columns[0][1]["class_names"]]
    for label, payload, _ in columns:
        if [str(c) for c in payload["class_names"]] != class_names:
            raise SystemExit(f"{label} has a different class order")
    num_classes = len(class_names)

    if args.sort:
        columns.sort(key=lambda item: -shannon_bits(item[2]))

    largest = max(counts.max() for _, _, counts in columns)
    positions = np.arange(num_classes)

    figure, axes = plt.subplots(
        1, len(columns), figsize=(1.28 * len(columns) + 2.6, 6.0), sharey=True
    )
    if len(columns) == 1:
        axes = [axes]
    plt.subplots_adjust(wspace=0)

    for i, (axis, (label, _, counts)) in enumerate(zip(axes, columns)):
        axis.barh(positions, counts, color=BAR_COLOR, align="center")
        axis.set_title(label, fontsize=12, pad=7)
        axis.set_xlim(0, largest * 1.05)
        axis.set_xticks([])
        if i == 0:
            axis.invert_yaxis()
            axis.set_yticks(positions)
            axis.set_yticklabels(class_names, fontsize=10)
            axis.tick_params(axis="y", length=0, pad=8)
            axis.text(-0.12, -0.028, "Entropy", transform=axis.transAxes,
                      ha="right", va="top", fontsize=11)
            axis.text(-0.12, -0.072, "Classes", transform=axis.transAxes,
                      ha="right", va="top", fontsize=11)
        else:
            axis.tick_params(left=False)
        axis.text(0.5, -0.028, f"{shannon_bits(counts):.3f}",
                  transform=axis.transAxes, ha="center", va="top", fontsize=11)
        axis.text(0.5, -0.072, f"{int((counts > 0).sum())}/{num_classes}",
                  transform=axis.transAxes, ha="center", va="top", fontsize=11)

    os.makedirs(ASSETS, exist_ok=True)
    out = args.out or os.path.join(
        ASSETS,
        f"selection_entropy_{args.dataset}_{args.budget}.png",
    )
    figure.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[fig] wrote {out}   (max possible {np.log2(num_classes):.3f} bits)")
    for label, _, counts in columns:
        print(f"[fig] {label:<22} {shannon_bits(counts):.3f} bits   "
              f"{int((counts > 0).sum())}/{num_classes} classes")


if __name__ == "__main__":
    main()
