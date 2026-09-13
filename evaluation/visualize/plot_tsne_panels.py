"""t-SNE of the feature space, with each sampler's selection marked.

The right half of Fig. 3 in the CODAPath paper, standalone: the pool coloured by
TRUE class, with each sampler's selected samples overlaid in their own class
colour, drawn larger and rimmed in black. One panel per sampler, stacked
vertically like the published figure (use --cols for a grid).

Every panel shares ONE embedding, so a difference between panels is a difference
in what was selected and not in how the map was drawn -- two t-SNE runs of the
same data give different pictures, which would make the comparison meaningless.

Pool colours follow the retired `evaluation/plots.py::visualize_tsne` (alpha
0.45, small markers). The selection marker differs from it deliberately -- see
the comment beside `SELECTED_MARKER_SCALE`.

Run:  python3.11 codapath/evaluation/visualize/plot_tsne_panels.py \
          --tsne data_upload/tsne_cache/tsne_histoset_s42.npz --budget 25 \
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
import seaborn as sns

POOL_MARKER_SIZE = 4.0        # one pool point
SELECTED_BASE_SIZE = 24.0     # the selection marker before scaling
SELECTED_MARKER_SCALE = 1.2   # 1.2x that, so the black rim has room to read
SELECTED_EDGE_WIDTH = 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsne", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--selection", action="append", required=True,
                        metavar="LABEL=PATH", help="one panel, in order")
    parser.add_argument("--cols", type=int, default=1)
    parser.add_argument("--dataset", default="histoset")
    parser.add_argument("--panel-size", type=float, default=4.4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    blob = np.load(args.tsne, allow_pickle=True)
    embedding = blob["embedding"]
    labels = blob["labels"]
    class_names = [str(c) for c in blob["class_names"]]
    row_of = {int(r): position for position, r in enumerate(blob["rows"])}

    panels = []
    for item in args.selection:
        if "=" not in item:
            raise SystemExit(f"--selection wants LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["budget"]) != args.budget:
            raise SystemExit(
                f"{path} is budget {payload['budget']}, expected {args.budget}"
            )
        panels.append((label, payload))

    # One split, one row order: otherwise index i is a different patch per run,
    # and the embedding would be showing someone else's dataset.
    fingerprints = {p["train_fingerprint"] for _, p in panels}
    if len(fingerprints) != 1:
        raise SystemExit(f"selections span {len(fingerprints)} different splits")
    if str(blob["train_fingerprint"]) not in fingerprints:
        raise SystemExit("the embedding was built on a different split")

    palette = sns.color_palette("tab20", len(class_names))
    columns = max(1, args.cols)
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(
        rows, columns,
        figsize=(args.panel_size * columns, args.panel_size * rows),
        squeeze=False,
    )

    for position, (label, payload) in enumerate(panels):
        axis = axes[position // columns][position % columns]
        for index, name in enumerate(class_names):
            mask = labels == index
            axis.scatter(embedding[mask, 0], embedding[mask, 1],
                         s=POOL_MARKER_SIZE, color=palette[index],
                         alpha=0.45, linewidths=0, rasterized=True,
                         label=name if position == 0 else None)

        selected = [int(i) for i in payload["selected_indices"]]
        missing = [i for i in selected if i not in row_of]
        if missing:
            raise SystemExit(
                f"{label}: {len(missing)} of {len(selected)} selected rows are "
                "outside the embedding -- recompute it with --keep covering it"
            )
        spots = np.array([row_of[i] for i in selected])
        # Selected points carry their OWN class colour, not a flat black: the
        # question the figure answers is which clusters a method reached, and a
        # monochrome dot makes the reader trace it back to the cluster under it.
        # They are separated from the pool by size and a black rim instead.
        selected_colors = [palette[int(v)] for v in payload["selected_labels"]]
        axis.scatter(embedding[spots, 0], embedding[spots, 1],
                     s=SELECTED_MARKER_SCALE * SELECTED_BASE_SIZE,
                     c=selected_colors, edgecolors="black",
                     linewidths=SELECTED_EDGE_WIDTH, zorder=5)
        axis.set_title(label, fontsize=13)
        axis.set_xticks([])
        axis.set_yticks([])

    for position in range(len(panels), rows * columns):
        axes[position // columns][position % columns].axis("off")

    handles, names = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, names, title="Classes", loc="center left",
                  bbox_to_anchor=(0.995, 0.5), fontsize=9, title_fontsize=10,
                  frameon=False, markerscale=2.8)
    figure.tight_layout()

    os.makedirs(ASSETS, exist_ok=True)
    out = args.out or os.path.join(
        ASSETS,
        f"tsne_panels_{args.dataset}_{args.budget}.png",
    )
    figure.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
