"""The accuracy-vs-budget curve renderer, shared by the script and the notebook.

Split out of `plot_accuracy.py` so `evaluate_al_sampler.ipynb` can draw the same
figure without importing a script that would run an argparse main(). Everything
else in `visualize/` is a standalone script; this one file is an importable
module on purpose.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = ["plot_accuracy_curves"]

# Fraction of figure height reserved above the panels for the legend.
LEGEND_STRIP = 0.13


def plot_accuracy_curves(
    budgets_dict: Dict[str, List[int]],
    acc_data: List[Dict[str, List[float]]],
    methods: Optional[List[str]] = None,
    highlight: str = "Ours",
    dataset_titles: Optional[Dict[str, str]] = None,
    ncols_legend: int = 5,
    save_path: str = "accuracy_vs_budget.png",
    std_data: Optional[List[Dict[str, List[float]]]] = None,
    band_methods: Optional[List[str]] = None,
    panel_size: Tuple[float, float] = (5.0, 5.2),
    linewidth: float = 1.2,
    markersize: float = 4.0,
    highlight_scale: float = 1.2,
) -> None:
    """Raw accuracy vs cumulative budget, one square-ish panel per dataset.

    Plots accuracy values directly so a method's rank is read off its vertical position, the
    way AL papers usually show it. `highlight` is drawn on top (zorder), at
    full opacity, and at `highlight_scale` (default 1.2) times the shared line
    width and marker size, so it is findable inside a ten-method cluster
    without being heavy enough to cover the gaps the figure exists to show.
    Pass 1.0 to draw it exactly like every baseline.

    `std_data` mirrors `acc_data` -- same list-of-dicts shape, same keys -- and
    holds the between-seed standard deviation of each point. When given, each
    curve gets a translucent +-1 std band. Omit it and the function behaves
    exactly as before, so single-seed callers need no change.

    `band_methods` restricts the shading to a subset (typically just the
    highlighted method): with ten baselines on one panel, ten overlapping bands
    hide the curves they are meant to annotate. Defaults to every method that
    has std values.

    `panel_size` is the (width, height) of ONE panel in inches -- widen it for
    a rectangular layout. Note that a wide panel does not by itself separate
    curves bunched at the top: what compresses them is a weak baseline
    (coreset, entropy) dragging the shared y-axis down, which is a VERTICAL
    problem. `linewidth` and `markersize` set the SHARED size every method
    (highlight included) is drawn at -- keep both small on a ten-method panel
    or the markers themselves cover the gaps the figure exists to show.
    """
    _PALETTE = [
        "#d62728", "#7f7f7f", "#ff7f0e", "#2ca02c", "#17becf",
        "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#1f77b4",
        "#aec7e8", "#ffbb78",
    ]
    _MARKERS = ["*", "o", "v", "^", "s", "D", "P", "X", "h", "<", "p", ">"]

    datasets = list(budgets_dict.keys())
    if dataset_titles is None:
        dataset_titles = {d: d for d in datasets}

    if methods is None:
        methods = list(acc_data[0].keys())
        if highlight in methods:
            methods = [highlight] + [m for m in methods if m != highlight]

    style = {
        m: {"color": _PALETTE[i % len(_PALETTE)], "marker": _MARKERS[i % len(_MARKERS)]}
        for i, m in enumerate(methods)
    }
    if highlight in style:
        style[highlight]["linewidth"] = linewidth * highlight_scale
        style[highlight]["markersize"] = markersize * highlight_scale
        style[highlight]["zorder"] = 10

    fig, axes = plt.subplots(
        nrows=1, ncols=len(datasets),
        figsize=(panel_size[0] * len(datasets), panel_size[1]),
    )
    if len(datasets) == 1:
        axes = [axes]

    for ax, dset in zip(axes, datasets):
        budget = np.array(budgets_dict[dset])
        current = acc_data[datasets.index(dset)] if isinstance(acc_data, list) else acc_data[dset]

        current_std = None
        if std_data is not None:
            current_std = (std_data[datasets.index(dset)]
                           if isinstance(std_data, list) else std_data.get(dset))

        for m in methods:
            vals = current.get(m)
            if vals is None:
                continue
            spread = None if current_std is None else current_std.get(m)
            if spread is not None and (band_methods is None or m in band_methods):
                centre = np.asarray(vals, dtype=float)
                half = np.asarray(spread, dtype=float)
                ax.fill_between(
                    budget, centre - half, centre + half,
                    color=style[m]["color"],
                    alpha=0.30 if m == highlight else 0.15,
                    linewidth=0,
                    zorder=style[m].get("zorder", 2) - 1,
                )
            ax.plot(
                budget, vals,
                label=m,
                color=style[m]["color"],
                marker=style[m]["marker"],
                linewidth=style[m].get("linewidth", linewidth),
                markersize=style[m].get("markersize", markersize),
                zorder=style[m].get("zorder", 2),
                alpha=1.0 if m == highlight else 0.85,
            )

        ax.set_xticks(budget)
        ax.set_xlabel("Cumulative Budget", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(dataset_titles.get(dset, dset), fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    # Lay the axes out first, reserving a strip just tall enough for the legend,
    # then anchor the legend into that strip. Two failure modes bracket this:
    # anchoring at 1.12 over a `top=0.80` margin leaves a band of empty figure
    # as tall as the legend, and pulling it flush to 1.0 makes it overlap the
    # panel titles. `LEGEND_STRIP` is the fraction of figure height it gets.
    plt.tight_layout(rect=(0, 0, 1, 1.0 - LEGEND_STRIP))
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0 - LEGEND_STRIP * 0.08),
               ncol=min(ncols_legend, len(methods)), fontsize=11, frameon=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
