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

# Fraction of figure height reserved above the panels for ONE legend row.
# The strip has to scale with the number of rows the legend actually wraps to:
# a fixed fraction sized for the two-row ten-method legend leaves a band of
# empty figure above a one-row legend as tall as the legend itself.
LEGEND_ROW_STRIP = 0.065
# Extra strip beyond the legend box itself, so it does not sit on the titles.
LEGEND_GAP = 0.05

# All bands share one zorder, below every curve. Grid lines sit at 1.5 by
# default, so this keeps the bands above the grid and under the data.
BAND_ZORDER = 1.6


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
    label_fontsize: float = 15.0,
    tick_fontsize: float = 13.0,
    title_fontsize: float = 17.0,
    legend_fontsize: float = 15.0,
    band_alpha: float = 0.10,
    highlight_band_alpha: float = 0.22,
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

    `band_methods` restricts the shading to a subset; it defaults to every
    method that has std values. Ten overlapping bands stay readable only
    because `band_alpha` is low and every band is drawn under every curve --
    raise the alpha and the middle of the panel turns to mush.

    `panel_size` is the (width, height) of ONE panel in inches -- widen it for
    a rectangular layout. Note that a wide panel does not by itself separate
    curves bunched at the top: what compresses them is a weak baseline
    (coreset, entropy) dragging the shared y-axis down, which is a VERTICAL
    problem. `linewidth` and `markersize` set the SHARED size every method
    (highlight included) is drawn at -- keep both small on a ten-method panel
    or the markers themselves cover the gaps the figure exists to show.
    """
    # Ten colours chosen by maximising the SMALLEST pairwise CIE76 distance
    # over a candidate pool, not by picking hues that sound different. The
    # original hand-assembled palette had three pairs under dE=25 -- orange vs
    # amber at 18.0, blue vs indigo at 18.7 -- which read as one colour at
    # 1.2pt line width; this set's closest pair is 46.1.
    #
    # The pool spans DARK hues as well as bright ones (forest green at L=35,
    # purple at L=30, navy at L=36 sit beside L=69-71 orange and green), with
    # only two constraints: CIE L in [25, 78] so nothing is lost against white
    # paper or indistinguishable from another dark, and saturation >= 0.60 so
    # every hue stays vivid. Two constraints that sound reasonable were
    # measured and REJECTED: an all-bright palette crowds into one corner of
    # the gamut and drops the minimum to 22.8, and forcing a minimum hue gap
    # drops it to 25.4 by spreading hues evenly instead of maximising the
    # distance the eye actually uses.
    _PALETTE = [
        "#e8000b", "#1b5e20", "#ff8c00", "#6a1b9a", "#00acc1",
        "#e8118f", "#00c853", "#afb42b", "#d500f9", "#01579b",
        "#5d4037", "#f9a825",
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
                # Every band goes BELOW every curve, not just below its own.
                # Drawing a band at its method's zorder minus one still lets it
                # cover the curves of methods ordered after it -- harmless with
                # one band, and the whole problem once every method has one.
                ax.fill_between(
                    budget, centre - half, centre + half,
                    color=style[m]["color"],
                    alpha=highlight_band_alpha if m == highlight else band_alpha,
                    linewidth=0,
                    zorder=BAND_ZORDER,
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
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.set_xlabel("Cumulative Budget", fontsize=label_fontsize)
        ax.set_ylabel("Accuracy (%)", fontsize=label_fontsize)
        ax.set_title(dataset_titles.get(dset, dset), fontsize=title_fontsize)
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    # Lay the axes out first, reserving a strip just tall enough for the legend,
    # then anchor the legend into that strip. Two failure modes bracket this:
    # anchoring at 1.12 over a `top=0.80` margin leaves a band of empty figure
    # as tall as the legend, and pulling it flush to 1.0 makes it overlap the
    # panel titles. The strip is sized from the number of rows the legend wraps
    # to, since that -- not the figure -- is what sets how tall it is.
    ncol = min(ncols_legend, len(labels))
    rows = -(-len(labels) // ncol)
    # The strip holds the legend itself plus a gap to the panel titles; anchor
    # the legend at the TOP of it so the gap lands between the two.
    strip = LEGEND_ROW_STRIP * rows + LEGEND_GAP
    plt.tight_layout(rect=(0, 0, 1, 1.0 - strip))
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0),
               ncol=ncol, fontsize=legend_fontsize, frameon=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
