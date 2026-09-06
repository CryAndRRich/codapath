"""Diagnostic figures. Import only when a notebook actually plots something —
these pull in matplotlib/seaborn/sklearn.manifold, which the sampling and
training paths do not need.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from utils.runtime import clear_memory

__all__ = ["visualize_tsne", "plot_acc_diff", "plot_accuracy_curves", "plot_label_diversity"]


def visualize_tsne(
    features: np.ndarray,
    true_labels: np.ndarray,
    class_names: List[str],
    title: str,
    color_map: Dict[str, str],
    seed: int,
    max_samples: int = 25000,
    selected_indices: Optional[List[int]] = None,
) -> None:
    num_total = len(features)

    if num_total > max_samples:
        np.random.seed(seed)
        if selected_indices is not None and len(selected_indices) > 0:
            sel_array  = np.array(selected_indices)
            unsel_array = np.setdiff1d(np.arange(num_total), sel_array)
            num_to_sample = max(0, max_samples - len(sel_array))
            if num_to_sample > 0:
                sampled_unsel = np.random.choice(unsel_array, num_to_sample, replace=False)
                indices = np.concatenate([sel_array, sampled_unsel])
            else:
                indices = sel_array[:max_samples]
            local_selected_indices = np.arange(len(sel_array[:max_samples]))
        else:
            indices = np.random.choice(num_total, max_samples, replace=False)
            local_selected_indices = []

        X_feat = features[indices]
        y_lbl  = true_labels[indices]
    else:
        X_feat = features
        y_lbl  = true_labels
        local_selected_indices = np.array(selected_indices) if selected_indices is not None else []

    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, n_jobs=-1)
    X_2d = tsne.fit_transform(X_feat)

    plt.figure(figsize=(12, 9))
    label_names = [class_names[lbl] for lbl in y_lbl]
    sns.scatterplot(
        x=X_2d[:, 0], y=X_2d[:, 1],
        hue=label_names, hue_order=sorted(class_names),
        palette=color_map, legend="full",
        alpha=0.4, s=15, edgecolor=None,
    )
    if len(local_selected_indices) > 0:
        plt.scatter(
            X_2d[local_selected_indices, 0],
            X_2d[local_selected_indices, 1],
            color="black", edgecolor="white", linewidth=0.5,
            s=12, label="Selected Samples", zorder=5,
        )

    plt.title(title, fontsize=16, fontweight="bold")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout()

    os.makedirs("tsne_plots", exist_ok=True)
    plot_name = "_".join(title.lower().split())
    plt.savefig(f"tsne_plots/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    del X_feat, y_lbl, X_2d, label_names
    clear_memory()


def plot_acc_diff(
    budgets_dict: Dict[str, List[int]],
    acc_data: List[Dict[str, List[float]]],
    methods: Optional[List[str]] = None,
    mode: str = "linear",
) -> None:
    _PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    ]
    _MARKERS = ["o", "v", "^", "s", "D", "x", "p", "h", "*", "P", "X", ">"]

    datasets       = list(budgets_dict.keys())
    datasets_title = [f"({chr(97+i)}) {d}" for i, d in enumerate(datasets)]

    if methods is None:
        methods = list(acc_data[0].keys())

    style = {
        m: {"color": _PALETTE[i % len(_PALETTE)], "marker": _MARKERS[i % len(_MARKERS)]}
        for i, m in enumerate(methods)
    }
    for m in methods:
        if m.lower() == "pact":
            style[m]["linewidth"] = 2.5
            style[m]["markersize"] = 9
            style[m]["zorder"] = 10

    fig, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(6 * len(datasets), 5.5))
    if len(datasets) == 1:
        axes = [axes]

    for col, (ax, dset, dtitle) in enumerate(zip(axes, datasets, datasets_title)):
        budget      = np.array(budgets_dict[dset])
        current     = acc_data[col]
        random_vals = np.array(current.get("random", current.get("Random", [0.0] * len(budget))))

        ax.axhline(0, color="black", linewidth=1.5, zorder=1)
        for m in methods:
            vals = np.array(current.get(m, [np.nan] * len(budget)))
            diff = vals - random_vals
            ax.plot(
                budget, diff,
                label=m if col == 0 else "",
                color=style[m]["color"],
                marker=style[m]["marker"],
                markersize=style[m].get("markersize", 6),
                linewidth=style[m].get("linewidth", 1.5),
                zorder=style[m].get("zorder", 2),
            )

        ax.set_xticks(budget)
        ax.set_xticklabels(budget, rotation=0, ha="center", fontsize=11)
        ax.set_xlabel("Cumulative Budget", fontsize=12, labelpad=10)
        ax.set_ylabel("Accuracy Difference (%)", fontsize=12)
        ax.set_title(dtitle, fontsize=14, pad=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
               ncol=min(8, len(methods)), fontsize=12, frameon=True)
    plt.suptitle(f"Training mode: {mode}", fontsize=11, style="italic", y=1.0)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    save_path = f"al_results_acc_{mode}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


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
    linewidth: float = 1.4,
    markersize: float = 6.0,
    highlight_scale: float = 1.5,
) -> None:
    """Raw accuracy vs cumulative budget, one square-ish panel per dataset.

    Unlike plot_acc_diff (which plots the gap to random), this plots accuracy
    values directly so a method's rank is read off its vertical position, the
    way AL papers usually show it. `highlight` is drawn thicker/on top so the
    method under study stands out against the baseline cluster.

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
    problem. `linewidth`, `markersize` and `highlight_scale` (how much thicker
    the highlighted method is drawn) are exposed for the same reason -- a heavy
    line covers the very gaps the figure exists to show.
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
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               ncol=min(ncols_legend, len(methods)), fontsize=11, frameon=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


def plot_label_diversity(
    dataset: str,
    methods: List[str],
    budget: int,
    class_names: List[str],
    selected_sample_dir: str,
    figsize: Tuple[int, int] = (11, 6),
) -> None:
    num_classes = len(class_names)
    method_counts: List[np.ndarray] = []
    method_entropies: List[float] = []

    for method in methods:
        filename = f"{dataset}_{method.lower()}_budget_{budget}.pt"
        filepath = os.path.join(selected_sample_dir, filename)
        counts   = np.zeros(num_classes)
        entropy  = 0.0

        if os.path.exists(filepath):
            data  = torch.load(filepath, map_location="cpu", weights_only=False)
            lbls  = data["selected_labels"]
            unique_lbls, lbl_counts = np.unique(lbls, return_counts=True)
            for lbl, cnt in zip(unique_lbls, lbl_counts):
                counts[int(lbl)] = cnt
            total = np.sum(counts)
            if total > 0:
                probs   = counts[counts > 0] / total
                entropy = float(-np.sum(probs * np.log2(probs)))
        else:
            print(f"File not found: {filepath}")

        method_counts.append(counts)
        method_entropies.append(entropy)

    max_count = max((max(c) for c in method_counts), default=1)
    _, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=figsize, sharey=True)
    plt.subplots_adjust(wspace=0)
    y_positions = np.arange(num_classes)

    for i, (ax, method, counts, entropy) in enumerate(
        zip(axes, methods, method_counts, method_entropies)
    ):
        ax.barh(y_positions, counts, color="#C44E52", align="center")
        ax.set_title(method, fontsize=13, pad=7)
        ax.set_xlim(0, max_count * 1.05)
        ax.set_xticks([])

        if i == 0:
            ax.invert_yaxis()
            ax.set_yticks(y_positions)
            ax.set_yticklabels(class_names, fontsize=11)
            ax.tick_params(axis="y", length=0, pad=10)
            ax.text(-0.15, -0.025, "Entropy", transform=ax.transAxes,
                    ha="right", va="top", fontsize=11)
        else:
            ax.tick_params(left=False)

        ax.text(0.5, -0.025, f"{entropy:.3f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=11)

    save_path = f"label_diversity_{dataset}_{budget}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
