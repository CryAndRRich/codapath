"""Diagnostic figures. Import only when a notebook actually plots something —
these pull in matplotlib/seaborn/sklearn.manifold, which the sampling and
training paths do not need.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from utils.runtime import clear_memory

__all__ = ["visualize_tsne", "plot_acc_diff", "plot_label_diversity"]


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
        if m.lower() in {"codapath", "scalpel"}:
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

