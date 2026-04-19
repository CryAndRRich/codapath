from typing import List, Optional, Tuple, Dict
import os

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE

import torch
from torch import nn
from torch.utils.data import DataLoader

from set_up import clear_memory
from model import extract_image_embeddings

def evaluate_model(model: nn.Module, 
                   test_loader: DataLoader, 
                   test_labels: np.ndarray, 
                   device: torch.device) -> Tuple[float, float, float, float]:
    _, _, probs = extract_image_embeddings(test_loader, model, device=device)
    preds = np.argmax(probs, axis=1)
    
    acc = accuracy_score(test_labels, preds)
    pre = precision_score(test_labels, preds, average="macro", zero_division=0)
    rec = recall_score(test_labels, preds, average="macro", zero_division=0)
    f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
    
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {pre * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"Macro F1 : {f1 * 100:.2f}%")
    
    return acc, pre, rec, f1

def visualize_tsne(embeddings_proj: np.ndarray, 
                   true_labels: np.ndarray, 
                   class_names: List[str], 
                   title: str, 
                   color_map: Dict[str, str], 
                   seed: int, 
                   max_samples: int = 25000,
                   selected_indices: Optional[List[int]] = None) -> None:
    
    num_total = len(embeddings_proj)
    
    if num_total > max_samples:
        np.random.seed(seed)
        
        if selected_indices is not None and len(selected_indices) > 0:
            sel_array = np.array(selected_indices)
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
            
        X_feat = embeddings_proj[indices]
        y_lbl = true_labels[indices]
    else:
        X_feat = embeddings_proj
        y_lbl = true_labels
        local_selected_indices = np.array(selected_indices) if selected_indices is not None else []
        
    tsne = TSNE(
        n_components=2, 
        random_state=seed, 
        perplexity=30, 
        n_jobs=-1
    )
    X_2d = tsne.fit_transform(X_feat)
    
    plt.figure(figsize=(12, 9))
    label_names = [class_names[lbl] for lbl in y_lbl]
    hue_order = sorted(class_names) 
    
    normal_size = 15
    
    sns.scatterplot(
        x=X_2d[:, 0], 
        y=X_2d[:, 1],
        hue=label_names, 
        hue_order=hue_order, 
        palette=color_map,   
        legend="full", 
        alpha=0.4, 
        s=normal_size,
        edgecolor=None
    )
    
    if len(local_selected_indices) > 0:
        plt.scatter(
            X_2d[local_selected_indices, 0],
            X_2d[local_selected_indices, 1],
            color="black",
            edgecolor="white", 
            linewidth=0.5, 
            s=normal_size * 0.8, 
            label="Selected Samples",
            zorder=5 
        )
    
    plt.title(title, fontsize=16, fontweight="bold")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    
    os.makedirs("tsne_plots", exist_ok=True)
    plot_name = "_".join(title.lower().split())
    save_path = f"tsne_plots/{plot_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    del X_feat, y_lbl, X_2d, label_names
    clear_memory()


def plot_acc_diff(budgets_dict: Dict[str, List[int]],
                  acc_data: List[Dict[str, List[float]]]) -> None:
    
    styles = {
        "Random":    {"color": "black",    "marker": "o", "zorder": 5},
        "Margin":    {"color": "orange",   "marker": "v", "zorder": 2},
        "Entropy":   {"color": "red",      "marker": "^", "zorder": 2},
        "Coreset":   {"color": "teal",     "marker": "s", "zorder": 2},
        "BADGE":     {"color": "purple",   "marker": "D", "zorder": 2},
        "TypiClust": {"color": "brown",    "marker": "x", "zorder": 2},
        "ActiveFT":  {"color": "hotpink",  "marker": "p", "zorder": 2},
        "CODAPath":  {"color": "darkblue", "marker": "*", "zorder": 10, "markersize": 9, "linewidth": 2}
    }
    
    methods = list(styles.keys())
    datasets = ["PathMNIST", "HistoSet-5x14", "SkinTissue"]
    datasets_title = ["(a) PathMNIST", "(b) HistoSet-5x14", "(c) SkinTissue"]
    y_label = "Accuracy Difference (%)"
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5.5))
    
    for col in range(3):
        ax = axes[col] 
        dataset_name = datasets[col]
        title = datasets_title[col]
        budget = np.array(budgets_dict[dataset_name])
        
        current_data = acc_data[col]
        random_vals = np.array(current_data["Random"])
        
        ax.axhline(0, color="black", linewidth=1.5, zorder=1)
        
        for method in methods:
            method_vals = np.array(current_data[method])
            diff_vals = method_vals - random_vals
            
            ax.plot(budget, diff_vals, 
                    label=method if col == 0 else "", 
                    color=styles[method]["color"],
                    marker=styles[method]["marker"],
                    markersize=styles[method].get("markersize", 6),
                    linewidth=styles[method].get("linewidth", 1.5),
                    zorder=styles[method]["zorder"])
        
        ax.set_xticks(budget)
        ax.set_xticklabels(budget, rotation=0, ha="center", fontsize=11)
        
        ax.set_xlabel("Cumulative Budget", fontsize=12, labelpad=10)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, pad=10)
        
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(
        loc="upper center", 
        bbox_to_anchor=(0.5, 1.15),
        ncol=8, 
        fontsize=12, 
        frameon=True, 
        shadow=True
    )
    
    plt.tight_layout()
    
    plt.subplots_adjust(top=0.95) 
    
    plt.savefig("al_results_acc.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_label_diversity(dataset: str,
                         methods: List[str],
                         budget: int,
                         class_names: List[str],
                         selected_sample_dir: str,
                         figsize: tuple = (11, 6)) -> None:
    num_classes = len(class_names)
    method_counts = []
    method_entropies = []

    for method in methods:
        filename = f"{dataset}_{method.lower()}_budget_{budget}.pt"
        filepath = os.path.join(selected_sample_dir, filename)
        
        counts = np.zeros(num_classes)
        entropy = 0.0
        
        if os.path.exists(filepath):
            data = torch.load(filepath, map_location="cpu", weights_only=False)
            labels = data["selected_labels"]
            
            unique_lbls, lbl_counts = np.unique(labels, return_counts=True)
            for lbl, cnt in zip(unique_lbls, lbl_counts):
                counts[int(lbl)] = cnt
                
            total = np.sum(counts)
            if total > 0:
                probs = counts[counts > 0] / total
                entropy = -np.sum(probs * np.log2(probs))
        else:
            print(f"Cannot find the file: {filepath}")
            
        method_counts.append(counts)
        method_entropies.append(entropy)

    max_count = max([max(c) for c in method_counts]) if method_counts else 1

    _, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=figsize, sharey=True)
    plt.subplots_adjust(wspace=0)

    y_positions = np.arange(num_classes)

    for i, (ax, method, counts, entropy) in enumerate(zip(axes, methods, method_counts, method_entropies)):
        ax.barh(y_positions, counts, color="#C44E52", align="center")
        
        ax.set_title(method, fontsize=13, pad=7)
        
        ax.set_xlim(0, max_count * 1.05)
        ax.set_xticks([])
        
        if i == 0:
            ax.invert_yaxis()
            ax.set_yticks(y_positions)
            ax.set_yticklabels(class_names, fontsize=11)
            ax.tick_params(axis="y", length=0, pad=10)
            
            ax.text(
                -0.15, 
                -0.025, 
                "Entropy", 
                transform=ax.transAxes,
                ha="right", 
                va="top", 
                fontsize=11
            )
        else:
            ax.tick_params(left=False) 
            
        ax.text(
            0.5, 
            -0.025, 
            f"{entropy:.3f}", 
            transform=ax.transAxes, 
            ha="center", 
            va="top", 
            fontsize=11)

    save_path = f"label_diversity_{dataset}_{budget}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()