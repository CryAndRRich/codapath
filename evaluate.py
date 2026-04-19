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
    
    sns.scatterplot(
        x=X_2d[:, 0], 
        y=X_2d[:, 1],
        hue=label_names, 
        hue_order=hue_order, 
        palette=color_map,   
        legend="full", 
        alpha=0.4, 
        s=20,
        edgecolor=None
    )
    
    if len(local_selected_indices) > 0:
        plt.scatter(
            X_2d[local_selected_indices, 0],
            X_2d[local_selected_indices, 1],
            color="black",
            edgecolor="white", 
            linewidth=0.8,
            s=60, 
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

if __name__ == "__main__":
    budgets = {
        "PathMNIST": [50, 100, 150, 200, 250, 300],
        "HistoSet-5x14": [50, 100, 150, 200, 250, 300],
        "SkinTissue": [50, 100, 150, 200, 250, 300]
    }

    methods = ["Random", "Margin", "Entropy", "Coreset", "BADGE", "TypiClust", "ActiveFT", "CODAPath"]

    acc_pathmnist = {
        "CODAPath": [92.34, 92.7, 93.58, 94.05, 94.4, 94.67],
        "Random": [89.0, 93.59, 94.92, 94.94, 95.13, 95.42],
        "Margin": [88.54, 92.23, 93.84, 94.29, 92.87, 94.64],
        "Entropy": [54.71, 87.44, 84.76, 92.91, 93.15, 92.76],
        "Coreset": [80.61, 89.44, 92.84, 93.38, 94.74, 94.07],
        "BADGE": [88.94, 92.63, 95.11, 95.32, 95.06, 95.17],
        "TypiClust": [89.29, 89.21, 93.57, 94.44, 94.23, 94.89],
        "ActiveFT": [89.37, 91.75, 94.11, 95.58, 95.71, 95.28]
    }

    acc_histoset = {
        "CODAPath": [79.25, 82.13, 85.52, 86.93, 89.2, 90.29],
        "Random": [63.93, 77.75, 82.36, 86.84, 87.98, 88.64],
        "Margin": [67.23, 76.5, 86.02, 88.66, 90.36, 90.29],
        "Entropy": [49.05, 58.27, 69.55, 75.0, 73.54, 80.57],
        "Coreset": [56.68, 72.25, 77.46, 80.38, 81.86, 83.88],
        "BADGE": [65.59, 78.86, 84.36, 86.52, 87.61, 87.68],
        "TypiClust": [69.41, 79.41, 85.71, 85.34, 86.16, 88.93],
        "ActiveFT": [67.73, 76.75, 82.61, 86.23, 87.64, 88.2]
    }

    acc_skin = {
        "CODAPath": [80.0, 80.84, 84.65, 86.36, 87.65, 87.77],
        "Random": [73.01, 77.68, 83.2, 84.68, 86.11, 87.17],
        "Margin": [70.53, 77.94, 84.42, 85.67, 87.37, 88.08],
        "Entropy": [46.52, 58.63, 71.98, 80.51, 82.0, 86.16],
        "Coreset": [63.35, 71.8, 76.26, 78.15, 77.48, 78.29],
        "BADGE": [68.67, 75.04, 83.85, 86.34, 86.71, 87.39],
        "TypiClust": [73.73, 79.47, 83.71, 85.14, 86.5, 87.6],
        "ActiveFT": [72.32, 79.35, 84.07, 85.02, 86.38, 87.36]
    }

    acc_data_list = [acc_pathmnist, acc_histoset, acc_skin]

    plot_acc_diff(budgets, acc_data_list)