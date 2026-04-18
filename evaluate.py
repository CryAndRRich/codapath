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


def plot_diff(budgets_dict: Dict[str, List[int]],
              acc_data: List[Dict[str, List[float]]],
              f1_data: List[Dict[str, List[float]]]) -> None:
    
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
    y_labels = ["Accuracy Difference (%)", "Macro F1 Difference (%)"]
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 9))
    
    all_data = [acc_data, f1_data]
    
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            dataset_name = datasets[col]
            title = datasets_title[col]
            budget = np.array(budgets_dict[dataset_name])
            
            current_data = all_data[row][col]
            random_vals = np.array(current_data["Random"])
            
            ax.axhline(0, color="black", linewidth=1.5, zorder=1)
            
            for method in methods:
                method_vals = np.array(current_data[method])
                diff_vals = method_vals - random_vals
                
                ax.plot(budget, diff_vals, 
                        label=method if row==0 and col==0 else "", 
                        color=styles[method]["color"],
                        marker=styles[method]["marker"],
                        markersize=styles[method].get("markersize", 6),
                        linewidth=styles[method].get("linewidth", 1.5),
                        zorder=styles[method]["zorder"])
            
            ax.set_xticks(budget)
            
            ax.set_xticklabels(budget, rotation=0, ha="center", fontsize=11)
            ax.set_xlabel("Cumulative Budget", fontsize=12, labelpad=10)
            ax.set_ylabel(y_labels[row], fontsize=12)
            
            if row == 0:
                ax.set_title(title, fontsize=14, pad=10)
            
            ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(
        loc="upper center", 
        bbox_to_anchor=(0.5, 1.05),
        ncol=8, 
        fontsize=12, 
        frameon=True, 
        shadow=True
    )
    
    plt.tight_layout(pad=1.5)
    
    plt.subplots_adjust(hspace=0.25, top=0.92)
    
    plt.savefig("al_results.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    budgets = {
        "PathMNIST": [50, 100, 150, 200, 250, 300],
        "HistoSet-5x14": [50, 100, 150, 200, 250, 300],
        "SkinTissue": [50, 100, 150, 200, 250, 300]
    }

    methods = ["Random", "Margin", "Entropy", "Coreset", "BADGE", "TypiClust", "ActiveFT", "CODAPath"]

    acc_pathmnist = {
        "CODAPath": [91.84, 92.7, 93.58, 94.05, 94.4, 94.67],
        "Random": [89.0, 93.59, 94.93, 94.94, 95.14, 95.46],
        "Margin": [88.54, 93.34, 94.57, 95.1, 92.87, 94.64],
        "Entropy": [54.71, 90.14, 84.76, 92.91, 95.33, 92.76],
        "Coreset": [80.65, 89.44, 92.84, 93.4, 94.74, 94.07],
        "BADGE": [88.96, 92.95, 95.11, 95.32, 95.25, 95.17],
        "TypiClust": [89.29, 89.3, 93.57, 94.44, 94.23, 94.89],
        "ActiveFT": [89.4, 91.75, 94.12, 95.58, 95.71, 95.28]
    }

    acc_histoset = {
        "CODAPath": [78.27, 81.84, 85.52, 86.91, 89.20, 90.29],
        "Random": [63.95, 77.75, 82.36, 86.86, 87.98, 88.66],
        "Margin": [71.27, 79.00, 86.02, 89.29, 90.36, 91.04],
        "Entropy": [49.05, 58.27, 69.55, 75.00, 78.14, 80.57],
        "Coreset": [56.68, 72.29, 77.46, 80.38, 81.86, 83.93],
        "BADGE": [65.71, 78.86, 84.36, 86.52, 88.41, 87.68],
        "TypiClust": [69.91, 79.61, 85.71, 85.36, 86.18, 88.93],
        "ActiveFT": [67.73, 76.88, 82.61, 86.32, 87.64, 88.21]
    }

    acc_skin = {
        "CODAPath": [79.99, 80.78, 84.64, 86.33, 87.65, 87.77],
        "Random": [73.04, 77.68, 83.20, 84.74, 86.12, 87.17],
        "Margin": [70.53, 77.94, 84.59, 86.08, 87.37, 88.08],
        "Entropy": [50.90, 58.63, 71.98, 81.44, 82.00, 86.16],
        "Coreset": [63.35, 71.80, 76.31, 78.18, 77.52, 78.32],
        "BADGE": [68.67, 75.04, 83.85, 86.34, 86.87, 87.40],
        "TypiClust": [73.74, 79.47, 83.71, 85.14, 86.50, 87.69],
        "ActiveFT": [72.32, 79.47, 84.07, 85.02, 86.38, 87.40]
    }

    f1_pathmnist = {
        "CODAPath": [89.36, 90.19, 89.96, 90.65, 91.43, 91.85],
        "Random": [85.34, 91.49, 93.11, 92.84, 93.10, 93.50],
        "Margin": [85.29, 91.10, 92.38, 92.68, 89.63, 92.35],
        "Entropy": [47.48, 86.89, 83.78, 90.83, 94.04, 90.73],
        "Coreset": [75.37, 85.85, 90.48, 91.36, 92.36, 90.82],
        "BADGE": [85.80, 90.77, 92.61, 92.84, 92.75, 92.67],
        "TypiClust": [86.51, 87.07, 90.51, 91.73, 90.63, 92.39],
        "ActiveFT": [86.05, 89.53, 92.02, 93.76, 93.94, 93.23]
    }

    f1_histoset = {
        "CODAPath": [77.79, 81.29, 85.12, 86.46, 88.92, 90.05],
        "Random": [58.92, 77.12, 81.73, 86.64, 87.79, 88.44],
        "Margin": [70.12, 78.50, 85.73, 89.05, 90.25, 90.92],
        "Entropy": [41.06, 53.63, 65.14, 71.42, 76.94, 80.03],
        "Coreset": [51.51, 71.13, 77.17, 79.90, 81.43, 83.73],
        "BADGE": [63.93, 78.44, 83.92, 86.20, 88.15, 87.49],
        "TypiClust": [63.47, 78.92, 85.31, 84.60, 85.47, 88.53],
        "ActiveFT": [63.29, 75.62, 81.99, 86.09, 87.50, 88.02]
    }

    f1_skin = {
        "CODAPath": [61.38, 63.66, 69.63, 73.89, 76.95, 83.37],
        "Random": [53.50, 57.84, 70.60, 73.19, 79.15, 80.70],
        "Margin": [51.84, 67.01, 73.56, 80.15, 75.10, 78.68],
        "Entropy": [35.24, 50.17, 58.87, 74.58, 74.08, 80.94],
        "Coreset": [51.73, 63.76, 65.16, 67.49, 64.94, 69.40],
        "BADGE": [50.92, 60.86, 70.89, 76.62, 76.03, 75.15],
        "TypiClust": [55.96, 64.61, 68.06, 71.25, 75.20, 77.38],
        "ActiveFT": [52.17, 62.70, 72.71, 73.70, 79.82, 80.36]
    }

    acc_list = [acc_pathmnist, acc_histoset, acc_skin]
    f1_list = [f1_pathmnist, f1_histoset, f1_skin]

    plot_diff(budgets, acc_list, f1_list)