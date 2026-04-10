from typing import List, Tuple, Dict
import os

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE

import torch
from torch import nn
from torch.utils.data import DataLoader

from .set_up import clear_memory
from .model import extract_image_embeddings

def evaluate_model(model: nn.Module, 
                   test_loader: DataLoader, 
                   true_labels: np.ndarray, 
                   device: torch.device) -> Tuple[float, float, float, float]:
    _, probs, _ = extract_image_embeddings(test_loader, model, device=device)
    preds = np.argmax(probs, axis=1)
    
    acc = accuracy_score(true_labels, preds)
    pre = precision_score(true_labels, preds, average="macro", zero_division=0)
    rec = recall_score(true_labels, preds, average="macro", zero_division=0)
    f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
    
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
                   max_samples: int = 25000) -> None:
    if len(embeddings_proj) > max_samples:
        np.random.seed(seed)
        indices = np.random.choice(len(embeddings_proj), max_samples, replace=False)
        X_feat = embeddings_proj[indices]
        y_lbl = true_labels[indices]
    else:
        X_feat = embeddings_proj
        y_lbl = true_labels
        
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
        alpha=0.7, 
        s=20 
    )
    
    plt.title(title, fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    
    os.makedirs("tsne_plots", exist_ok=True)
    plot_name = "_".join(title.lower().split())
    save_path = f"tsne_plots/{plot_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    del X_feat, y_lbl, X_2d, label_names
    clear_memory()