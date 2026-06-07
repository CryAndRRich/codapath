import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from set_up import clear_memory


class LinearProbe(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @torch.inference_mode()
    def predict_proba(self, features_np: np.ndarray, device: torch.device) -> np.ndarray:
        self.eval()
        x = torch.tensor(features_np, device=device, dtype=torch.float32)
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy().astype(np.float32)


def _class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = np.where(counts > 0, total / (num_classes * counts), 0.0)
    return weights.astype(np.float32)


def train_linear(features: np.ndarray,
                 labels: np.ndarray,
                 num_classes: int,
                 num_epochs: int,
                 lr: float,
                 device: torch.device) -> LinearProbe:
    feat_dim = features.shape[1]
    probe = LinearProbe(feat_dim, num_classes).to(device)

    weights = _class_weights(labels, num_classes)
    weight_tensor = torch.tensor(weights, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=min(64, len(labels)), shuffle=True)

    probe.train()
    for _ in range(num_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()

    return probe


def train_knn_linear(all_features: np.ndarray,
                     labeled_indices: List[int],
                     labeled_labels: np.ndarray,
                     num_classes: int,
                     knn_k: int,
                     knn_threshold: float,
                     num_epochs: int,
                     lr: float,
                     device: torch.device) -> LinearProbe:
    labeled_set = set(labeled_indices)
    unlabeled_indices = [i for i in range(len(all_features)) if i not in labeled_set]

    labeled_features = all_features[labeled_indices]
    unlabeled_features = all_features[unlabeled_indices]

    warm_probe = train_linear(labeled_features, labeled_labels, num_classes,
                              max(1, num_epochs // 2), lr, device)

    labeled_tensor = torch.tensor(labeled_features, device=device, dtype=torch.float32)
    labeled_norm = F.normalize(labeled_tensor, p=2, dim=1)

    unlabeled_tensor = torch.tensor(unlabeled_features, device=device, dtype=torch.float32)
    unlabeled_norm = F.normalize(unlabeled_tensor, p=2, dim=1)

    k = min(knn_k, len(labeled_indices))
    sim = torch.matmul(unlabeled_norm, labeled_norm.T)            
    _, topk_ids = torch.topk(sim, k=k, dim=1)                  
    topk_ids_np = topk_ids.cpu().numpy()

    pseudo_labels = np.array([
        np.bincount(labeled_labels[topk_ids_np[i]], minlength=num_classes).argmax()
        for i in range(len(unlabeled_indices))
    ], dtype=np.int64)

    del labeled_tensor, labeled_norm, unlabeled_tensor, unlabeled_norm, sim, topk_ids
    clear_memory()

    unlabeled_probs = warm_probe.predict_proba(unlabeled_features, device)
    confident_mask = unlabeled_probs.max(axis=1) >= knn_threshold
    del warm_probe

    combined_features = np.vstack([labeled_features, unlabeled_features[confident_mask]])
    combined_labels = np.concatenate([labeled_labels, pseudo_labels[confident_mask]])

    final_probe = train_linear(combined_features, combined_labels, num_classes,
                               num_epochs, lr, device)

    clear_memory()
    return final_probe


def save_model(probe: LinearProbe, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "feat_dim": probe.fc.in_features,
        "num_classes": probe.fc.out_features,
        "weight": probe.fc.weight.data.cpu(),
        "bias": probe.fc.bias.data.cpu(),
    }, save_path)


def load_model(load_path: str, device: torch.device) -> LinearProbe:
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    probe = LinearProbe(checkpoint["feat_dim"], checkpoint["num_classes"])
    probe.fc.weight.data.copy_(checkpoint["weight"].to(device))
    probe.fc.bias.data.copy_(checkpoint["bias"].to(device))
    return probe.to(device).eval()