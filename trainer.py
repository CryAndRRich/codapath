import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class LinearProbe(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @torch.inference_mode()
    def predict_logits(self,
                       features_np: np.ndarray,
                       device: torch.device,
                       batch_size: int = 8192) -> np.ndarray:
        self.eval()
        outputs = []
        for start in range(0, len(features_np), batch_size):
            x = torch.as_tensor(
                features_np[start:start + batch_size],
                device=device,
                dtype=torch.float32,
            )
            outputs.append(self(x).cpu().numpy().astype(np.float32))
        if not outputs:
            return np.empty((0, self.fc.out_features), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    @torch.inference_mode()
    def predict_proba(self,
                      features_np: np.ndarray,
                      device: torch.device,
                      batch_size: int = 8192) -> np.ndarray:
        logits = self.predict_logits(features_np, device, batch_size=batch_size)
        if len(logits) == 0:
            return logits
        logits_t = torch.from_numpy(logits)
        return F.softmax(logits_t, dim=1).numpy().astype(np.float32)


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
                 device: torch.device,
                 weight_decay: float = 0.0) -> LinearProbe:
    if len(labels) == 0:
        raise ValueError("Cannot train a linear probe with zero labeled samples")
    feat_dim = features.shape[1]
    probe = LinearProbe(feat_dim, num_classes).to(device)

    weights = _class_weights(labels, num_classes)
    weight_tensor = torch.tensor(weights, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

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