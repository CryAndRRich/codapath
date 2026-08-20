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
    weights = np.zeros_like(counts, dtype=np.float32)
    np.divide(
        total,
        num_classes * counts,
        out=weights,
        where=counts > 0,
    )
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


def train_dual_linear(
    image_features: np.ndarray,
    cell_features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    num_epochs: int,
    lr: float,
    device: torch.device,
    *,
    cell_valid: np.ndarray,
    cell_reliability: np.ndarray | None = None,
    consistency_weight: float = 0.0,
    consistency_mode: str = "symmetric_js",
    weight_decay: float = 0.0,
) -> tuple[LinearProbe, LinearProbe]:
    """Jointly fit image/cell probes with optional reliability-gated JSD.

    The image cross-entropy uses every labeled patch.  Cell cross-entropy and
    cross-view consistency only use patches with a valid CellViT view.  The
    full-batch update is intentional: active-learning budgets are at most a
    few hundred points, and keeping the two differently-sized labeled views
    aligned is both simpler and less error-prone than paired mini-batches.

    ``consistency_weight=0`` is the research baseline.  A positive value is an
    ablation, because making the two probes agree too strongly can erase the
    disagreement signal used by acquisition.
    """
    image_features = np.asarray(image_features, dtype=np.float32)
    cell_features = np.asarray(cell_features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    cell_valid = np.asarray(cell_valid, dtype=bool)
    if image_features.ndim != 2 or cell_features.ndim != 2:
        raise ValueError("image_features and cell_features must be 2-D")
    if not (
        len(image_features) == len(cell_features) == len(labels) == len(cell_valid)
    ):
        raise ValueError("Dual-probe arrays must align by labeled patch")
    if len(labels) == 0:
        raise ValueError("Cannot train dual probes with zero labeled samples")
    if not cell_valid.any():
        raise ValueError("Cannot train a cell probe without a valid cell view")
    if consistency_weight < 0.0:
        raise ValueError("consistency_weight must be non-negative")
    valid_consistency_modes = {"symmetric_js", "visual_teacher", "cell_teacher"}
    if consistency_mode not in valid_consistency_modes:
        raise ValueError(
            f"consistency_mode must be one of {sorted(valid_consistency_modes)}"
        )

    if cell_reliability is None:
        reliability = np.ones(len(labels), dtype=np.float32)
    else:
        reliability = np.clip(
            np.asarray(cell_reliability, dtype=np.float32), 0.0, 1.0
        )
        if len(reliability) != len(labels):
            raise ValueError("cell_reliability must align by labeled patch")

    image_probe = LinearProbe(image_features.shape[1], num_classes).to(device)
    cell_probe = LinearProbe(cell_features.shape[1], num_classes).to(device)
    image_weights = torch.as_tensor(
        _class_weights(labels, num_classes), device=device
    )
    cell_weights = torch.as_tensor(
        _class_weights(labels[cell_valid], num_classes), device=device
    )
    image_ce = nn.CrossEntropyLoss(weight=image_weights)
    cell_ce = nn.CrossEntropyLoss(weight=cell_weights)
    optimizer = optim.Adam(
        list(image_probe.parameters()) + list(cell_probe.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    image_x = torch.as_tensor(image_features, dtype=torch.float32, device=device)
    cell_x = torch.as_tensor(
        cell_features[cell_valid], dtype=torch.float32, device=device
    )
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    cell_y = torch.as_tensor(labels[cell_valid], dtype=torch.long, device=device)
    rho = torch.as_tensor(
        reliability[cell_valid], dtype=torch.float32, device=device
    )

    image_probe.train()
    cell_probe.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        image_logits = image_probe(image_x)
        cell_logits = cell_probe(cell_x)
        loss = image_ce(image_logits, y) + cell_ce(cell_logits, cell_y)
        if consistency_weight > 0.0:
            image_valid_logits = image_logits[
                torch.as_tensor(cell_valid, dtype=torch.bool, device=device)
            ]
            p = F.softmax(image_valid_logits, dim=1).clamp_min(1e-8)
            q = F.softmax(cell_logits, dim=1).clamp_min(1e-8)
            if consistency_mode == "symmetric_js":
                middle = 0.5 * (p + q)
                auxiliary = 0.5 * (
                    (p * (p.log() - middle.log())).sum(dim=1)
                    + (q * (q.log() - middle.log())).sum(dim=1)
                )
            elif consistency_mode == "visual_teacher":
                teacher = p.detach()
                auxiliary = (
                    teacher * (teacher.log() - q.log())
                ).sum(dim=1)
            else:
                teacher = q.detach()
                auxiliary = (
                    teacher * (teacher.log() - p.log())
                ).sum(dim=1)
            denom = rho.sum().clamp_min(1e-8)
            loss = loss + consistency_weight * (rho * auxiliary).sum() / denom
        loss.backward()
        optimizer.step()

    return image_probe, cell_probe


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
