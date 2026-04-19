from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from set_up import clear_memory


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int, device: torch.device) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        features = F.normalize(features, p=2, dim=1)
        centers_batch = self.centers.index_select(0, labels)
        return (features - centers_batch).pow(2).sum() / 2.0 / batch_size


def train_contrastive(
    model: nn.Module,
    labeled_loader: DataLoader,
    num_epochs: int,
    learn_rate: float,
    device: torch.device,
    use_center_loss: bool = True,
    verbose: bool = False,
) -> nn.Module:
    model = model.to(device)
    model.train()
    try:
        compiled_model = torch.compile(model)
        if verbose:
            print("Enabled torch.compile()")
    except Exception:
        compiled_model = model

    num_classes = model.classification_head.out_features
    feat_dim = model.classification_head.in_features

    if hasattr(labeled_loader.dataset, "labels"):
        all_labels = labeled_loader.dataset.labels
        valid_labels = all_labels[all_labels >= 0]
        class_counts = np.bincount(valid_labels, minlength=num_classes)
    else:
        class_counts = np.zeros(num_classes)
        for _, labels in labeled_loader:
            class_counts += np.bincount(labels.cpu().numpy(), minlength=num_classes)

    total_samples = np.sum(class_counts)
    class_weights = np.ones(num_classes, dtype=np.float32)
    for class_idx in range(num_classes):
        if class_counts[class_idx] > 0:
            class_weights[class_idx] = total_samples / (num_classes * class_counts[class_idx])
        else:
            class_weights[class_idx] = 0.0

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    trainable_params = [param for param in compiled_model.parameters() if param.requires_grad]
    optimizer_model = optim.AdamW(trainable_params, lr=learn_rate, weight_decay=1e-4)

    criterion_ce = nn.CrossEntropyLoss(weight=weight_tensor)
    criterion_center = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, device=device)
    optimizer_center = optim.SGD(criterion_center.parameters(), lr=0.5) if use_center_loss else None
    center_loss_weight = 0.05

    scaler = GradScaler(device="cuda", enabled=device.type == "cuda")

    for epoch in range(num_epochs):
        total_loss = 0.0
        batches = 0

        for images, labels in labeled_loader:
            images = images.to(device)
            labels = labels.long().to(device)
            optimizer_model.zero_grad()
            if optimizer_center is not None:
                optimizer_center.zero_grad()

            amp_context = autocast(device_type=device.type) if device.type == "cuda" else nullcontext()
            with amp_context:
                _, projected_features, logits = compiled_model(images)
                loss_ce = criterion_ce(logits, labels)
                if use_center_loss:
                    loss_center = criterion_center(projected_features, labels)
                    loss_total = loss_ce + center_loss_weight * loss_center
                else:
                    loss_center = None
                    loss_total = loss_ce

            scaler.scale(loss_total).backward()
            scaler.step(optimizer_model)
            if optimizer_center is not None:
                scaler.step(optimizer_center)
            scaler.update()

            total_loss += loss_total.item()
            batches += 1

            del images, labels, projected_features, logits, loss_ce, loss_total
            if loss_center is not None:
                del loss_center

        if verbose:
            avg_loss = total_loss / batches if batches > 0 else 0.0
            print(f"Epoch [{epoch + 1:02d}/{num_epochs}] | Loss: {avg_loss:.4f}")

        clear_memory()

    return compiled_model

