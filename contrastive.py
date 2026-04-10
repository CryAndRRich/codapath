import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from set_up import clear_memory

class CenterLoss(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 feat_dim: int, 
                 device=torch.device) -> None:
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device        
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, 
                features: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)        
        features = F.normalize(features, p=2, dim=1)
        centers_batch = self.centers.index_select(0, labels)
        loss = (features - centers_batch).pow(2).sum() / 2.0 / batch_size
        return loss
    
def train_contrastive(model: nn.Module, 
                      labeled_loader: DataLoader, 
                      num_epochs: int, 
                      learn_rate: float, 
                      device: torch.device,
                      verbose: bool = False) -> nn.Module:
    model = model.to(device)
    model.train() 
    try:
        compiled_model = torch.compile(model)
        print("Kích hoạt torch.compile()")
    except:
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
            counts = np.bincount(labels.cpu().numpy(), minlength=num_classes)
            class_counts += counts

    total_samples = np.sum(class_counts)
    class_weights = np.ones(num_classes, dtype=np.float32)    
    for c in range(num_classes):
        if class_counts[c] > 0:
            class_weights[c] = total_samples / (num_classes * class_counts[c])
        else:
            class_weights[c] = 0.0    
            
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)  

    trainable_params = [p for p in compiled_model.parameters() if p.requires_grad]
    optimizer_model = optim.AdamW(
        trainable_params, 
        lr=learn_rate, 
        weight_decay=1e-4
    )    

    criterion_ce = nn.CrossEntropyLoss(weight=weight_tensor)
    criterion_center = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, device=device)
    optimizer_center = optim.SGD(criterion_center.parameters(), lr=0.5)   
    alpha = 0.05  

    scaler = GradScaler(device="cuda")
    for epoch in range(num_epochs):
        total_loss = 0.0
        batches = 0

        for images, labels in labeled_loader:
            images, labels = images.to(device), labels.long().to(device)
            optimizer_model.zero_grad()
            optimizer_center.zero_grad()

            with autocast(device_type="cuda"):
                _, f_proj, logits = compiled_model(images)
                loss_ce = criterion_ce(logits, labels)
                loss_cent = criterion_center(f_proj, labels)
                loss_total = loss_ce + alpha * loss_cent  

            scaler.scale(loss_total).backward()
            scaler.step(optimizer_model)
            scaler.step(optimizer_center)
            scaler.update()

            total_loss += loss_total.item()
            batches += 1

            del images, labels, f_proj, logits, loss_ce, loss_cent, loss_total

        if verbose:
            avg_loss = total_loss / batches if batches > 0 else 0
            print(f"Epoch [{epoch + 1:02d}/{num_epochs}] | Loss: {avg_loss:.4f}")
        
        clear_memory()

    return compiled_model