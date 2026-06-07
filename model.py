import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Dinov2Model


class DINOv2Extractor(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-base") -> None:
        super().__init__()
        self.encoder = Dinov2Model.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values=x).last_hidden_state[:, 0, :] 


@torch.inference_mode()
def extract_image_features(dataloader: DataLoader,
                            extractor: nn.Module,
                            device: torch.device) -> np.ndarray:
    extractor = extractor.to(device)
    extractor.eval()

    all_features = []
    for images, _ in tqdm(dataloader, desc="Extracting image features", leave=False):
        images = images.to(device, non_blocking=True)
        features = extractor(images)
        all_features.append(features.cpu().numpy().astype(np.float32))

    return np.vstack(all_features)