"""DINOv2 encoding for RGB regions selected by CellViT instance masks."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from model import DINOv2Extractor


IMAGENET_MEAN_RGB = np.asarray([124, 116, 104], dtype=np.uint8)


def masked_nucleus_crop(
    rgb: np.ndarray,
    instance_map: np.ndarray,
    instance_id: int,
    bbox: np.ndarray,
    padding: float = 0.25,
) -> Image.Image:
    """Crop one nucleus, masking non-instance pixels to ImageNet mean RGB."""
    y0, x0, y1, x1 = [int(value) for value in bbox]
    height = max(1, y1 - y0)
    width = max(1, x1 - x0)
    pad_y = int(round(height * padding))
    pad_x = int(round(width * padding))
    y0 = max(0, y0 - pad_y)
    x0 = max(0, x0 - pad_x)
    y1 = min(rgb.shape[0], y1 + pad_y)
    x1 = min(rgb.shape[1], x1 + pad_x)
    crop = rgb[y0:y1, x0:x1].copy()
    mask = instance_map[y0:y1, x0:x1] == int(instance_id)
    crop[~mask] = IMAGENET_MEAN_RGB
    return Image.fromarray(crop, mode="RGB")


class DINOCellCropEncoder:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        batch_size: int = 256,
    ) -> None:
        self.model = DINOv2Extractor(model_name).to(device).eval()
        self.device = device
        self.batch_size = int(batch_size)
        self.feature_dim = int(self.model.encoder.config.hidden_size)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.inference_mode()
    def encode(self, crops: Sequence[Image.Image]) -> np.ndarray:
        if not crops:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        outputs: List[np.ndarray] = []
        for start in range(0, len(crops), self.batch_size):
            batch = torch.stack([
                self.transform(crop)
                for crop in crops[start:start + self.batch_size]
            ]).to(self.device, non_blocking=True)
            outputs.append(self.model(batch).cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)
