"""Thin patch-level adapter around the official CellViT++ implementation.

The optional ``cellvit`` dependency is imported only when this extractor is
constructed. Active-learning runs consume the portable cache and therefore do
not need the CellViT environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from .postprocess import stack_prediction_maps_numpy


SUPPORTED_CELLVIT_VERSION = "1.0.9"


@dataclass
class PatchCells:
    rgb: np.ndarray
    instance_map: np.ndarray
    instance_ids: np.ndarray
    embeddings: np.ndarray
    bboxes: np.ndarray
    confidence: np.ndarray


def _unflatten_config(flat: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in flat.items():
        current = result
        parts = key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result


def _pad_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


class CellViTPatchExtractor:
    """Run CellViT-256 on raw RGB patches and return cells plus tokens."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        *,
        input_mpp: float,
        model_mpp: float,
        magnification: int,
        min_cell_area: int = 10,
        max_cells_per_patch: Optional[int] = None,
        mixed_precision: bool = True,
    ) -> None:
        if input_mpp <= 0 or model_mpp <= 0:
            raise ValueError("input_mpp and model_mpp must be positive")
        if magnification not in {20, 40}:
            raise ValueError("magnification must be 20 or 40")
        expected_model_mpp = {20: 0.5, 40: 0.25}[magnification]
        if abs(float(model_mpp) - expected_model_mpp) > 0.035:
            raise ValueError(
                f"CellViT {magnification}x expects model_mpp≈"
                f"{expected_model_mpp}, got {model_mpp}"
            )
        checkpoint_name = os.path.basename(checkpoint_path).lower()
        if "x20" in checkpoint_name and magnification != 20:
            raise ValueError("An x20 checkpoint must use magnification=20")
        if "x40" in checkpoint_name and magnification != 40:
            raise ValueError("An x40 checkpoint must use magnification=40")
        try:
            installed_version = importlib.metadata.version("cellvit")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportError(
                "Install the verified wheel with: pip install --no-deps "
                f"cellvit=={SUPPORTED_CELLVIT_VERSION}"
            ) from exc
        if installed_version != SUPPORTED_CELLVIT_VERSION:
            raise RuntimeError(
                f"This adapter is verified against cellvit=="
                f"{SUPPORTED_CELLVIT_VERSION}, found {installed_version}."
            )
        try:
            from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
            from cellvit.models.cell_segmentation.postprocessing import (
                DetectionCellPostProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "CellViT is required only for feature extraction. Install the "
                "official TIO-IKIM CellViT-plus-plus package/environment first."
            ) from exc

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        architecture = checkpoint.get("arch")
        if architecture != "CellViT256":
            raise NotImplementedError(
                "The first controlled experiment supports CellViT256 only; "
                f"checkpoint architecture is {architecture!r}."
            )
        run_config = _unflatten_config(checkpoint["config"])
        self.model = CellViT256(
            model256_path=None,
            num_nuclei_classes=run_config["data"]["num_nuclei_classes"],
            num_tissue_classes=run_config["data"]["num_tissue_classes"],
            regression_loss=run_config.get("model", {}).get(
                "regression_loss", False
            ),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        normalize = run_config.get("transformations", {}).get("normalize", {})
        self.mean = tuple(normalize.get("mean", (0.5, 0.5, 0.5)))
        self.std = tuple(normalize.get("std", (0.5, 0.5, 0.5)))
        self.num_nuclei_classes = int(run_config["data"]["num_nuclei_classes"])
        self.postprocessor = DetectionCellPostProcessor(
            nr_types=self.num_nuclei_classes,
            magnification=magnification,
        )
        self.device = device
        self.input_mpp = float(input_mpp)
        self.model_mpp = float(model_mpp)
        self.min_cell_area = int(min_cell_area)
        self.max_cells_per_patch = max_cells_per_patch
        self.mixed_precision = bool(mixed_precision and device.type == "cuda")
        patch_size = self.model.patch_size
        if isinstance(patch_size, (tuple, list)):
            if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
                raise ValueError(f"CellViT patch size must be square, got {patch_size}")
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
        self.embedding_dim = int(self.model.embed_dim)

    def _resize(self, image: Image.Image) -> Image.Image:
        scale = self.input_mpp / self.model_mpp
        if abs(scale - 1.0) < 1e-6:
            return image.convert("RGB")
        width = max(1, int(round(image.width * scale)))
        height = max(1, int(round(image.height * scale)))
        return image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)

    def _prepare_batch(self, images: Sequence[Image.Image]):
        resized = [self._resize(image) for image in images]
        max_height = _pad_to_multiple(
            max(image.height for image in resized), self.patch_size
        )
        max_width = _pad_to_multiple(
            max(image.width for image in resized), self.patch_size
        )
        tensors = []
        valid_shapes = []
        for image in resized:
            valid_shapes.append((image.height, image.width))
            tensor = TF.to_tensor(image)
            tensor = TF.normalize(tensor, self.mean, self.std)
            tensor = F.pad(
                tensor,
                (0, max_width - image.width, 0, max_height - image.height),
                value=0.0,
            )
            tensors.append(tensor)
        return torch.stack(tensors), resized, valid_shapes

    @staticmethod
    def _prepare_predictions(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "nuclei_binary_map": F.softmax(
                outputs["nuclei_binary_map"], dim=1
            ).permute(0, 2, 3, 1),
            "nuclei_type_map": F.softmax(
                outputs["nuclei_type_map"], dim=1
            ).permute(0, 2, 3, 1),
            "hv_map": outputs["hv_map"].permute(0, 2, 3, 1),
        }

    @staticmethod
    def _post_process_without_numba_stack(postprocessor, predictions):
        """Use official per-image CellViT processing after NumPy map stacking.

        ``DetectionCellPostProcessor.post_process_batch`` delegates only its
        initial argmax/stack operation to a Numba function that is incompatible
        with the current Kaggle Python 3.12 image. Everything after this small
        compatibility boundary remains the official CellViT implementation.
        """
        prediction_arrays = {
            key: value.detach().cpu().numpy()
            for key, value in predictions.items()
        }
        pred_maps = stack_prediction_maps_numpy(
            prediction_arrays["nuclei_type_map"],
            prediction_arrays["nuclei_binary_map"],
            prediction_arrays["hv_map"],
        )
        instance_maps = []
        cell_dicts = []
        for pred_map in pred_maps:
            pred_inst, cells = postprocessor.post_process_single_image(pred_map)
            instance_maps.append(pred_inst)
            cell_dicts.append(cells)
        return np.stack(instance_maps).astype(np.int32, copy=False), cell_dicts

    @torch.inference_mode()
    def extract_batch(self, images: Sequence[Image.Image]) -> List[PatchCells]:
        if not images:
            return []
        batch, resized, valid_shapes = self._prepare_batch(images)
        batch = batch.to(self.device, non_blocking=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.mixed_precision,
        ):
            outputs = self.model(batch, retrieve_tokens=True)
        predictions = self._prepare_predictions(outputs)
        instance_maps, cell_dicts = self._post_process_without_numba_stack(
            self.postprocessor, predictions
        )
        tokens = outputs["tokens"].detach().cpu()
        if tokens.ndim != 4:
            raise ValueError(
                f"Expected CellViT tokens with shape (B,C,H,W), got {tokens.shape}"
            )
        instance_maps = np.asarray(instance_maps, dtype=np.int32)

        results: List[PatchCells] = []
        for batch_idx, (height, width) in enumerate(valid_shapes):
            instance_map = instance_maps[batch_idx, :height, :width]
            cells = []
            for instance_id, cell in cell_dicts[batch_idx].items():
                bbox = np.asarray(cell["bbox"], dtype=np.int32)
                y0, x0 = bbox[0]
                y1, x1 = bbox[1]
                y0, x0 = max(0, y0), max(0, x0)
                y1, x1 = min(height, y1), min(width, x1)
                if y1 <= y0 or x1 <= x0:
                    continue
                area = int(np.count_nonzero(instance_map[y0:y1, x0:x1] == instance_id))
                if area < self.min_cell_area:
                    continue
                token_y0 = max(0, y0 // self.patch_size)
                token_x0 = max(0, x0 // self.patch_size)
                token_y1 = min(tokens.shape[2], math.ceil(y1 / self.patch_size))
                token_x1 = min(tokens.shape[3], math.ceil(x1 / self.patch_size))
                cell_token = tokens[
                    batch_idx, :, token_y0:token_y1, token_x0:token_x1
                ].flatten(1).mean(dim=1)
                confidence = float(cell.get("type_prob", 1.0))
                cells.append((
                    int(instance_id), confidence,
                    np.asarray([y0, x0, y1, x1], dtype=np.int32),
                    cell_token.numpy().astype(np.float32),
                ))

            cells.sort(key=lambda item: (-item[1], item[0]))
            if self.max_cells_per_patch is not None:
                cells = cells[:self.max_cells_per_patch]
            instance_ids = np.asarray([cell[0] for cell in cells], dtype=np.int32)
            confidence = np.asarray([cell[1] for cell in cells], dtype=np.float32)
            bboxes = np.asarray([cell[2] for cell in cells], dtype=np.int32).reshape(-1, 4)
            embeddings = np.asarray(
                [cell[3] for cell in cells], dtype=np.float32
            ).reshape(-1, self.embedding_dim)
            results.append(PatchCells(
                rgb=np.asarray(resized[batch_idx], dtype=np.uint8),
                instance_map=instance_map,
                instance_ids=instance_ids,
                embeddings=embeddings,
                bboxes=bboxes,
                confidence=confidence,
            ))
        return results
