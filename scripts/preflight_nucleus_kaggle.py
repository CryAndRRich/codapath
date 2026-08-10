"""Fail-fast Kaggle preflight for the exact nucleus extraction path.

This loads the real checkpoint, runs CellViT post-processing and DINOv2 on a
small raw-RGB batch, and estimates cache storage before a full extraction.
"""

import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from load_data import RawRGBDataset, get_data_loaders
from nucleus.cellvit_extractor import (
    CellViTPatchExtractor,
    SUPPORTED_CELLVIT_VERSION,
)
from nucleus.crop_dino import DINOCellCropEncoder, masked_nucleus_crop


CELLVIT_VERSION = SUPPORTED_CELLVIT_VERSION


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--vit_name", default="facebook/dinov2-base",
        help="Hugging Face model ID or local DINOv2 directory.",
    )
    parser.add_argument("--cache_dir", default="/kaggle/working/nucleus_features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_mpp", type=float, default=None)
    parser.add_argument("--model_mpp", type=float, default=None)
    parser.add_argument("--magnification", type=int, choices=[20, 40], default=None)
    parser.add_argument("--smoke_samples", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _require_runtime() -> None:
    required = ["cv2", "einops", "numba", "pandas", "scipy", "skimage"]
    missing = []
    for module in required:
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing.append(f"{module}: {exc}")
    if missing:
        raise RuntimeError(
            "CellViT runtime imports failed. Do not install the full dependency "
            "tree over Kaggle's PyTorch; install only the missing packages.\n- "
            + "\n- ".join(missing)
        )
    try:
        version = importlib.metadata.version("cellvit")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Install the pinned wheel first: pip install --no-deps "
            f"cellvit=={CELLVIT_VERSION}"
        ) from exc
    if version != CELLVIT_VERSION:
        raise RuntimeError(
            f"Expected cellvit=={CELLVIT_VERSION}, found {version}. "
            "The adapter is verified against the pinned wheel only."
        )


def _resolve_scale(args, config):
    extraction = config["nucleus_extraction"]
    dataset_cfg = extraction.get("datasets", {}).get(args.dataset, {})
    input_mpp = args.input_mpp or dataset_cfg.get("input_mpp")
    model_mpp = args.model_mpp or dataset_cfg.get("model_mpp")
    magnification = args.magnification or dataset_cfg.get("magnification")
    if input_mpp is None or model_mpp is None or magnification is None:
        raise ValueError(
            "MPP/magnification unresolved. Pass them explicitly; HistoSet must "
            "not use one guessed global scale."
        )
    return float(input_mpp), float(model_mpp), int(magnification)


def main() -> None:
    args = parse_args()
    if args.smoke_samples < 1:
        raise ValueError("smoke_samples must be positive")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.dataset not in config["datasets"]:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    _require_runtime()
    data_path = Path(args.data_path)
    checkpoint = Path(args.checkpoint)
    cache_dir = Path(args.cache_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Kaggle dataset path missing: {data_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CellViT checkpoint missing: {checkpoint}")
    if str(cache_dir).startswith("/kaggle/input/"):
        raise ValueError("cache_dir is under read-only /kaggle/input")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir):
        pass

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Attach a Kaggle GPU and use --device cuda")
    gpu = torch.cuda.get_device_properties(0)
    free_vram, total_vram = torch.cuda.mem_get_info()
    free_disk = shutil.disk_usage(cache_dir).free
    print(
        f"[env] python={sys.version.split()[0]} torch={torch.__version__} "
        f"cellvit={CELLVIT_VERSION}"
    )
    print(
        f"[gpu] {gpu.name} total={total_vram / 2**30:.1f} GiB "
        f"free={free_vram / 2**30:.1f} GiB"
    )
    print(f"[disk] writable={cache_dir} free={free_disk / 2**30:.1f} GiB")

    checkpoint_obj = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    required_keys = {"arch", "config", "model_state_dict"}
    missing_keys = required_keys - checkpoint_obj.keys()
    if missing_keys:
        raise ValueError(f"Checkpoint missing keys: {sorted(missing_keys)}")
    if checkpoint_obj["arch"] != "CellViT256":
        raise ValueError(
            f"Expected CellViT256 checkpoint, found {checkpoint_obj['arch']!r}"
        )
    del checkpoint_obj

    input_mpp, model_mpp, magnification = _resolve_scale(args, config)
    train_loader, _, _ = get_data_loaders(
        str(data_path), args.seed, verbose=True
    )
    raw_dataset = RawRGBDataset(train_loader.dataset)
    count = min(args.smoke_samples, len(raw_dataset))
    images = [raw_dataset[idx][0] for idx in range(count)]
    if not images or not all(isinstance(image, Image.Image) for image in images):
        raise RuntimeError("Raw-RGB loader did not return PIL images")
    print(f"[data] smoke raw sizes={[image.size for image in images]}")

    extraction = config["nucleus_extraction"]
    cellvit = CellViTPatchExtractor(
        str(checkpoint), device,
        input_mpp=input_mpp,
        model_mpp=model_mpp,
        magnification=magnification,
        min_cell_area=extraction.get("min_cell_area", 10),
        max_cells_per_patch=extraction.get("max_cells_per_patch"),
    )
    cellvit_start = time.perf_counter()
    patches = cellvit.extract_batch(images)
    cellvit_seconds = time.perf_counter() - cellvit_start
    if len(patches) != count:
        raise RuntimeError("CellViT output batch length mismatch")
    for patch in patches:
        if patch.instance_map.shape != patch.rgb.shape[:2]:
            raise RuntimeError("CellViT instance map/RGB shape mismatch")
        if patch.embeddings.shape != (len(patch.instance_ids), cellvit.embedding_dim):
            raise RuntimeError("CellViT token rows do not align with instances")
    detected = sum(len(patch.instance_ids) for patch in patches)
    print(f"[cellvit] smoke patches={count} detected_cells={detected}")

    crop_dino = DINOCellCropEncoder(
        args.vit_name,
        device, batch_size=1,
    )
    crops = []
    for patch in patches:
        for instance_id, bbox in zip(patch.instance_ids, patch.bboxes):
            crops.append(masked_nucleus_crop(
                patch.rgb, patch.instance_map, int(instance_id), bbox,
                padding=float(extraction.get("crop_padding", 0.25)),
            ))
            if len(crops) >= 8:
                break
        if len(crops) >= 8:
            break
    # Always execute one DINO forward even when CellViT detects no nucleus.
    dino_inputs = crops or [Image.fromarray(patches[0].rgb)]
    dino_start = time.perf_counter()
    dino_features = crop_dino.encode(dino_inputs)
    dino_seconds = time.perf_counter() - dino_start
    if dino_features.shape != (len(dino_inputs), crop_dino.feature_dim):
        raise RuntimeError("DINO crop feature shape mismatch")
    if not np.all(np.isfinite(dino_features)):
        raise RuntimeError("DINO crop features contain non-finite values")
    print(f"[dino] smoke feature shape={dino_features.shape}")

    mean_cells = detected / count
    bytes_per_cell = 2 * (cellvit.embedding_dim + crop_dino.feature_dim) + 20
    estimated_bytes = len(raw_dataset) * mean_cells * bytes_per_cell
    peak_bytes = 3 * estimated_bytes  # resumable shards + assembly memmap + final cache
    print(
        f"[estimate] mean_cells={mean_cells:.1f}/patch, "
        f"cache≈{estimated_bytes / 2**30:.2f} GiB, "
        f"peak_disk≈{peak_bytes / 2**30:.2f} GiB (rough smoke estimate)"
    )
    seconds_per_patch = cellvit_seconds / count
    seconds_per_crop = dino_seconds / len(dino_inputs)
    estimated_hours = len(raw_dataset) * (
        seconds_per_patch + mean_cells * seconds_per_crop
    ) / 3600
    print(
        f"[estimate] cold-start runtime upper estimate≈{estimated_hours:.1f} h "
        "(preflight batch=1 and first-call JIT make this deliberately pessimistic)"
    )
    if peak_bytes > free_disk:
        raise RuntimeError(
            "Estimated peak extraction storage exceeds free Kaggle disk"
        )
    if detected == 0:
        print(
            "[warning] No nucleus in the smoke batch. Runtime is valid, but "
            "inspect more QC patches before trusting the experiment."
        )
    print("[PREFLIGHT PASS] exact CellViT + postprocess + DINO path is runnable")


if __name__ == "__main__":
    main()
