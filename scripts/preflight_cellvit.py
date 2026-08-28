"""Fail-fast Kaggle preflight for the exact nucleus extraction path.

This loads the real checkpoint, runs CellViT post-processing and DINOv2 on a
small raw-RGB batch, and estimates cache storage before a full extraction.
"""

import argparse
import importlib
import importlib.metadata
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

from data.loaders import RawRGBDataset, get_data_loaders
from features.cellvit.extractor import (
    CellViTPatchExtractor,
    SUPPORTED_CELLVIT_VERSION,
)
from features.cellvit.crops import DINOCellCropEncoder, masked_nucleus_crop


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
    parser.add_argument("--cache_dir", default="/kaggle/working/cellvit_features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_mpp", type=float, default=None)
    parser.add_argument("--model_mpp", type=float, default=None)
    parser.add_argument("--magnification", type=int, choices=[20, 40], default=None)
    parser.add_argument("--smoke_samples", type=int, default=8)
    parser.add_argument("--cellvit_batch_size", type=int, default=2)
    parser.add_argument("--dino_crop_batch_size", type=int, default=32)
    parser.add_argument("--max_estimated_hours", type=float, default=None)
    parser.add_argument(
        "--shards", type=int, default=1,
        help="How many GPUs the real extraction will split across. Preflight "
             "benchmarks ONE GPU, so without this the estimate is the "
             "single-card figure and the safety gate rejects runs that would "
             "comfortably finish on two.",
    )
    parser.add_argument("--max_cells_per_patch", type=int, default=None)
    parser.add_argument(
        "--skip_crop_dino", action="store_true",
        help="Estimate for a run that stores only CellViT cell embeddings.",
    )
    parser.add_argument(
        "--save_instance_maps", action="store_true",
        help="Include the compressed instance-map sidecar in the size estimate.",
    )
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
    extraction = config["cellvit_extraction"]
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
    if args.cellvit_batch_size < 1:
        raise ValueError("cellvit_batch_size must be positive")
    if args.dino_crop_batch_size < 1:
        raise ValueError("dino_crop_batch_size must be positive")
    if args.max_estimated_hours is not None and args.max_estimated_hours <= 0:
        raise ValueError("max_estimated_hours must be positive")
    if args.shards < 1:
        raise ValueError("shards must be positive")
    if args.max_cells_per_patch is not None and args.max_cells_per_patch < 1:
        raise ValueError("max_cells_per_patch must be positive")
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
    print("[compat] CellViT prediction-map stack: NumPy float32 (Numba JIT bypass)")
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
    scale_factor = input_mpp / model_mpp
    print(
        f"[scale] input={input_mpp:.4f} MPP -> CellViT {magnification}x "
        f"at {model_mpp:.4f} MPP (resize={scale_factor:.3f}x)"
    )
    if scale_factor > 1.5:
        print(
            "[scale warning] Input is being upsampled substantially. This "
            "matches the checkpoint pixel scale but cannot recover spatial "
            "detail absent from the source image. Prefer a matching x20 "
            "checkpoint for native 0.5-MPP data when available."
        )
    train_loader, _, _ = get_data_loaders(
        str(data_path), args.seed, verbose=True
    )
    raw_dataset = RawRGBDataset(train_loader.dataset)
    count = min(args.smoke_samples, len(raw_dataset))
    smoke_indices = np.linspace(
        0, len(raw_dataset) - 1, num=count, dtype=np.int64
    ).tolist()
    images = [raw_dataset[idx][0] for idx in smoke_indices]
    if not images or not all(isinstance(image, Image.Image) for image in images):
        raise RuntimeError("Raw-RGB loader did not return PIL images")
    if any(min(image.size) < 128 for image in images):
        raise RuntimeError(
            "Smoke data contains patches smaller than 128 px. For PathMNIST, "
            "mount pathmnist_224.npz rather than the native 28 px release."
        )
    print(
        f"[data] smoke indices={smoke_indices} "
        f"raw sizes={[image.size for image in images]}"
    )

    extraction = config["cellvit_extraction"]
    cellvit = CellViTPatchExtractor(
        str(checkpoint), device,
        input_mpp=input_mpp,
        model_mpp=model_mpp,
        magnification=magnification,
        min_cell_area=extraction.get("min_cell_area", 10),
        max_cells_per_patch=(
            args.max_cells_per_patch
            if args.max_cells_per_patch is not None
            else extraction.get("max_cells_per_patch")
        ),
    )
    # Warm up model kernels and post-processing, then benchmark the exact batch.
    cellvit.extract_batch(images[:args.cellvit_batch_size])
    cellvit_start = time.perf_counter()
    patches = []
    for start in range(0, count, args.cellvit_batch_size):
        patches.extend(
            cellvit.extract_batch(images[start:start + args.cellvit_batch_size])
        )
    cellvit_seconds = time.perf_counter() - cellvit_start
    if len(patches) != count:
        raise RuntimeError("CellViT output batch length mismatch")
    for patch in patches:
        if patch.instance_map.shape != patch.rgb.shape[:2]:
            raise RuntimeError("CellViT instance map/RGB shape mismatch")
        if patch.embeddings.shape != (len(patch.instance_ids), cellvit.embedding_dim):
            raise RuntimeError("CellViT token rows do not align with instances")
    detected = sum(len(patch.instance_ids) for patch in patches)
    print(
        f"[cellvit] smoke patches={count} detected_cells={detected} "
        f"(configured batch={args.cellvit_batch_size})"
    )

    crop_dino = None if args.skip_crop_dino else DINOCellCropEncoder(
        args.vit_name,
        device, batch_size=args.dino_crop_batch_size,
    )
    # With --skip_crop_dino none of this runs: the crops are never cut, the
    # encoder is never built, and neither its bytes nor its hours belong in an
    # estimate for a run that will not produce them.
    if crop_dino is None:
        dino_seconds = 0.0
        crop_dino_dim = 0
        print("[dino] skipped (--skip_crop_dino): no crop features will be stored")
        dino_inputs = []
    else:
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
        # Execute one full configured DINO batch even when the smoke patches
        # contain fewer nuclei. Repeating a real crop is sufficient to test peak
        # memory.
        seed_inputs = crops or [Image.fromarray(patches[0].rgb)]
        dino_inputs = [
            seed_inputs[idx % len(seed_inputs)]
            for idx in range(args.dino_crop_batch_size)
        ]
        crop_dino.encode(dino_inputs)  # warm-up
        dino_start = time.perf_counter()
        dino_features = crop_dino.encode(dino_inputs)
        dino_seconds = time.perf_counter() - dino_start
        crop_dino_dim = crop_dino.feature_dim
        if dino_features.shape != (len(dino_inputs), crop_dino_dim):
            raise RuntimeError("DINO crop feature shape mismatch")
        if not np.all(np.isfinite(dino_features)):
            raise RuntimeError("DINO crop features contain non-finite values")
        print(
            f"[dino] smoke feature shape={dino_features.shape} "
            f"(configured batch={args.dino_crop_batch_size})"
        )

    mean_cells = detected / count
    bytes_per_cell = 2 * (cellvit.embedding_dim + crop_dino_dim) + 20
    estimated_bytes = len(raw_dataset) * mean_cells * bytes_per_cell
    peak_bytes = 3 * estimated_bytes  # resumable shards + assembly memmap + final cache

    # The instance-map sidecar, measured rather than assumed: compress the maps
    # this smoke batch actually produced. A label image is mostly background and
    # piecewise constant, so the ratio is large (hundreds of x) but it depends on
    # nucleus density, which is exactly what varies by dataset.
    instance_map_bytes = 0
    if args.save_instance_maps:
        import zlib

        sampled = [
            len(zlib.compress(
                np.ascontiguousarray(patch.instance_map, dtype=np.uint16).tobytes(), 6
            ))
            for patch in patches
        ]
        per_patch = sum(sampled) / max(1, len(sampled))
        raw_per_patch = patches[0].instance_map.size * 2 if patches else 0
        instance_map_bytes = len(raw_dataset) * per_patch
        ratio = raw_per_patch / per_patch if per_patch else 0
        print(
            f"[estimate] instance maps: {per_patch / 1024:.1f} KiB/patch "
            f"({ratio:.0f}x smaller than raw uint16) -> "
            f"{instance_map_bytes / 2**30:.2f} GiB total"
        )
        # Blobs exist per-batch during the build and once merged afterwards.
        peak_bytes += 2 * instance_map_bytes
    print(
        f"[estimate] mean_cells={mean_cells:.1f}/patch, "
        f"cache≈{(estimated_bytes + instance_map_bytes) / 2**30:.2f} GiB, "
        f"peak_disk≈{peak_bytes / 2**30:.2f} GiB (rough smoke estimate)"
    )
    seconds_per_patch = cellvit_seconds / count
    seconds_per_crop = (dino_seconds / len(dino_inputs)) if dino_inputs else 0.0
    cellvit_hours = len(raw_dataset) * seconds_per_patch / 3600
    crop_dino_hours = (
        len(raw_dataset) * mean_cells * seconds_per_crop / 3600
    )
    gpu_hours = cellvit_hours + crop_dino_hours
    # What the session will actually take. Extraction splits whole batches over
    # the shards, and the shards are independent processes on separate cards, so
    # wall clock is GPU-hours divided by the number of cards. Measured against
    # real runs: the one-GPU figure overestimated a two-GPU session by 1.3-1.5x
    # (pathmnist 4.5h -> 3.0h, histoset 5.2h -> 4.0h), which is this factor plus
    # the pilot's warm-up overhead.
    estimated_hours = gpu_hours / args.shards
    print(
        f"[estimate] warmed pilot runtime≈{gpu_hours:.1f} GPU-h "
        f"at CellViT/DINO batches={args.cellvit_batch_size}/"
        f"{args.dino_crop_batch_size}"
    )
    print(
        f"[estimate] runtime breakdown: CellViT≈{cellvit_hours:.1f} h + "
        f"crop-DINO≈{crop_dino_hours:.1f} h"
    )
    if args.shards > 1:
        print(
            f"[estimate] wall clock over {args.shards} GPUs "
            f"≈{estimated_hours:.1f} h (this is what the 12-hour Kaggle limit "
            "applies to)"
        )
    disk_cap = max(
        1,
        int(0.8 * free_disk / (3 * len(raw_dataset) * bytes_per_cell)),
    )
    cap_candidates = [disk_cap]
    budget_gpu_hours = (
        None if args.max_estimated_hours is None
        else args.max_estimated_hours * args.shards
    )
    if (
        budget_gpu_hours is not None
        and budget_gpu_hours > cellvit_hours
        and seconds_per_crop > 0
    ):
        time_cap = int(
            (budget_gpu_hours - cellvit_hours)
            * 3600
            / (len(raw_dataset) * seconds_per_crop)
        )
        cap_candidates.append(max(1, time_cap))
    recommended_cap = min(cap_candidates)
    if mean_cells > recommended_cap:
        print(
            f"[recommend] set MAX_CELLS_PER_PATCH <= {recommended_cap} "
            "to satisfy the current disk/time safety margins; rerun preflight "
            "because the smoke estimate is dataset- and GPU-specific."
        )
    if peak_bytes > free_disk:
        raise RuntimeError(
            "Estimated peak extraction storage exceeds free Kaggle disk"
        )
    if (
        args.max_estimated_hours is not None
        and estimated_hours > args.max_estimated_hours
    ):
        detail = (
            f" ({gpu_hours:.1f} GPU-h over {args.shards} GPUs)"
            if args.shards > 1 else ""
        )
        raise RuntimeError(
            f"Estimated extraction time {estimated_hours:.1f} h{detail} exceeds "
            f"the configured safety limit {args.max_estimated_hours:.1f} h. "
            "Reduce max_cells_per_patch, raise the limit knowingly, or use a "
            "longer-running GPU environment. Note the estimate is a warmed "
            "single-batch pilot and has measured 1.3-1.5x pessimistic against "
            "real runs."
        )
    if detected == 0:
        print(
            "[warning] No nucleus in the smoke batch. Runtime is valid, but "
            "inspect more QC patches before trusting the experiment."
        )
    print("[PREFLIGHT PASS] exact CellViT + postprocess + DINO path is runnable")


if __name__ == "__main__":
    main()
