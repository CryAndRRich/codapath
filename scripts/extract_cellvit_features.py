"""Extract aligned CellViT cell tokens and masked-crop DINOv2 features.

Run this script in an environment containing the official CellViT++ package.
The resulting cache can be mounted read-only for normal active-learning runs.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from data.identity import sample_order_fingerprint
from data.loaders import RawRGBDataset, get_data_loaders, get_sample_ids
from features.cellvit.cache import save_cellvit_cache, sha256_file
from features.cellvit.extractor import CellViTPatchExtractor
from features.cellvit.crops import DINOCellCropEncoder, masked_nucleus_crop


class _PersistentBuildDirectory:
    """Keep extraction shards after failure; remove them only after success."""

    def __init__(self, path: str, overwrite: bool) -> None:
        self.path = path
        self.overwrite = overwrite

    def __enter__(self) -> str:
        if self.overwrite and os.path.isdir(self.path):
            shutil.rmtree(self.path)
        os.makedirs(self.path, exist_ok=True)
        return self.path

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False


def _save_qc(path: str, rgb: np.ndarray, instance_map: np.ndarray) -> None:
    boundary = np.zeros_like(instance_map, dtype=bool)
    boundary[1:] |= instance_map[1:] != instance_map[:-1]
    boundary[:, 1:] |= instance_map[:, 1:] != instance_map[:, :-1]
    boundary &= instance_map > 0
    overlay = rgb.copy()
    overlay[boundary] = np.asarray([255, 0, 0], dtype=np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(overlay).save(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--data_path", default=None,
        help="Override datasets.<name>.path (required for mounted Kaggle inputs).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--input_mpp", type=float, default=None)
    parser.add_argument("--model_mpp", type=float, default=None)
    parser.add_argument("--magnification", type=int, choices=[20, 40], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--vit_name", default=None,
        help="Hugging Face model ID or local DINOv2 directory.",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--dino_crop_batch_size", type=int, default=None)
    parser.add_argument("--max_cells_per_patch", type=int, default=None)
    parser.add_argument("--qc_count", type=int, default=16)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacing an existing cache for this dataset/seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.dataset not in config["datasets"]:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    extraction = config["cellvit_extraction"]
    dataset_config = extraction.get("datasets", {}).get(args.dataset, {})

    checkpoint = (
        args.checkpoint
        or dataset_config.get("checkpoint")
        or extraction.get("checkpoint")
    )
    cache_root = args.cache_dir or config.get(
        "cellvit_cache_dir", "cellvit_features"
    )
    cache_path = os.path.join(cache_root, f"{args.dataset}_seed{args.seed}")
    manifest_path = os.path.join(cache_path, "manifest.json")
    if os.path.exists(manifest_path) and not args.overwrite:
        raise FileExistsError(
            f"Cache already exists: {cache_path}. Use --overwrite only after "
            "checking its manifest and extraction settings."
        )
    input_mpp = args.input_mpp or dataset_config.get(
        "input_mpp", extraction.get("input_mpp")
    )
    model_mpp = args.model_mpp or dataset_config.get(
        "model_mpp", extraction.get("model_mpp")
    )
    magnification = args.magnification or dataset_config.get(
        "magnification", extraction.get("magnification")
    )
    if checkpoint is None:
        raise ValueError(
            "A CellViT256 checkpoint must be supplied via --checkpoint or the "
            "dataset-specific cellvit_extraction config."
        )
    if input_mpp is None or model_mpp is None or magnification is None:
        raise ValueError(
            "input_mpp, model_mpp, and magnification must be explicitly configured"
        )
    if args.qc_count < 0:
        raise ValueError("qc_count must be non-negative")
    if args.max_cells_per_patch is not None and args.max_cells_per_patch < 1:
        raise ValueError("max_cells_per_patch must be positive")
    device = torch.device(args.device or config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no Kaggle GPU is attached")
    data_path = args.data_path or config["datasets"][args.dataset]["path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset path does not exist: {data_path}. On Kaggle pass "
            "--data_path /kaggle/input/... explicitly."
        )

    train_loader, _, _ = get_data_loaders(
        data_path, args.seed, verbose=True
    )
    raw_dataset = RawRGBDataset(train_loader.dataset)
    sample_ids = get_sample_ids(train_loader.dataset)
    sample_fingerprint = sample_order_fingerprint(sample_ids)
    checkpoint_hash = sha256_file(checkpoint)

    max_cells_per_patch = (
        args.max_cells_per_patch
        if args.max_cells_per_patch is not None
        else extraction.get("max_cells_per_patch")
    )
    cellvit = CellViTPatchExtractor(
        checkpoint, device,
        input_mpp=input_mpp,
        model_mpp=model_mpp,
        magnification=magnification,
        min_cell_area=extraction.get("min_cell_area", 10),
        max_cells_per_patch=max_cells_per_patch,
    )
    vit_name = args.vit_name or config.get("models", {}).get(
        "vit", "facebook/dinov2-base"
    )
    crop_dino = DINOCellCropEncoder(
        vit_name,
        device,
        batch_size=(
            args.dino_crop_batch_size
            or extraction.get("dino_crop_batch_size", 32)
        ),
    )

    offsets = [0]
    batch_size = int(args.batch_size or extraction.get("batch_size", 2))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    qc_dir = os.path.join(cache_path, "qc")

    def concatenate(values, dimension, dtype=np.float32):
        nonempty = [value for value in values if len(value)]
        if nonempty:
            return np.concatenate(nonempty, axis=0).astype(dtype, copy=False)
        shape = (0,) if dimension == 1 else (0, dimension)
        return np.empty(shape, dtype=dtype)

    os.makedirs(cache_root, exist_ok=True)
    persistent_build_path = os.path.join(cache_path, ".build")
    with _PersistentBuildDirectory(
        persistent_build_path, overwrite=args.overwrite
    ) as build_dir:
        build_state = {
            "dataset": args.dataset,
            "seed": args.seed,
            "sample_fingerprint": sample_fingerprint,
            "checkpoint_sha256": checkpoint_hash,
            "input_mpp": float(input_mpp),
            "model_mpp": float(model_mpp),
            "magnification": int(magnification),
            "dino_backbone": vit_name,
            "batch_size": batch_size,
            "crop_padding": float(extraction.get("crop_padding", 0.25)),
            "min_cell_area": int(extraction.get("min_cell_area", 10)),
            "max_cells_per_patch": max_cells_per_patch,
        }
        state_path = os.path.join(build_dir, "state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                previous_state = json.load(f)
            if previous_state != build_state:
                raise RuntimeError(
                    "Partial extraction settings differ from this run. Inspect "
                    f"{state_path}, then rerun with --overwrite to restart."
                )
            print(f"[nucleus] Resuming partial extraction from {build_dir}")
        else:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(build_state, f, indent=2, sort_keys=True)

        shard_paths = []
        for start in range(0, len(raw_dataset), batch_size):
            stop = min(start + batch_size, len(raw_dataset))
            shard_path = os.path.join(build_dir, f"{start:09d}.npz")
            if os.path.exists(shard_path):
                with np.load(shard_path, allow_pickle=False) as shard:
                    counts = shard["counts"].astype(np.int64).tolist()
                    if len(counts) != stop - start:
                        raise RuntimeError(f"Invalid resumed shard: {shard_path}")
                    for count in counts:
                        offsets.append(offsets[-1] + int(count))
                shard_paths.append(shard_path)
                print(
                    f"[nucleus] resumed {stop}/{len(raw_dataset)} patches | "
                    f"cells={offsets[-1]}"
                )
                continue

            images = [raw_dataset[idx][0] for idx in range(start, stop)]
            patches = cellvit.extract_batch(images)
            crops = []
            batch_cellvit = []
            batch_confidence = []
            batch_bboxes = []
            for local_idx, patch in enumerate(patches):
                if start + local_idx < args.qc_count:
                    _save_qc(
                        os.path.join(qc_dir, f"{start + local_idx:06d}.png"),
                        patch.rgb, patch.instance_map,
                    )
                for instance_id, bbox in zip(patch.instance_ids, patch.bboxes):
                    crops.append(masked_nucleus_crop(
                        patch.rgb, patch.instance_map, int(instance_id), bbox,
                        padding=float(extraction.get("crop_padding", 0.25)),
                    ))
                batch_cellvit.append(patch.embeddings)
                batch_confidence.append(patch.confidence)
                batch_bboxes.append(patch.bboxes)
                offsets.append(offsets[-1] + len(patch.instance_ids))

            encoded_crops = crop_dino.encode(crops)
            batch_cells = sum(len(patch.instance_ids) for patch in patches)
            if len(encoded_crops) != batch_cells:
                raise RuntimeError("Crop-DINO rows do not align with CellViT cells")
            temporary_shard_path = f"{shard_path}.tmp.npz"
            np.savez(
                temporary_shard_path,
                cellvit=concatenate(
                    batch_cellvit, cellvit.embedding_dim, np.float16
                ),
                crop_dino=encoded_crops.astype(np.float16, copy=False),
                confidence=concatenate(
                    batch_confidence, 1, np.float32
                ).reshape(-1),
                bboxes=concatenate(batch_bboxes, 4, np.int32),
                counts=np.asarray(
                    [len(patch.instance_ids) for patch in patches],
                    dtype=np.int32,
                ),
            )
            os.replace(temporary_shard_path, shard_path)
            shard_paths.append(shard_path)
            print(
                f"[nucleus] {stop}/{len(raw_dataset)} patches | "
                f"cells={offsets[-1]}"
            )

        total_cells = offsets[-1]
        cellvit_memmap = np.lib.format.open_memmap(
            os.path.join(build_dir, "cellvit.npy"), mode="w+", dtype=np.float16,
            shape=(total_cells, cellvit.embedding_dim),
        )
        crop_dino_memmap = np.lib.format.open_memmap(
            os.path.join(build_dir, "crop_dino.npy"), mode="w+", dtype=np.float16,
            shape=(total_cells, crop_dino.feature_dim),
        )
        confidence_memmap = np.lib.format.open_memmap(
            os.path.join(build_dir, "confidence.npy"), mode="w+", dtype=np.float32,
            shape=(total_cells,),
        )
        bboxes_memmap = np.lib.format.open_memmap(
            os.path.join(build_dir, "bboxes.npy"), mode="w+", dtype=np.int32,
            shape=(total_cells, 4),
        )
        cursor = 0
        for shard_path in shard_paths:
            with np.load(shard_path, allow_pickle=False) as shard:
                count = len(shard["confidence"])
                if not (
                    len(shard["cellvit"]) == len(shard["crop_dino"])
                    == len(shard["bboxes"]) == count
                ):
                    raise RuntimeError(f"Misaligned extraction shard: {shard_path}")
                end = cursor + count
                cellvit_memmap[cursor:end] = shard["cellvit"]
                crop_dino_memmap[cursor:end] = shard["crop_dino"]
                confidence_memmap[cursor:end] = shard["confidence"]
                bboxes_memmap[cursor:end] = shard["bboxes"]
                cursor = end
        if cursor != total_cells:
            raise RuntimeError("Extraction shards do not match ragged offsets")
        for array in (
            cellvit_memmap, crop_dino_memmap, confidence_memmap, bboxes_memmap
        ):
            array.flush()

        save_cellvit_cache(
            cache_path,
            offsets=np.asarray(offsets, dtype=np.int64),
            confidence=confidence_memmap,
            sample_ids=sample_ids,
            cellvit_embeddings=cellvit_memmap,
            cell_dino_features=crop_dino_memmap,
            bboxes=bboxes_memmap,
            manifest={
                "dataset": args.dataset,
                "seed": args.seed,
                "cellvit_architecture": "CellViT256",
                "checkpoint_path": os.path.basename(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "input_mpp": float(input_mpp),
                "model_mpp": float(model_mpp),
                "magnification": int(magnification),
                "dino_backbone": vit_name,
                "crop_padding": float(extraction.get("crop_padding", 0.25)),
                "max_cells_per_patch": max_cells_per_patch,
                "feature_dtype": "float16",
            },
        )
    shutil.rmtree(persistent_build_path)
    print(f"[nucleus] Saved cache → {cache_path}")


if __name__ == "__main__":
    main()
