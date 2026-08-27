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
from features.cellvit.segmaps import (
    InstanceMapWriter, merge_blobs, overlay_boundaries,
)


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(overlay_boundaries(rgb, instance_map)).save(path)


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
        "--save_instance_maps", action="store_true",
        help="Also store CellViT's per-pixel instance map for every patch, "
             "zlib-compressed. This is the raw segmentation output, kept for "
             "visualisation; no sampler reads it. Measured ~230-300x compression "
             "on capped patches (~0.15 GiB per 100k), because a label image is "
             "mostly background and piecewise constant.",
    )
    parser.add_argument(
        "--mmap_cache_dir", default=None,
        help="Directory holding the .npz re-exported as memory-mappable .npy "
             "files. Required for a multi-GPU run on a large .npz: an eager read "
             "costs ~15 GiB per process. Export it once in the parent (numpy "
             "ignores mmap_mode for .npz, so only standalone .npy can be mapped).",
    )
    parser.add_argument(
        "--shard_index", type=int, default=None,
        help="Extract only this shard of the patch range, then stop before "
             "assembly. One process per GPU on a Kaggle T4 x2 session.",
    )
    parser.add_argument(
        "--shard_count", type=int, default=1,
        help="Total number of shards the patch range is split into.",
    )
    parser.add_argument(
        "--assemble_only", action="store_true",
        help="Skip extraction and merge existing shards into the final cache. "
             "Run once, after every --shard_index process has finished.",
    )
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
    if args.shard_count < 1:
        raise ValueError("shard_count must be positive")
    if args.shard_index is not None and not 0 <= args.shard_index < args.shard_count:
        raise ValueError(
            f"shard_index {args.shard_index} outside [0, {args.shard_count})"
        )
    if args.assemble_only and args.shard_index is not None:
        raise ValueError("--assemble_only and --shard_index are mutually exclusive")
    if args.assemble_only and args.overwrite:
        # --overwrite empties the .build directory on entry, which is exactly the
        # shard output --assemble_only exists to merge. Together they would
        # delete hours of GPU work and then fail on the first missing batch.
        raise ValueError(
            "--assemble_only with --overwrite would delete the shards being "
            "assembled. Drop --overwrite, or delete the cache directory by hand "
            "to restart extraction from scratch."
        )
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
        data_path, args.seed, verbose=True, mmap_cache_dir=args.mmap_cache_dir
    )
    if args.mmap_cache_dir:
        print(f"[nucleus] mmap={getattr(train_loader.dataset, 'mmap', None)}")
    raw_dataset = RawRGBDataset(train_loader.dataset)
    sample_ids = get_sample_ids(train_loader.dataset)
    sample_fingerprint = sample_order_fingerprint(sample_ids)
    checkpoint_hash = sha256_file(checkpoint)

    max_cells_per_patch = (
        args.max_cells_per_patch
        if args.max_cells_per_patch is not None
        else extraction.get("max_cells_per_patch")
    )
    vit_name = args.vit_name or config.get("models", {}).get(
        "vit", "facebook/dinov2-base"
    )
    # Assembly only concatenates existing batch files, so it must not load two
    # networks onto the GPU: that is minutes of download and VRAM for nothing,
    # and it would make the merge pass need a GPU it does not use. The feature
    # dimensions it needs for the output memmaps are read off the batch files.
    if args.assemble_only:
        cellvit = None
        crop_dino = None
    else:
        cellvit = CellViTPatchExtractor(
            checkpoint, device,
            input_mpp=input_mpp,
            model_mpp=model_mpp,
            magnification=magnification,
            min_cell_area=extraction.get("min_cell_area", 10),
            max_cells_per_patch=max_cells_per_patch,
        )
        crop_dino = DINOCellCropEncoder(
            vit_name,
            device,
            batch_size=(
                args.dino_crop_batch_size
                or extraction.get("dino_crop_batch_size", 32)
            ),
        )

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
            # A resumed build must not mix batches that stored an instance map
            # with batches that did not: the blob would be missing patches and
            # its offsets index would silently misalign with the patch order.
            "save_instance_maps": bool(args.save_instance_maps),
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

        # Every batch start offset, in the exact order a serial run visits them.
        # Sharding assigns whole batches, never splitting one, so each batch has
        # the same members as in a serial run: CellViT pads a batch to the
        # largest image in it, so a batch with different members is a different
        # forward pass on a variable-size dataset.
        all_starts = list(range(0, len(raw_dataset), batch_size))
        if args.assemble_only:
            my_starts = []
            print("[nucleus] --assemble_only: merging existing batch files")
        elif args.shard_index is None:
            my_starts = all_starts
        else:
            # Contiguous blocks of batches. Either split covers the same
            # batches in total; contiguity is chosen because it makes each
            # worker's progress log a single advancing range, and because a
            # resumed run with the same shard_count reuses exactly the npz
            # files that worker already wrote.
            per_shard, remainder = divmod(len(all_starts), args.shard_count)
            first = args.shard_index * per_shard + min(args.shard_index, remainder)
            count = per_shard + (1 if args.shard_index < remainder else 0)
            my_starts = all_starts[first:first + count]
            print(
                f"[nucleus] shard {args.shard_index}/{args.shard_count}: "
                f"{len(my_starts)} of {len(all_starts)} batches, patches "
                f"[{my_starts[0] if my_starts else 0}:"
                f"{min(my_starts[-1] + batch_size, len(raw_dataset)) if my_starts else 0})"
            )

        # `offsets` is a running prefix sum over the WHOLE dataset, so it can
        # only be built by a process that sees every batch. A shard worker
        # writes its own npz files and leaves the offsets to the assembly pass.
        shard_paths = []
        for start in my_starts:
            stop = min(start + batch_size, len(raw_dataset))
            shard_path = os.path.join(build_dir, f"{start:09d}.npz")
            if os.path.exists(shard_path):
                with np.load(shard_path, allow_pickle=False) as shard:
                    if len(shard["counts"]) != stop - start:
                        raise RuntimeError(f"Invalid resumed shard: {shard_path}")
                shard_paths.append(shard_path)
                print(f"[nucleus] resumed {stop}/{len(raw_dataset)} patches")
                continue

            images = [raw_dataset[idx][0] for idx in range(start, stop)]
            patches = cellvit.extract_batch(images)
            # One blob per batch, beside its npz, so resume and shard assembly
            # work on exactly the same unit as the features do.
            if args.save_instance_maps:
                map_path = os.path.join(build_dir, f"{start:09d}.maps")
                with InstanceMapWriter(map_path + ".tmp") as writer:
                    for patch in patches:
                        writer.append(patch.instance_map)
                    map_offsets, map_shapes = writer.close()
                np.save(map_path + ".idx.tmp.npy", np.concatenate(
                    [map_offsets[:, None],
                     np.vstack([map_shapes, [[0, 0]]])], axis=1
                ))
                os.replace(map_path + ".tmp", map_path)
                os.replace(map_path + ".idx.tmp.npy", map_path + ".idx.npy")
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
                f"cells={int(np.sum([len(p.instance_ids) for p in patches]))} in batch"
            )

        if args.shard_index is not None:
            print(
                f"[nucleus] shard {args.shard_index}/{args.shard_count} wrote "
                f"{len(shard_paths)} batch files to {build_dir}. Run this script "
                "again with --assemble_only once every shard has finished."
            )
            return

        # Assembly. Rebuild `offsets` by reading every batch file in patch order
        # rather than accumulating during extraction: with several processes
        # sharding the range, no single process saw every batch, and a resumed
        # run sees only the batches it recomputed. The files on disk are the one
        # complete record.
        expected_starts = list(range(0, len(raw_dataset), batch_size))
        offsets = [0]
        ordered_shard_paths = []
        for start in expected_starts:
            stop = min(start + batch_size, len(raw_dataset))
            shard_path = os.path.join(build_dir, f"{start:09d}.npz")
            if not os.path.exists(shard_path):
                raise FileNotFoundError(
                    f"Missing batch file for patches [{start}:{stop}) at "
                    f"{shard_path}. Assembly needs every batch; re-run the "
                    "shard workers to fill the gap before assembling."
                )
            with np.load(shard_path, allow_pickle=False) as shard:
                counts = shard["counts"].astype(np.int64).tolist()
            if len(counts) != stop - start:
                raise RuntimeError(
                    f"Batch file {shard_path} covers {len(counts)} patches, "
                    f"expected {stop - start}. It was written with a different "
                    "batch_size; restart with --overwrite."
                )
            for count in counts:
                offsets.append(offsets[-1] + int(count))
            ordered_shard_paths.append(shard_path)
        if len(offsets) != len(raw_dataset) + 1:
            raise RuntimeError(
                f"Rebuilt offsets cover {len(offsets) - 1} patches, expected "
                f"{len(raw_dataset)}"
            )
        shard_paths = ordered_shard_paths
        print(
            f"[nucleus] assembling {len(shard_paths)} batch files | "
            f"patches={len(raw_dataset)} cells={offsets[-1]}"
        )

        total_cells = offsets[-1]
        # Read the feature widths off the first batch file that holds any cell.
        # Under --assemble_only the models are not loaded, so their `.dim`
        # attributes are unavailable; when they ARE loaded this is the same
        # number, cross-checked below.
        cellvit_dim = None
        crop_dino_dim = None
        for shard_path in shard_paths:
            with np.load(shard_path, allow_pickle=False) as shard:
                if len(shard["confidence"]):
                    cellvit_dim = int(shard["cellvit"].shape[1])
                    crop_dino_dim = int(shard["crop_dino"].shape[1])
                    break
        if cellvit_dim is None:
            raise RuntimeError(
                "No batch file contains a single cell. CellViT detected nothing "
                "anywhere, which is a scale/checkpoint problem, not a cache one."
            )
        if cellvit is not None and cellvit_dim != cellvit.embedding_dim:
            raise RuntimeError(
                f"Batch files carry {cellvit_dim}-d CellViT tokens but the loaded "
                f"model produces {cellvit.embedding_dim}-d"
            )
        if crop_dino is not None and crop_dino_dim != crop_dino.feature_dim:
            raise RuntimeError(
                f"Batch files carry {crop_dino_dim}-d crop-DINO features but the "
                f"loaded model produces {crop_dino.feature_dim}-d"
            )
        cellvit_memmap = np.lib.format.open_memmap(
            os.path.join(build_dir, "cellvit.npy"), mode="w+", dtype=np.float16,
            shape=(total_cells, cellvit_dim),
        )
        crop_dino_memmap = np.lib.format.open_memmap(
            os.path.join(build_dir, "crop_dino.npy"), mode="w+", dtype=np.float16,
            shape=(total_cells, crop_dino_dim),
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

        # Merge the per-batch instance-map blobs in patch order. Done here, in
        # the same pass that rebuilt `offsets`, so the sidecar's patch order is
        # the cache's patch order by construction rather than by assumption.
        if args.save_instance_maps:
            os.makedirs(cache_path, exist_ok=True)
            blob_paths = []
            blob_indexes = []
            for shard_path in shard_paths:
                map_path = shard_path[: -len(".npz")] + ".maps"
                index_path = map_path + ".idx.npy"
                if not (os.path.exists(map_path) and os.path.exists(index_path)):
                    raise FileNotFoundError(
                        f"--save_instance_maps was requested but {map_path} is "
                        "missing. It was extracted without the flag; restart "
                        "with --overwrite to rebuild."
                    )
                index = np.load(index_path)
                blob_paths.append(map_path)
                blob_indexes.append((index[:, 0], index[:-1, 1:3]))
            merge_blobs(cache_path, blob_paths, blob_indexes, len(sample_ids))
            sidecar_bytes = os.path.getsize(
                os.path.join(cache_path, "instance_maps.bin")
            )
            print(
                f"[nucleus] instance maps: {sidecar_bytes / 2**20:.1f} MiB for "
                f"{len(sample_ids)} patches "
                f"({sidecar_bytes / max(1, len(sample_ids)):.0f} B/patch)"
            )

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
                "has_instance_maps": bool(args.save_instance_maps),
            },
        )
    shutil.rmtree(persistent_build_path)
    print(f"[nucleus] Saved cache → {cache_path}")


if __name__ == "__main__":
    main()
