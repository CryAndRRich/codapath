# CellViT Extraction Environment

Nucleus extraction is separate from normal active-learning runs. The adapter is locked to the official `cellvit==1.0.9` wheel, whose model/checkpoint contract matches the [CellViT++ source](https://github.com/TIO-IKIM/CellViT-plus-plus).

## Kaggle

Do not install CellViT's complete dependency tree over Kaggle's existing PyTorch stack; it pins NumPy, OpenCV, Pydantic, Ray, and WSI dependencies that this patch-level adapter never imports. Use:

```bash
pip install --no-deps --require-hashes -r requirements-kaggle-cellvit.txt
pip install "transformers>=4.27" "PyYAML>=6" "einops>=0.6.1"
```

Kaggle normally already provides PyTorch, torchvision, NumPy, pandas, SciPy, scikit-image, Numba and OpenCV. `scripts/preflight_nucleus_kaggle.py` imports each one and runs the real CellViT postprocessor, so an ABI problem is found before full extraction.

Kaggle's Python 3.12 image cannot JIT-compile the mixed-dtype `stack_pred_maps` helper shipped in `cellvit==1.0.9`. The adapter bypasses only that four-channel `argmax + stack` helper with an equivalent NumPy float32 buffer. It still calls the official CellViT per-image HoVer/morphology/watershed post-processing unchanged; do not downgrade the whole Kaggle NumPy/PyTorch stack for this issue.

Use [extract_nucleus_features.ipynb](scripts/extract_nucleus_features.ipynb). The Kaggle notebooks clone and verify the `tiendung` branch. Mount the `nckh2026` dataset and matching CellViT256 x20/x40 checkpoint under `/kaggle/input`, write caches only under `/kaggle/working`, and do not continue unless the preflight prints `[PREFLIGHT PASS]`. The default PathMNIST mount is `/kaggle/input/nckh2026/pathmnist_224.npz`; edit the single configuration cell if Kaggle assigns a different slug.

Scale is guarded in code: an x20 checkpoint must use 20× and approximately 0.5 MPP; an x40 checkpoint must use 40× and approximately 0.25 MPP. Use a current official checkpoint—the CellViT authors explicitly warn that older checkpoints predate a corrected training release.

For PathMNIST (`input_mpp=0.5`), the preferred native-scale choice is an x20 checkpoint (`model_mpp=0.5`, `magnification=20`). If only the official x40 AMP checkpoint is available, set `model_mpp=0.25`, `magnification=40`; the adapter then upsamples by 2×. This is executable and scale-consistent, but interpolation cannot recreate detail missing from the 0.5-MPP source, so record the checkpoint choice as an experimental limitation.

The official documentation recommends at least 24 GB VRAM for the complete WSI pipeline. This project uses only CellViT256 patch inference and defaults conservatively to CellViT batch 2 plus DINO crop batch 32 for Kaggle T4/P100. Preflight now executes both configured batch sizes, so an OOM occurs before the full job. If it OOMs, lower both to 1/8.

After extraction succeeds, save `/kaggle/working/nucleus_features` as notebook output, publish/attach it as a Kaggle Dataset, and set `NUCLEUS_FEATURE_DIR` in `run_al.ipynb` to `/kaggle/input/<output-slug>/nucleus_features`. Active-learning checkpoints are written to `/kaggle/working/checkpoints`.

On a Kaggle T4, the uncapped PathMNIST pilot detected about 67.9 nuclei/patch and estimated 14.7 GiB final cache, 44.1 GiB resumable peak storage, and 30 hours for CellViT plus crop-DINO. The notebook therefore defaults to `MAX_CELLS_PER_PATCH=16`; keep this fixed across all four controlled variants. Preflight reports CellViT and crop-DINO time separately and recommends a smaller cap when needed. A cap changes the experimental protocol and must be recorded in results/manifests.

Normal `run.py` jobs consume the portable cache and do not import CellViT.

## Final-Learner Alignment Diagnostic

After the four controlled acquisition variants finish, run the cheap diagnostic below before extracting a separate nucleus test cache or implementing another sampler. It reuses the aligned train caches and saved selected indices, trains the same linear probe on each selected set, and evaluates on the unselected train pool. The output is explicitly diagnostic—not official test accuracy.

```bash
python scripts/evaluate_nucleus_alignment.py \
  --dataset pathmnist \
  --data_path /kaggle/input/datasets/cryandrrich/nckh2026/pathmnist_224.npz \
  --feature_cache_dir /kaggle/input/datasets/cryandrrich/nckh2026/features \
  --nucleus_cache_dir /kaggle/input/<nucleus-output-slug>/nucleus_features \
  --checkpoint_dir /kaggle/input/<al-output-slug>/checkpoints/pathmnist \
  --selection_run nucleus_cellvit_embedding_disagreement \
  --output /kaggle/working/nucleus_alignment_pathmnist_seed42.json
```

The representations are DINO original/normalized, cell mean, DINO+cell mean, DINO+cell moments, and DINO+conditional-cell residual. Cell moments add weighted standard deviation, cell count, and detection confidence before an unlabeled PCA. Three probe initializations are averaged by default. Omit `--selection_run` to auto-select disagreement; pass `--all_runs` only when all four acquisition variants should be audited.
