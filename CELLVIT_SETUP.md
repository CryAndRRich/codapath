# CellViT Extraction Environment

Nucleus extraction is separate from normal active-learning runs. The adapter is locked to the official `cellvit==1.0.9` wheel, whose model/checkpoint contract matches the [CellViT++ source](https://github.com/TIO-IKIM/CellViT-plus-plus).

## Kaggle

Do not install CellViT's complete dependency tree over Kaggle's existing PyTorch stack; it pins NumPy, OpenCV, Pydantic, Ray, and WSI dependencies that this patch-level adapter never imports. Use:

```bash
pip install --no-deps --require-hashes -r requirements-kaggle-cellvit.txt
pip install "transformers>=4.27" "PyYAML>=6" "einops>=0.6.1"
```

Kaggle normally already provides PyTorch, torchvision, NumPy, pandas, SciPy, scikit-image, Numba and OpenCV. `scripts/preflight_nucleus_kaggle.py` imports each one and runs the real CellViT postprocessor, so an ABI problem is found before full extraction.

Use [extract_nucleus_features.ipynb](scripts/extract_nucleus_features.ipynb). The Kaggle notebooks clone and verify the `tiendung` branch. Mount the `nckh2026` dataset and matching CellViT256 x20/x40 checkpoint under `/kaggle/input`, write caches only under `/kaggle/working`, and do not continue unless the preflight prints `[PREFLIGHT PASS]`. The default PathMNIST mount is `/kaggle/input/nckh2026/pathmnist_224.npz`; edit the single configuration cell if Kaggle assigns a different slug.

Scale is guarded in code: an x20 checkpoint must use 20× and approximately 0.5 MPP; an x40 checkpoint must use 40× and approximately 0.25 MPP. Use a current official checkpoint—the CellViT authors explicitly warn that older checkpoints predate a corrected training release.

For PathMNIST (`input_mpp=0.5`), the preferred native-scale choice is an x20 checkpoint (`model_mpp=0.5`, `magnification=20`). If only the official x40 AMP checkpoint is available, set `model_mpp=0.25`, `magnification=40`; the adapter then upsamples by 2×. This is executable and scale-consistent, but interpolation cannot recreate detail missing from the 0.5-MPP source, so record the checkpoint choice as an experimental limitation.

The official documentation recommends at least 24 GB VRAM for the complete WSI pipeline. This project uses only CellViT256 patch inference and defaults conservatively to CellViT batch 2 plus DINO crop batch 32 for Kaggle T4/P100. Preflight now executes both configured batch sizes, so an OOM occurs before the full job. If it OOMs, lower both to 1/8.

After extraction succeeds, save `/kaggle/working/nucleus_features` as notebook output, publish/attach it as a Kaggle Dataset, and set `NUCLEUS_FEATURE_DIR` in `run_al.ipynb` to `/kaggle/input/<output-slug>/nucleus_features`. Active-learning checkpoints are written to `/kaggle/working/checkpoints`.

Normal `run.py` jobs consume the portable cache and do not import CellViT.
