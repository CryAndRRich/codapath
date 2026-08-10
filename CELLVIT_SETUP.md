# CellViT Extraction Environment

Nucleus extraction is separate from normal active-learning runs. The adapter is locked to the official `cellvit==1.0.9` wheel, whose model/checkpoint contract matches the [CellViT++ source](https://github.com/TIO-IKIM/CellViT-plus-plus).

## Kaggle

Do not install CellViT's complete dependency tree over Kaggle's existing PyTorch stack; it pins NumPy, OpenCV, Pydantic, Ray, and WSI dependencies that this patch-level adapter never imports. Use:

```bash
pip install --no-deps --require-hashes -r requirements-kaggle-cellvit.txt
pip install "transformers>=4.27" "PyYAML>=6" "einops>=0.6.1"
```

Kaggle normally already provides PyTorch, torchvision, NumPy, pandas, SciPy, scikit-image, Numba and OpenCV. `scripts/preflight_nucleus_kaggle.py` imports each one and runs the real CellViT postprocessor, so an ABI problem is found before full extraction.

Use [extract_nucleus_features.ipynb](scripts/extract_nucleus_features.ipynb). Mount the matching CellViT256 x20/x40 checkpoint under `/kaggle/input`, write caches only under `/kaggle/working`, and do not continue unless the preflight prints `[PREFLIGHT PASS]`.

The official documentation recommends at least 24 GB VRAM for the complete WSI pipeline. This project uses only CellViT256 patch inference and defaults conservatively to CellViT batch 2 plus DINO crop batch 32 for Kaggle T4/P100. If preflight OOMs, lower both to 1/8.

Normal `run.py` jobs consume the portable cache and do not import CellViT.
