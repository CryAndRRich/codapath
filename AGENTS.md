# Repository Guidelines

## Project Structure & Module Organization

This repository contains CODAPath, a Python research codebase for active learning in pathology. Core runtime modules live at the repository root: `run.py` is the training entry point, `model.py` defines the model and training helpers, `sampling.py` contains active-learning samplers, `load_data.py` builds datasets/loaders, `evaluate.py` provides metrics and visualization utilities, and `contrastive.py` contains contrastive-learning logic. Configuration is centralized in `config/config.yaml`. Experimental variants live in `ablations/`, notebooks in `scripts/`, and pretrained or generated `.pth` files in `weights/<sampler>/<dataset>/`. Dataset paths referenced by config should resolve under `data/`, which is not committed.

## Build, Test, and Development Commands

Create an isolated environment before installing dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Run the default experiment from `config/config.yaml`:

```bash
python run.py --config config/config.yaml
```

Override common options from the command line:

```bash
python run.py --dataset pathmnist --sampler_name codapath --device cuda --num_epochs 25
```

Use `--device cpu` for local smoke checks when CUDA is unavailable. Open `scripts/codapath.ipynb` or `scripts/evaluate.ipynb` for exploratory runs and plotting.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation, type hints on public functions, and clear snake_case names for variables, functions, and module files. Keep dataset keys, sampler names, and weight paths lowercase, for example `pathmnist`, `skintissue`, and `weights/codapath/pathmnist/...`. Prefer config-driven changes over hard-coded experiment settings. Keep imports grouped as standard library, third-party, then local modules.

## Testing Guidelines

There is currently no automated test suite. Before submitting changes, run a small CPU or CUDA smoke test with a known config and verify that data loading, sampler selection, checkpoint creation, and metric printing still work. For new reusable logic, add focused tests under a new `tests/` directory using `pytest`, with files named `test_<module>.py`.

## Commit & Pull Request Guidelines

Recent history uses short imperative commit subjects such as `fix cpu error` and `Add plot func`. Keep commits concise but descriptive, ideally naming the affected behavior, e.g. `fix pathmnist loader split`. Pull requests should include the purpose, key config or command used for validation, affected datasets/samplers, and screenshots or generated plot paths when visualization output changes.

## Security & Configuration Tips

Do not commit raw datasets, secrets, local environment files, or large new checkpoints unless they are intentional project artifacts. Keep machine-specific paths in `config/config.yaml` or local overrides, and document any required external dataset layout in the PR.
