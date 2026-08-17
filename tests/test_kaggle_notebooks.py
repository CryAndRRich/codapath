import ast
import json
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    PROJECT / "scripts" / "extract_nucleus_features.ipynb",
    PROJECT / "scripts" / "extract_features.ipynb",
    PROJECT / "scripts" / "run_al.ipynb",
    PROJECT / "scripts" / "evaluate.ipynb",
]


def _source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def test_kaggle_notebooks_are_valid_python_and_pin_experiment_branch():
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        all_source = "\n".join(_source(cell) for cell in notebook["cells"])
        assert 'REPO_BRANCH = "namhai"' in all_source or (
            "REPO_BRANCH = 'namhai'" in all_source
        )
        assert "/kaggle/input" in all_source
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = _source(cell)
            python_source = "\n".join(
                line for line in source.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            ast.parse(python_source, filename=f"{path.name}:cell-{index}")


def test_nucleus_notebook_runs_exact_preflight_before_full_extraction():
    path = PROJECT / "scripts" / "extract_nucleus_features.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [_source(cell) for cell in notebook["cells"]]
    preflight_index = next(
        i for i, source in enumerate(cells)
        if "preflight_nucleus_kaggle.py" in source
    )
    extraction_index = next(
        i for i, source in enumerate(cells)
        if "extract_nucleus_features.py" in source
    )
    assert preflight_index < extraction_index
    assert "--dino_crop_batch_size" in cells[preflight_index]
    assert "--cellvit_batch_size" in cells[preflight_index]
    assert "--input_mpp" in cells[preflight_index]
    assert "--model_mpp" in cells[preflight_index]
    assert "--magnification" in cells[preflight_index]
    assert "--smoke_samples', str(SMOKE_SAMPLES)" in cells[preflight_index]
    assert "--max_estimated_hours" in cells[preflight_index]
    assert "--input_mpp" in cells[extraction_index]
    assert "--model_mpp" in cells[extraction_index]
    assert "--magnification" in cells[extraction_index]


def test_run_notebook_matches_controlled_budget_protocol():
    config = yaml.safe_load(
        (PROJECT / "config" / "config.yaml").read_text(encoding="utf-8")
    )
    assert config["cumulative_budget"] == [25, 50, 75, 100, 125, 150, 175, 200]
    notebook = json.loads(
        (PROJECT / "scripts" / "run_al.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(_source(cell) for cell in notebook["cells"])
    assert 'SAMPLER_NAME = "nucleus_coverage"' in source
    assert "No nucleus cache found" in source
    assert "OUTPUT_DIR = \"/kaggle/working/checkpoints\"" in source
    # All three coverage feature spaces run back to back in one session.
    for coverage_source in ("dino", "cellvit", "concat"):
        assert f'{{"coverage_source": "{coverage_source}"}}' in source
    assert "SAMPLER_VARIANTS" in source


def test_run_notebook_loads_nucleus_cache_for_graph_deuce_too():
    """graph_deuce reads the CellViT nucleus cache exactly like nucleus_al/
    nucleus_coverage (EXPERIMENT.md Hướng 3 mục 7.1/7.4) — if this condition
    in run_al.ipynb ever drops "graph_deuce" again, NUCLEUS_FEATURE_DIR stays
    an unresolved placeholder and load_nucleus_cache fails deep inside main()
    instead of with a clear assert here."""
    notebook = json.loads(
        (PROJECT / "scripts" / "run_al.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(_source(cell) for cell in notebook["cells"])
    assert 'SAMPLER_NAME in ("nucleus_al", "nucleus_coverage", "graph_deuce")' in source
    assert '"graph_deuce"' in source  # SAMPLER_NAME comment / variant docs mention it


def test_run_notebook_resolves_caches_instead_of_hardcoding_paths():
    """Publishing /kaggle/working/<name> as a Kaggle Dataset remounts it as
    .../<name>/<name>, so a hard-coded cache path breaks with a bare
    AssertionError. Both caches must be located by search, and a DINOv2 cache
    miss must fall back to a writable directory: re-extracting into a read-only
    /kaggle/input only fails after the full extraction has already run."""
    notebook = json.loads(
        (PROJECT / "scripts" / "run_al.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(_source(cell) for cell in notebook["cells"])
    assert 'FEATURE_DIR = "/kaggle/input' in source
    assert "def dir_containing(" in source
    assert "dir_containing(f\"{DATASET}_seed{SEED}/manifest.json\"" in source
    assert 'startswith("/kaggle/input")' in source
    assert "will extract this session" in source
