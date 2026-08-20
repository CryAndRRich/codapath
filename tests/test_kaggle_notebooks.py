import ast
import json
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    PROJECT / "scripts" / "extract_nucleus_features.ipynb",
    PROJECT / "scripts" / "extract_features.ipynb",
    PROJECT / "scripts" / "run_al.ipynb",
    PROJECT / "scripts" / "run_nucleus_coverage.ipynb",
    PROJECT / "scripts" / "evaluate.ipynb",
    PROJECT / "scripts" / "evaluate_nucleus_alignment.ipynb",
]


def _source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def test_kaggle_notebooks_are_valid_python_and_pin_experiment_branch():
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        all_source = "\n".join(_source(cell) for cell in notebook["cells"])
        assert 'REPO_BRANCH = "tiendung"' in all_source or (
            "REPO_BRANCH = 'tiendung'" in all_source
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
    assert 'SAMPLER_NAME = "nucleus_al"' in source
    assert "Missing nucleus cache" in source
    assert "OUTPUT_DIR = \"/kaggle/working/checkpoints\"" in source


def test_alignment_notebook_is_diagnostic_and_writes_to_working():
    notebook = json.loads(
        (PROJECT / "scripts" / "evaluate_nucleus_alignment.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(_source(cell) for cell in notebook["cells"])
    assert "evaluate_nucleus_alignment.py" in source
    assert "nucleus_cellvit_embedding_disagreement" in source
    assert "--probe_repeats" in source
    assert "--budgets" in source
    assert "/kaggle/working/nucleus_alignment_" in source
    assert "unselected training pool" in source


def test_nucleus_coverage_notebook_runs_controlled_feature_spaces():
    notebook = json.loads(
        (PROJECT / "scripts" / "run_nucleus_coverage.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(_source(cell) for cell in notebook["cells"])
    assert 'SAMPLER_NAME = "nucleus_coverage"' in source
    for coverage_source in ("dino", "cellvit", "concat"):
        assert f'{{"coverage_source": "{coverage_source}"}}' in source
    assert "No nucleus cache found" in source
    assert "not** the original UHerding UCoverage objective" in source
    assert 'OUTPUT_DIR = "/kaggle/working/checkpoints"' in source
