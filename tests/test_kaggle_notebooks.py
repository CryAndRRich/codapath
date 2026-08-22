import ast
import json
from pathlib import Path

import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT / "notebooks"
EXPECTED_NOTEBOOKS = {
    "extract_visual_features.ipynb",
    "extract_nucleus_features.ipynb",
    "run_al_sampler.ipynb",
    "evaluate_al_sampler.ipynb",
}


def _source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _all_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(_source(cell) for cell in notebook["cells"])


def test_notebook_set_is_exactly_the_four_supported_entry_points():
    """Extra notebooks drift out of sync with the code and silently rot; the
    project deliberately keeps one per stage of the pipeline."""
    assert {p.name for p in NOTEBOOK_DIR.glob("*.ipynb")} == EXPECTED_NOTEBOOKS


@pytest.mark.parametrize("name", sorted(EXPECTED_NOTEBOOKS))
def test_notebook_cells_are_valid_python_and_pin_the_branch(name):
    path = NOTEBOOK_DIR / name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = _all_source(path)
    assert 'REPO_BRANCH = "namhai"' in source
    assert "/kaggle/input" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        body = "\n".join(
            line for line in _source(cell).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        ast.parse(body, filename=f"{name}:cell-{index}")


@pytest.mark.parametrize("name", sorted(EXPECTED_NOTEBOOKS - {"evaluate_al_sampler.ipynb"}))
def test_producing_notebooks_zip_their_output(name):
    """Kaggle output is only convenient to retrieve as a single archive, and the
    console scrollback is gone once the session ends."""
    assert "make_archive" in _all_source(NOTEBOOK_DIR / name)


def test_nucleus_notebook_runs_preflight_before_full_extraction():
    source = _all_source(NOTEBOOK_DIR / "extract_nucleus_features.ipynb")
    preflight = source.index("preflight_cellvit.py")
    extraction = source.index("extract_cellvit_features.py")
    assert preflight < extraction
    for flag in ("--input_mpp", "--model_mpp", "--magnification", "--max_estimated_hours"):
        assert flag in source
    assert "requirements-cellvit.txt" in source


def test_run_notebook_matches_the_controlled_budget_protocol():
    config = yaml.safe_load((PROJECT / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert config["cumulative_budget"] == [25, 50, 75, 100, 125, 150, 175, 200]
    source = _all_source(NOTEBOOK_DIR / "run_al_sampler.ipynb")
    assert 'SAMPLER = "scalpel"' in source
    assert "main.run(" in source
    # The CellViT cache must be located for exactly the samplers that declare it,
    # or CELLVIT_DIR stays an unresolved placeholder and the failure surfaces deep
    # inside the first variant instead of here.
    assert 'if "cell_embeddings" in spec.needs:' in source
    assert "def dir_containing(" in source
    assert 'startswith("/kaggle/input")' in source


def test_run_notebook_offers_the_disagreement_ablation():
    """`visual_margin` is the control that isolates what the cell view adds, so
    it should be one keystroke away rather than something to remember."""
    source = _all_source(NOTEBOOK_DIR / "run_al_sampler.ipynb")
    assert '"uncertainty_mode": "visual_margin"' in source
