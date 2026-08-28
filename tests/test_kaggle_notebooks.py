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
    # Extraction is dispatched through the shard worker, which invokes
    # scripts/extract_cellvit_features.py in a child process. The gate is
    # unchanged: nothing may launch a full extraction above the preflight cell.
    extraction = source.index("run_cellvit_shard(")
    assert preflight < extraction
    for flag in ("--input_mpp", "--model_mpp", "--magnification", "--max_estimated_hours"):
        assert flag in source
    assert "requirements-cellvit.txt" in source


@pytest.mark.parametrize(
    "name", ["extract_visual_features.ipynb", "extract_nucleus_features.ipynb"]
)
def test_extraction_notebooks_use_every_gpu(name):
    """Half of a T4 x2 session sitting idle is the largest waste available in
    this project, and extraction is the longest stage. Both notebooks shard the
    work across the visible GPUs rather than assuming one card."""
    source = _all_source(NOTEBOOK_DIR / name)
    assert "visible_gpu_count()" in source
    assert "run_variants_parallel(" in source
    # A shard worker must not be handed a device: it pins its own card and
    # passes 'cuda:0' itself, which run_variants_parallel enforces.
    assert "num_workers=SHARDS" in source


@pytest.mark.parametrize(
    "name", ["extract_visual_features.ipynb", "extract_nucleus_features.ipynb"]
)
def test_extraction_notebooks_verify_the_cache_before_archiving_it(name):
    """A cache whose rows are misaligned with its split loads without error and
    silently corrupts every downstream experiment. Catch that before the archive
    is published and attached to a run notebook, not after."""
    source = _all_source(NOTEBOOK_DIR / name)
    checks = [source.index(m) for m in ("# Verify", "make_archive") if m in source]
    assert len(checks) == 2, f"{name} must both verify and archive"
    assert checks[0] < checks[1], f"{name} archives before verifying"


def test_run_notebook_matches_the_controlled_budget_protocol():
    config = yaml.safe_load((PROJECT / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert config["cumulative_budget"] == [25, 50, 75, 100, 125, 150, 175, 200]
    source = _all_source(NOTEBOOK_DIR / "run_al_sampler.ipynb")
    assert 'SAMPLER = "scalpel"' in source
    # Work is dispatched through the GPU-worker entry point, which resolves its
    # own device: `main.run` takes a torch.device the parent cannot pin.
    assert "run_variants_parallel(" in source
    assert "main.run_on_worker" in source
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


def test_cellvit_wheel_is_installed_with_no_deps():
    """--no-deps is load-bearing, and its absence fails only on Kaggle.

    The cellvit wheel declares 22 dependencies, 19 unpinned. A --hash on any line
    puts pip into --require-hashes mode for the whole file, which then demands ==
    pins for every transitive dependency; CellViT's lack them, so pip aborts with
    "In --require-hashes mode, all requirements must have their versions pinned".
    Installing that tree would also replace Kaggle's NumPy, OpenCV and Ray.
    """
    # Parse the pip ARGUMENT LIST, not the cell text. A substring search passes on
    # the comments that explain why --no-deps matters, so it would still pass with
    # the flag deleted — which is exactly the bug being guarded against.
    notebook = json.loads(
        (NOTEBOOK_DIR / "extract_nucleus_features.ipynb").read_text(encoding="utf-8")
    )
    calls = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("\n".join(
            line for line in _source(cell).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not isinstance(node.args[0], (ast.List, ast.Tuple)):
                continue
            argv = [
                element.value for element in node.args[0].elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
            if "pip" in argv and "requirements-cellvit.txt" in argv:
                calls.append(argv)

    assert len(calls) == 1, f"expected one pip call for the wheel, got {len(calls)}"
    argv = calls[0]
    assert "--no-deps" in argv, f"cellvit must be installed with --no-deps: {argv}"
    assert "--require-hashes" in argv, argv


def test_hashed_and_unhashed_requirements_stay_in_separate_files():
    """--require-hashes applies per FILE, so one unhashed line disables the check
    for the pinned wheel too. PyYAML and Pillow are C extensions with a distinct
    wheel and hash per platform, so hashing them would break on a Kaggle image
    bump — they belong in the unhashed file."""
    hashed = (PROJECT / "requirements-cellvit.txt").read_text(encoding="utf-8")
    extra = (PROJECT / "requirements-cellvit-extra.txt").read_text(encoding="utf-8")

    def requirement_lines(text):
        return [l.strip() for l in text.splitlines()
                if l.strip() and not l.lstrip().startswith("#")]

    for line in requirement_lines(hashed):
        assert "--hash=sha256:" in line, f"unhashed line in hashed file: {line}"
        assert "==" in line, f"unpinned line in hashed file: {line}"
    for line in requirement_lines(extra):
        assert "--hash" not in line, f"hash in the unhashed file: {line}"


def test_nucleus_notebook_installs_only_its_own_requirements():
    """requirements.txt carries matplotlib/seaborn/open_clip for the sampler and
    evaluation notebooks; nothing on the extraction path imports them."""
    source = _all_source(NOTEBOOK_DIR / "extract_nucleus_features.ipynb")
    assert '"-r", "requirements.txt"' not in source
    assert "requirements-cellvit.txt" in source
    assert "requirements-cellvit-extra.txt" in source


def test_extra_requirements_cover_what_no_deps_leaves_out():
    """--no-deps means CellViT brings none of its own dependencies. einops is the
    one its model code imports that Kaggle does not ship."""
    extra = (PROJECT / "requirements-cellvit-extra.txt").read_text(encoding="utf-8")
    for package in ("einops", "transformers", "PyYAML", "Pillow", "tqdm"):
        assert package in extra, package


def test_preflight_gate_is_measured_in_wall_clock_not_gpu_hours():
    """The pilot benchmarks one GPU; the run uses every visible one.

    Without --shards the gate compares GPU-hours against a wall-clock limit and
    rejects jobs that finish comfortably: skintissue estimated 14.5 GPU-h, which
    is ~7.2 h over two T4s, well inside both the 10 h limit and Kaggle's 12 h
    session cap.
    """
    source = (PROJECT / "scripts" / "preflight_cellvit.py").read_text(encoding="utf-8")
    assert '"--shards"' in source
    assert "estimated_hours = gpu_hours / args.shards" in source
    # The cap recommendation must reason in the same units as the gate, or it
    # suggests a cell cap that does not correspond to the limit being enforced.
    assert "budget_gpu_hours" in source

    notebook_source = _all_source(NOTEBOOK_DIR / "extract_nucleus_features.ipynb")
    assert "--shards" in notebook_source
    # The count passed must be the one the real run will use, not a constant.
    assert "PREFLIGHT_SHARDS" in notebook_source
    assert "visible_gpu_count" in notebook_source


def test_nucleus_notebook_can_store_cell_embeddings_without_crop_dino():
    """cell_source defaults to cellvit_embedding, so the crop encoder is a
    second forward pass whose output the default run never reads."""
    source = _all_source(NOTEBOOK_DIR / "extract_nucleus_features.ipynb")
    assert "SKIP_CROP_DINO" in source
    assert "--skip_crop_dino" in source
    # Instance maps are the visualisation sidecar and must stay available.
    assert "SAVE_INSTANCE_MAPS" in source
