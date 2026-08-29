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
    "extract_vlm_features.ipynb",
    "run_al_sampler.ipynb",
    "run_al_baseline.ipynb",
    "evaluate_al_sampler.ipynb",
}


def _source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _all_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(_source(cell) for cell in notebook["cells"])


def test_notebook_set_is_exactly_the_supported_entry_points():
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
    # The cache-lookup search itself lives in utils/kaggle.py, shared with every
    # other notebook, rather than being duplicated inline here.
    assert "from utils.kaggle import" in source
    assert "find_cellvit_cache" in source
    assert "find_visual_cache" in source
    assert 'startswith("/kaggle/input")' in source


def test_run_notebook_offers_the_disagreement_ablation():
    """`visual_margin` is the control that isolates what the cell view adds, so
    it should be one keystroke away rather than something to remember."""
    source = _all_source(NOTEBOOK_DIR / "run_al_sampler.ipynb")
    assert '"uncertainty_mode": "visual_margin"' in source


def test_baseline_notebook_only_runs_a_declared_baseline():
    """`scalpel` needs a CellViT cell view, which this notebook never loads a
    cache for -- picking it here must fail loudly in the notebook's own assert
    cell, not deep inside main.py after GPU time is already spent on budget 25."""
    source = _all_source(NOTEBOOK_DIR / "run_al_baseline.ipynb")
    assert "from sampling.specs import BASELINE_SAMPLERS" in source
    assert "assert SAMPLER in BASELINE_SAMPLERS" in source
    assert '"cell_embeddings" not in spec.needs' in source
    assert '"text_embeddings" not in spec.needs' in source


def test_baseline_notebook_never_touches_cellvit():
    """The whole point of this notebook is to run from the published DINOv2
    cache alone. If a CellViT cache path, a `find_cellvit_cache` import, or
    the extraction requirements file creeps back in, the split from
    run_al_sampler.ipynb has been undone by accident.

    This does not forbid the word "CellViT" outright: the assert cell
    legitimately explains the sampler it rejects and why, in a comment and in
    the assert's own message. What must never appear is an actual mechanism
    for reading a CellViT cache.
    """
    source = _all_source(NOTEBOOK_DIR / "run_al_baseline.ipynb")
    assert "find_cellvit_cache" not in source
    assert "CELLVIT_DIR" not in source
    assert "cellvit_cache_dir=" not in source
    assert "requirements-cellvit" not in source


def test_baseline_notebook_never_requests_a_vlm_or_text_prior():
    source = _all_source(NOTEBOOK_DIR / "run_al_baseline.ipynb")
    assert "text_embeddings" not in source or '"text_embeddings" not in spec.needs' in source
    for name in ("CONCH", "vlm_primary", "vlm_secondary", "generate_class_description"):
        assert name not in source


def test_baseline_notebook_resolves_features_by_name_not_hardcoded_path():
    source = _all_source(NOTEBOOK_DIR / "run_al_baseline.ipynb")
    assert "from utils.kaggle import find_data_root, find_visual_cache" in source
    assert "find_visual_cache(DATASET, SEED, vit_name, hint=FEATURE_DIR)" in source


def test_baseline_notebook_supports_resume_and_both_gpus():
    source = _all_source(NOTEBOOK_DIR / "run_al_baseline.ipynb")
    assert "run_variants_parallel(" in source
    assert "main.run_on_worker" in source
    # Resume is keyed on the MERGED results file, which only exists once the
    # whole sweep finished -- a run killed mid-sweep must re-run, not be
    # mistaken for done.
    assert 'f"{RUN}_results.pt").is_file()' in source


def test_baseline_notebook_keeps_raw_images_and_features_as_separate_mounts():
    """Two Kaggle Datasets, and the feature cache cannot stand in for the raw
    images.

    The cache holds DINOv2 matrices only. The oracle labels a sampler selects
    on, the labels the probe trains against, and the sample-order fingerprint
    that VALIDATES the cache all come from the dataset itself, so `main.run`
    opens it either way. Dropping DATA_PATHS to "run from the cache alone"
    fails inside `get_data_loaders` before the cache is ever consulted -- so
    both mounts must stay declared and distinct.
    """
    source = _all_source(NOTEBOOK_DIR / "run_al_baseline.ipynb")
    assert "DATA_ROOT" in source and "FEATURE_DIR" in source
    assert "find_data_root(" in source
    assert "DATA_PATHS" in source
    # The raw-image loader must still be reached; the cache only replaces the
    # backbone forward pass, never the dataset.
    assert 'DATA_PATHS[DATASET]' in source


@pytest.mark.parametrize(
    "name",
    [
        "run_al_baseline.ipynb",
        "run_al_sampler.ipynb",
        "extract_visual_features.ipynb",
        "extract_nucleus_features.ipynb",
    ],
)
def test_one_configuration_per_run(name):
    """A run produces one archive, so it must describe one configuration.

    A list of seeds, datasets or variants would put several results under a
    single archive name that cannot say which result is which -- the same
    collision `_default_run_name` guards against, one level up. Sweeping is
    done by running the notebook again, which costs nothing: both GPUs are used
    by splitting the budget list (or sharding one extraction), not by bundling
    configurations into one session.
    """
    source = _all_source(NOTEBOOK_DIR / name)
    for plural in ("SEEDS", "DATASETS", "VARIANTS"):
        assert plural not in source, f"{name} still sweeps {plural} in one run"
    assert "SEED = " in source


@pytest.mark.parametrize("name", ["run_al_baseline.ipynb", "run_al_sampler.ipynb"])
def test_run_notebooks_print_an_ordered_results_table(name):
    """Two GPUs share one output stream, so their progress lines interleave and
    a budget's numbers can appear anywhere. The summary cell re-reads the saved
    results file instead of trusting the log, so the table is ordered no matter
    how the run printed.
    """
    source = _all_source(NOTEBOOK_DIR / name)
    assert 'torch.load(results_path, weights_only=False)' in source
    assert "for budget in sorted(linear):" in source
    # Metrics only: the per-step acquisition scores stay in the saved files.
    for metric in ("accuracy", "precision", "recall", "macro F1"):
        assert metric in source


def test_baseline_notebook_splits_budgets_by_the_prefix_exact_flag():
    """The GPU split must be decided by `spec.prefix_exact`, not a name list.

    A hand-kept list of "shardable" samplers is exactly the thing that goes
    stale when a sampler is added: it would silently either repeat a shared
    selection pass per shard, or leave a card idle. Asserting on the flag makes
    the notebook track `sampling.specs` automatically.
    """
    source = _all_source(NOTEBOOK_DIR / "run_al_baseline.ipynb")
    assert "SPLIT_BUDGETS" in source
    assert "not spec.prefix_exact" in source
    assert "shard_tag=tag" in source
    # Round-robin, not contiguous: cost grows with the budget, so a contiguous
    # split hands one worker every expensive budget.
    assert "budgets[i::n]" in source


def test_baseline_notebook_merges_shards_back_into_one_results_file():
    """Downstream must not need to know a run was sharded."""
    source = _all_source(NOTEBOOK_DIR / "run_al_baseline.ipynb")
    assert "main.merge_budget_shards" in source


def test_baseline_notebook_does_not_fit_palm():
    """PALM belongs to evaluate_al_sampler.ipynb, which reads these files.

    Checks the run path, not the word: the intro cell legitimately explains
    that PALM is deferred, so a substring test would pass on that prose.
    """
    source = _all_source(NOTEBOOK_DIR / "run_al_baseline.ipynb")
    assert "palm_evaluate" not in source
    assert "_fit_palm" not in source


def test_baseline_notebook_archives_exactly_like_the_extraction_notebooks():
    """One zip at the top of /kaggle/working, loose files deleted.

    A "Save & Run All" session has no terminal and no kaggle CLI, so the Output
    tab is the only way a file leaves it -- printing `kaggle datasets create`
    is advice nobody in that session can follow. Leaving the originals beside
    the zip also doubles the download and can push a session over the ~20 GB
    Output quota, at which point the tab shows NOTHING, including the files
    that were fine. Both extraction notebooks already do it this way.
    """
    source = _all_source(NOTEBOOK_DIR / "run_al_baseline.ipynb")
    assert "results_archive_stem(DATASET, SAMPLER, SEED)" in source
    assert 'shutil.make_archive(str(ARCHIVE), "zip", root_dir=SOURCE)' in source
    # The loose checkpoints must go once the zip exists.
    assert "shutil.rmtree(SOURCE, ignore_errors=True)" in source
    # The CLI path is unusable in the session this notebook runs in.
    assert "kaggle datasets create" not in source
    assert "dataset-metadata.json" not in source


def test_every_publishing_notebook_writes_its_zip_outside_the_source_tree():
    """`make_archive` writing inside the directory it archives packs a partial
    copy of itself on a second run, so each notebook asserts the two are
    distinct rather than trusting the layout."""
    for name in (
        "run_al_baseline.ipynb",
        "extract_visual_features.ipynb",
        "extract_nucleus_features.ipynb",
        "extract_vlm_features.ipynb",
    ):
        source = _all_source(NOTEBOOK_DIR / name)
        assert "SOURCE.resolve() != WORKING.resolve()" in source, name


def test_vlm_notebook_installs_the_conch_package_not_just_open_clip():
    """`pip install open_clip_torch` alone is not enough -- CONCH has its own
    tokenizer and factory (features/vlm.py module docstring). Missing this
    line means every cell past it fails on the very first import."""
    source = _all_source(NOTEBOOK_DIR / "extract_vlm_features.ipynb")
    assert "git+https://github.com/mahmoodlab/CONCH.git" in source


def test_vlm_notebook_writes_both_embedding_spaces_in_one_pass():
    """The two CONCH embedding spaces (probe space vs. text-comparison space)
    are not interchangeable, and re-running the whole dataloader for the
    second one would double the cost of the notebook's expensive part
    (448x448, 4x DINOv2's pixels). Both must come from ONE call."""
    source = _all_source(NOTEBOOK_DIR / "extract_vlm_features.ipynb")
    assert "get_or_extract_vlm_features(" in source
    assert "cached['train']" in source or 'cached["train"]' in source
    assert "cached['proj_train']" in source or 'cached["proj_train"]' in source


def test_vlm_notebook_uses_the_factorys_own_preprocess():
    """§10.3's trap: hand-rolling a Normalize() for CONCH silently uses the
    wrong (ImageNet, not OpenAI CLIP) statistics. The notebook must pass
    `transform=` built from `load_conch`'s own returned preprocess, never
    construct its own transforms.Normalize for this model."""
    source = _all_source(NOTEBOOK_DIR / "extract_vlm_features.ipynb")
    assert "load_conch(" in source
    assert "transform=conch_preprocess" in source
    assert "transforms.Normalize" not in source


def test_vlm_notebook_reads_logit_scale_from_the_loaded_checkpoint():
    """A hard-coded temperature (e.g. 0.05) is the mistake §10.6 documents:
    `logit_scale` is a LEARNED parameter, and the official zero-shot code
    reads it off the checkpoint, not a config constant."""
    source = _all_source(NOTEBOOK_DIR / "extract_vlm_features.ipynb")
    assert "conch_model.logit_scale.exp()" in source
    assert "logit_scale=0.05" not in source
    assert "logit_scale = 0.05" not in source


def test_vlm_notebook_asserts_class_order_before_using_official_prompts():
    """A silent class-order mismatch does not crash -- it permutes the
    confusion matrix and reports a wrong number that looks plausible. This
    must be checked in code, not assumed."""
    source = _all_source(NOTEBOOK_DIR / "extract_vlm_features.ipynb")
    assert "assert_class_order_matches_prompts(" in source


def test_vlm_notebook_rejects_the_official_prompts_on_a_non_pathmnist_dataset():
    """The official CONCH prompt set is CRC100K-specific (9 classes) and only
    exists for PathMNIST; HistoSet/SkinTissue must not silently reuse it."""
    source = _all_source(NOTEBOOK_DIR / "extract_vlm_features.ipynb")
    assert 'assert DATASET == "pathmnist"' in source


def test_vlm_notebook_asserts_a_loose_not_exact_zeroshot_threshold():
    """79.1% is the paper's number on CRC100K at its native scale; PathMNIST
    is resized 224->448 here, so an exact reproduction is not the bar (see
    PLAN_IMPLEMENT.md 4.2). The notebook must assert a looser bound, not the
    exact published figure, and must not silently skip the check."""
    source = _all_source(NOTEBOOK_DIR / "extract_vlm_features.ipynb")
    assert "zero_shot_accuracy > 0.70" in source
    assert "0.791" not in source


def test_vlm_notebook_keeps_dataset_seed_vlm_and_style_singular():
    """Same rule as every other notebook (§2.0): one configuration, one zip.
    A description STYLE is an extra axis this notebook has that the others
    don't, and it must follow the same rule."""
    source = _all_source(NOTEBOOK_DIR / "extract_vlm_features.ipynb")
    for plural in ("SEEDS", "DATASETS", "VLMS", "STYLES", "DESCRIPTION_STYLES"):
        assert plural not in source, f"still sweeps {plural} in one run"
    assert "DESCRIPTION_STYLE = " in source


def test_vlm_notebook_archive_stem_includes_the_description_style():
    """Two styles are two artifacts (different text prototype file), so the
    archive name must distinguish them -- not just dataset/seed/VLM."""
    source = _all_source(NOTEBOOK_DIR / "extract_vlm_features.ipynb")
    assert "vlm_archive_stem(DATASET, SEED, VLM, DESCRIPTION_STYLE)" in source


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
