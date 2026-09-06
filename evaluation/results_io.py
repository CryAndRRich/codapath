"""Read finished runs' metrics back out of their archives.

A run already computed accuracy, precision, recall and macro-F1 through
`metrics.py::evaluate_probe` and wrote them into `<run>_results.pt`. Every
curve-level analysis downstream -- PALM, ALDA, the comparison table -- needs
nothing but `(budget, accuracy)` per method, so it can read those numbers
directly instead of rebuilding test features and re-scoring saved probes.

That matters beyond convenience: re-scoring means a backbone forward pass over
the whole test set per encoder, and it is the reason the evaluation notebook
needed a GPU, the raw dataset and the DINOv2 checkpoint just to draw a table.

**What replaces the safety the recompute path provided.** Scoring every run
against one freshly built test matrix guaranteed the numbers were comparable.
Reading each run's own numbers does not -- so this module checks the thing that
recompute was implicitly proving: `test_fingerprint` (and `train_fingerprint`)
must agree across every run being compared. Two runs with the same fingerprints
saw the same split through the same `evaluate_probe`, so their numbers are
comparable; two that disagree are not, and `load_curves` raises rather than
quietly plotting them on one axis. `evaluation/palm.py` and `evaluation/alda.py`
consume this module's output and never touch a probe.

**Final-training runs are excluded by default.** A LoRA / auxiliary-loss /
augmentation run measures a different question (does the training pass help)
under a different protocol, so mixing it into a sampler comparison silently
answers neither. The marker is `final_train_cfg`, which `main.py` writes only
when that pass actually ran -- a structural fact about the run, unlike its
filename, which is merely a naming convention that a future axis could break.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "METRIC_KEYS",
    "RunResult",
    "load_run_result",
    "discover_runs",
    "load_curves",
    "average_seeds",
    "format_metric_table",
]

METRIC_KEYS = ("acc", "precision", "recall", "f1")


class RunResult:
    """One finished sweep: its metrics per budget plus the provenance to check."""

    def __init__(self, payload: Dict[str, Any], source: Path):
        self.source = source
        self.run_name: str = payload.get("run_name", "")
        self.sampler: str = payload.get("sampler", "")
        self.dataset: str = payload.get("dataset", "")
        self.seed: Optional[int] = payload.get("seed")
        self.num_classes: Optional[int] = payload.get("num_classes")
        self.class_names: Sequence[str] = payload.get("class_names") or ()
        self.visual_backbone: Optional[str] = payload.get("visual_backbone")
        self.train_fingerprint: Optional[str] = payload.get("train_fingerprint")
        self.test_fingerprint: Optional[str] = payload.get("test_fingerprint")
        self.sampler_config: Dict[str, Any] = payload.get("sampler_config") or {}
        self.linear: Dict[int, Dict[str, Any]] = {
            int(budget): values for budget, values in (payload.get("linear") or {}).items()
        }
        # Written only when a final-training pass actually ran, so its presence
        # -- not the run name -- is what identifies a LoRA/aux/augment variant.
        self.final_train_cfg: Optional[Dict[str, Any]] = payload.get("final_train_cfg")

    @property
    def budgets(self) -> List[int]:
        return sorted(self.linear)

    @property
    def method_label(self) -> str:
        """`run_name` with the seed suffix removed, so seeds group together.

        The run notebooks append `_s<SEED>` to the run name whenever the seed
        differs from the config default, so five seeds of one sampler arrive as
        five distinct `run_name`s (`activeft`, `activeft_s102`, ...) and would
        be averaged as five separate methods. Stripping the suffix that THIS
        run's own recorded seed implies -- rather than pattern-matching any
        trailing `_s<digits>` -- keeps a config axis that legitimately ends
        that way from being truncated.
        """
        suffix = f"_s{self.seed}"
        if self.seed is not None and self.run_name.endswith(suffix):
            return self.run_name[: -len(suffix)]
        return self.run_name

    @property
    def has_final_training(self) -> bool:
        return bool(self.final_train_cfg)

    def metric(self, name: str) -> List[float]:
        if name not in METRIC_KEYS:
            raise KeyError(f"unknown metric {name!r}; expected one of {METRIC_KEYS}")
        return [float(self.linear[budget][name]) for budget in self.budgets]

    def __repr__(self) -> str:
        return (
            f"RunResult(run_name={self.run_name!r}, dataset={self.dataset!r}, "
            f"seed={self.seed}, budgets={len(self.linear)}, "
            f"final_training={self.has_final_training})"
        )


def _read_results_member(archive: Path) -> Tuple[Dict[str, Any], str]:
    """Return the single `*_results.pt` payload inside a run archive.

    A budget-sharded run also writes `<run>_<tag>_results.pt` per shard, which
    `main.merge_budget_shards` folds into the unsharded file. Those shard files
    are partial sweeps, so picking one would silently produce a curve with half
    its budgets missing.
    """
    with zipfile.ZipFile(archive) as handle:
        names = [
            name for name in handle.namelist()
            if name.endswith("_results.pt") and "shard" not in Path(name).name
        ]
        if not names:
            raise FileNotFoundError(f"{archive}: no *_results.pt member (only {handle.namelist()[:5]}...)")
        if len(names) > 1:
            raise ValueError(f"{archive}: expected one merged *_results.pt, found {names}")
        payload = torch.load(io.BytesIO(handle.read(names[0])), map_location="cpu", weights_only=False)
    return payload, names[0]


def load_run_result(path: str | Path) -> RunResult:
    """Load one run from its `.zip` archive or its `*_results.pt` file."""
    path = Path(path)
    if path.suffix == ".zip":
        payload, member = _read_results_member(path)
        return RunResult(payload, path / member)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return RunResult(payload, path)


def discover_runs(
    root: str | Path,
    dataset: Optional[str] = None,
    seeds: Optional[Sequence[int]] = None,
    include_final_training: bool = False,
) -> List[RunResult]:
    """Find every run archive under `root` and load the ones asked for.

    Searches recursively, so both a flat directory of zips and the
    `weights/<dataset>/seed<N>/` layout work without being told which is which.
    `seeds` and `dataset` filter on what the payload itself records, not on the
    path, so a file that was moved or renamed is still classified correctly.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"results root does not exist: {root}")

    archives = sorted(root.rglob("*.zip")) if root.is_dir() else [root]
    runs: List[RunResult] = []
    for archive in archives:
        try:
            run = load_run_result(archive)
        except (FileNotFoundError, ValueError, zipfile.BadZipFile) as error:
            print(f"[results] skipping {archive.name}: {error}")
            continue
        if dataset is not None and run.dataset != dataset:
            continue
        if seeds is not None and run.seed not in set(seeds):
            continue
        if run.has_final_training and not include_final_training:
            continue
        runs.append(run)
    return runs


def _check_comparable(runs: Sequence[RunResult]) -> None:
    """Refuse to compare runs that did not see the same data.

    This is what the old recompute path guaranteed by construction. The
    fingerprints are per (dataset, seed): runs of different seeds legitimately
    differ, so they are grouped by seed before being compared.
    """
    by_seed: Dict[Any, List[RunResult]] = {}
    for run in runs:
        by_seed.setdefault(run.seed, []).append(run)

    for seed, group in by_seed.items():
        for field in ("test_fingerprint", "train_fingerprint"):
            values = {getattr(run, field) for run in group}
            if len(values) > 1:
                offenders = ", ".join(f"{run.run_name}={getattr(run, field)!r}" for run in group)
                raise ValueError(
                    f"seed {seed}: runs disagree on {field}, so their metrics were not "
                    f"computed against the same split and cannot be compared ({offenders})"
                )

    datasets = {run.dataset for run in runs}
    if len(datasets) > 1:
        raise ValueError(f"runs span multiple datasets {sorted(datasets)}; load one dataset at a time")


def load_curves(
    runs: Sequence[RunResult],
    metric: str = "acc",
    label_by: str = "method_label",
    require_same_budgets: bool = True,
) -> Dict[str, Dict[int, List[float]]]:
    """Group runs into `{method: {seed: [metric per budget]}}`.

    `label_by` picks the method label. `method_label` (the default) is the run
    name without its seed suffix, which is what groups several seeds of one
    configuration together while keeping config axes apart
    (`pact_disagreement` vs `pact_visual_margin`). `run_name` keeps every
    seed separate; `sampler` collapses all configs of one sampler into one
    curve, which is wrong whenever two configs of it were run.
    """
    if label_by not in ("method_label", "run_name", "sampler"):
        raise ValueError(
            f"label_by must be 'method_label', 'run_name' or 'sampler', got {label_by!r}"
        )
    if not runs:
        raise ValueError("no runs to build curves from")
    _check_comparable(runs)

    curves: Dict[str, Dict[int, List[float]]] = {}
    budget_sets: Dict[str, Tuple[int, ...]] = {}
    for run in runs:
        label = getattr(run, label_by)
        budgets = tuple(run.budgets)
        if label in curves and run.seed in curves[label]:
            raise ValueError(f"duplicate run for method={label!r} seed={run.seed} ({run.source})")
        curves.setdefault(label, {})[run.seed] = run.metric(metric)
        budget_sets.setdefault(label, budgets)
        if budget_sets[label] != budgets:
            raise ValueError(
                f"{label}: seeds disagree on budgets ({budget_sets[label]} vs {budgets}); "
                "a partially finished sweep cannot be averaged"
            )

    if require_same_budgets:
        distinct = set(budget_sets.values())
        if len(distinct) > 1:
            raise ValueError(
                f"methods disagree on budgets: {sorted(distinct)}. Pass "
                "require_same_budgets=False to compare them anyway."
            )
    return curves


def average_seeds(
    curves: Dict[str, Dict[int, List[float]]],
    budgets: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    """Collapse per-seed curves into mean/std, keeping the seed count visible.

    The mean is what PALM and ALDA are fitted on (the official PALM script
    averages repeated runs before fitting, rather than fitting each and
    averaging parameters). `std` and `n_seeds` are carried alongside so a
    single-seed curve is never mistaken for a converged one.
    """
    budgets = list(budgets)
    averaged: Dict[str, Dict[str, Any]] = {}
    for method, by_seed in curves.items():
        stacked = np.asarray([by_seed[seed] for seed in sorted(by_seed)], dtype=float)
        if stacked.shape[1] != len(budgets):
            raise ValueError(
                f"{method}: {stacked.shape[1]} values per seed but {len(budgets)} budgets given"
            )
        averaged[method] = {
            "budgets": budgets,
            "accuracies": stacked.mean(axis=0).tolist(),
            "std": stacked.std(axis=0, ddof=1).tolist() if len(stacked) > 1 else [0.0] * len(budgets),
            "n_seeds": int(len(stacked)),
            "seeds": sorted(by_seed),
        }
    return averaged


def format_metric_table(
    runs: Sequence[RunResult],
    metric: str = "acc",
    label_by: str = "method_label",
    sort_by_mean: bool = True,
) -> str:
    """Render one metric as `method x budget`, with the per-method mean last."""
    curves = load_curves(runs, metric=metric, label_by=label_by)
    budgets = sorted({budget for run in runs for budget in run.budgets})
    averaged = average_seeds(curves, budgets)

    order = list(averaged)
    if sort_by_mean:
        order.sort(key=lambda method: float(np.mean(averaged[method]["accuracies"])), reverse=True)

    width = max(len(method) for method in order) + 2
    seed_counts = {averaged[method]["n_seeds"] for method in order}
    title = f"{metric} by budget"
    title += f" (mean of {seed_counts.pop()} seeds)" if len(seed_counts) == 1 else " (varying seed counts)"

    lines = [title, ""]
    lines.append(f"{'method':<{width}}" + "".join(f"{budget:>9}" for budget in budgets) + f"{'mean':>10}{'seeds':>7}")
    lines.append("-" * (width + 9 * len(budgets) + 17))
    for method in order:
        values = averaged[method]["accuracies"]
        lines.append(
            f"{method:<{width}}"
            + "".join(f"{value:>9.4f}" for value in values)
            + f"{float(np.mean(values)):>10.4f}{averaged[method]['n_seeds']:>7}"
        )
    return "\n".join(lines)
