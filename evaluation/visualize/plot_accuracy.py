"""Accuracy-vs-budget curves, for one seed or averaged over several.

Replaces the old `plot_accuracy_seed42.py` / `plot_accuracy_seeds.py` pair,
which differed only in whether they averaged. One seed plots the curve straight
from each archive; several plot the mean with +-1 std bands.

Both read the extracted archives under
`data_upload/{baselines,PACT}/<dataset>/seed<N>/<run>/*_results.pt`. Reading the
archives rather than `analysist/*.csv` is deliberate: the CSVs only ever held 8
samplers, because refine and uncertainty_herding had run at seed 42 alone, and
the archives pick up every sampler that exists with no regeneration step.

The PACT row is the `poolcons5` configuration -- PACT as published -- and the
loader asserts `pool_consistency_weight == 5` rather than trusting the run name,
because the plain and poolcons runs differ by about the size of the gap to the
nearest baseline.

Only the highlighted method gets a shaded band unless `--band-all`: overlapping
bands from ten methods obscure the curves they annotate. The highlight is drawn
at the same line width as everything else -- it stands out by position and
zorder, not by being thicker.

Run:
    python3.11 codapath/evaluation/visualize/plot_accuracy.py                # seed 42
    python3.11 codapath/evaluation/visualize/plot_accuracy.py --seeds 38 42 611
"""
import argparse
import glob
import os
import statistics

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODAPATH = os.path.dirname(os.path.dirname(HERE))
PROJECT_ROOT = os.path.dirname(CODAPATH)
DATA_UPLOAD = os.path.join(PROJECT_ROOT, "data_upload")
ASSETS = os.path.join(CODAPATH, "assets", "img")

import sys
sys.path.insert(0, HERE)
from curves import plot_accuracy_curves

DATASETS = ["pathmnist", "histoset", "skintissue"]
DATASET_TITLES = {
    "pathmnist": "PathMNIST",
    "histoset": "HistoSet-5x14",
    "skintissue": "SkinTissue",
}

# (archive stem, display label). Order fixes colour/marker assignment, so the
# highlighted method comes first and keeps the palette's first entry.
BASELINES = [
    ("random", "Random"),
    ("margin", "Margin"),
    ("entropy", "Entropy"),
    ("coreset", "Coreset"),
    ("badge", "BADGE"),
    ("typiclust", "TypiClust"),
    ("activeft", "ActiveFT"),
    ("refine", "REFINE"),
    ("uncertainty_herding", "UHerding"),
]
HIGHLIGHT_LABEL = "PACT (Ours)"
METHOD_ORDER = [HIGHLIGHT_LABEL] + [label for _, label in BASELINES]
BUDGETS = [25, 50, 75, 100, 125, 150, 175, 200]

# Square panels: the competitive methods sit in a narrow band at the top
# because coreset/entropy drag the shared y-axis down, so height is what
# separates them.
PANEL_SIZE = (5.4, 5.4)


def _results_path(run_dir):
    hits = [p for p in glob.glob(os.path.join(run_dir, "*_results.pt"))
            if "shard" not in p]
    assert len(hits) == 1, (run_dir, hits)
    return hits[0]


def _curve(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    linear = payload["linear"]
    return payload, [linear[b]["acc"] * 100.0 for b in BUDGETS]


def load_baseline(dataset, seed, stem):
    run_dir = os.path.join(DATA_UPLOAD, "baselines", dataset, f"seed{seed}",
                           f"{dataset}_{stem}_seed{seed}")
    if not os.path.isdir(run_dir):
        return None
    payload, accuracy = _curve(_results_path(run_dir))
    assert payload["sampler"] == stem, run_dir
    assert payload["dataset"] == dataset and payload["seed"] == seed, run_dir
    return accuracy


def load_pact(dataset, seed):
    suffix = "" if seed == 42 else f"_s{seed}"
    run_dir = os.path.join(
        DATA_UPLOAD, "PACT", dataset, f"seed{seed}",
        f"{dataset}_pact_disagreement_poolcons5{suffix}_seed{seed}",
    )
    assert os.path.isdir(run_dir), f"missing PACT poolcons5 archive: {run_dir}"
    payload, accuracy = _curve(_results_path(run_dir))
    assert payload["sampler"] == "pact", run_dir
    assert payload["dataset"] == dataset and payload["seed"] == seed, run_dir
    # The run name is not proof of the configuration; the config is.
    assert payload["sampler_config"].get("pool_consistency_weight") == 5, run_dir
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--band-all", action="store_true",
                        help="shade every method, not only the highlight")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    seeds = arguments.seeds

    budgets_by_dataset = {}
    means_per_dataset, stds_per_dataset = [], []

    for dataset in DATASETS:
        curves = {HIGHLIGHT_LABEL: [load_pact(dataset, s) for s in seeds]}
        for stem, label in BASELINES:
            runs = [load_baseline(dataset, s, stem) for s in seeds]
            absent = [s for s, run in zip(seeds, runs) if run is None]
            if absent:
                print(f"WARNING: {dataset}/{stem} missing at seeds {absent}; skipped")
                continue
            curves[label] = runs

        means, stds = {}, {}
        for label in METHOD_ORDER:
            if label not in curves:
                continue
            per_seed = curves[label]
            means[label] = [statistics.mean(c[i] for c in per_seed)
                            for i in range(len(BUDGETS))]
            if len(seeds) > 1:
                stds[label] = [statistics.stdev([c[i] for c in per_seed])
                               for i in range(len(BUDGETS))]

        budgets_by_dataset[dataset] = BUDGETS
        means_per_dataset.append(means)
        stds_per_dataset.append(stds)

    labels = [label for label in METHOD_ORDER if label in means_per_dataset[0]]
    os.makedirs(ASSETS, exist_ok=True)
    suffix = "_".join(str(s) for s in seeds)
    out = arguments.out or os.path.join(ASSETS, f"accuracy_vs_budget_seeds{suffix}.png")

    extra = {}
    if len(seeds) > 1:
        extra = {
            "std_data": stds_per_dataset,
            "band_methods": None if arguments.band_all else [HIGHLIGHT_LABEL],
            "panel_size": PANEL_SIZE,
            "linewidth": 1.2,
            "markersize": 4.0,
            "highlight_scale": 1.0,
        }

    plot_accuracy_curves(
        budgets_by_dataset, means_per_dataset,
        methods=labels, highlight=HIGHLIGHT_LABEL,
        dataset_titles=DATASET_TITLES, save_path=out,
        **extra,
    )
    band = "every method" if arguments.band_all else HIGHLIGHT_LABEL + " only"
    print(f"[fig] wrote {out}")
    print(f"[fig] seeds {seeds}" + (f"; band on {band}" if len(seeds) > 1 else ""))


if __name__ == "__main__":
    main()
