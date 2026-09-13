"""Compute a t-SNE embedding of a DINOv2 feature cache, once, and cache it.

This is the expensive half of the "what did PACT select" figure, split out so
the plotting half can be re-run freely. It reads ONLY the feature cache and the
sample-id list -- no pixels, no model, no GPU -- so it runs on a laptop.

Row order is the contract. `selected_indices` in a `*_selected_budget_*.pt`
indexes the cache's TRAIN rows, so the embedding must be computed over those
rows in that order and nothing may reorder them. The guard is the fingerprint:
`sample_order_fingerprint(sample_ids)` must equal the cache manifest's
`train_fingerprint`, which is the same digest every run recorded when it
selected. A mismatch means the ids and the features are different orderings of
the dataset and every plotted point would be the wrong patch -- silently, since
both arrays still have 103495 rows.

Subsampling keeps every selected point and draws the rest at random, because a
uniform sample of a 103k pool would drop most of a 25-point selection. The kept
rows are written out with the embedding so the plotter never has to recompute
which row became which dot.

Run:
  python3.11 codapath/evaluation/visualize/compute_tsne.py \
      --features <cache>_train.npy --sample-ids sample_ids.npy \
      --manifest <cache>_manifest.json

The embedding lands in `data_upload/tsne_cache/` unless --out says otherwise.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

HERE = os.path.dirname(os.path.abspath(__file__))
CODAPATH = os.path.dirname(os.path.dirname(HERE))
PROJECT_ROOT = os.path.dirname(CODAPATH)
DEFAULT_CACHE = os.path.join(PROJECT_ROOT, "data_upload", "tsne_cache")
sys.path.insert(0, CODAPATH)

from data.identity import sample_order_fingerprint


def load_selection_rows(selection_paths):
    """Union of every selected row index across the given selection files."""
    import torch

    rows = set()
    for path in selection_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows.update(int(i) for i in payload["selected_indices"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default=None,
                        help="default: data_upload/tsne_cache/<name>.npz")
    parser.add_argument("--subsample", type=int, default=15000,
                        help="0 = embed the whole pool")
    parser.add_argument("--pca", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep", nargs="*", default=[],
                        help="selection .pt files whose rows must be kept")
    args = parser.parse_args()

    features = np.load(args.features, mmap_mode="r")
    sample_ids = [str(s) for s in np.load(args.sample_ids, allow_pickle=True)]
    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    # The alignment guard. Without it a wrong-order id list would still have the
    # right length and produce a plausible, wrong figure.
    if len(sample_ids) != features.shape[0]:
        raise SystemExit(
            f"sample ids ({len(sample_ids)}) and features "
            f"({features.shape[0]}) disagree on row count"
        )
    fingerprint = sample_order_fingerprint(sample_ids)
    expected = manifest.get("train_fingerprint")
    if expected is not None and fingerprint != expected:
        raise SystemExit(
            "sample-id order does not match the feature cache:\n"
            f"  ids      {fingerprint}\n  manifest {expected}"
        )
    print(f"[tsne] {features.shape[0]} rows, fingerprint {fingerprint[:16]} ok")

    # Class label per row, read off the ImageFolder path prefix, so the plot
    # needs no dataset. Sorted names reproduce ImageFolder's own class indices.
    prefixes = [s.split("/")[0] for s in sample_ids]
    class_names = sorted(set(prefixes))
    lookup = {name: i for i, name in enumerate(class_names)}
    labels = np.array([lookup[p] for p in prefixes], dtype=np.int16)

    keep = load_selection_rows(args.keep) if args.keep else set()
    total = features.shape[0]
    if args.subsample and args.subsample < total:
        rng = np.random.default_rng(args.seed)
        keep_rows = np.array(sorted(keep), dtype=np.int64)
        remaining = np.setdiff1d(np.arange(total, dtype=np.int64), keep_rows)
        extra = rng.choice(
            remaining, size=max(args.subsample - len(keep_rows), 0), replace=False
        )
        rows = np.sort(np.concatenate([keep_rows, extra]))
        print(f"[tsne] subsample {len(rows)} rows "
              f"({len(keep_rows)} selected kept, {len(extra)} random)")
    else:
        rows = np.arange(total, dtype=np.int64)
        print(f"[tsne] embedding all {total} rows")

    subset = np.ascontiguousarray(features[rows], dtype=np.float32)

    started = time.time()
    reduced = PCA(n_components=args.pca, random_state=args.seed).fit_transform(subset)
    print(f"[tsne] PCA{args.pca} in {time.time() - started:.1f}s")

    started = time.time()
    embedding = TSNE(
        n_components=2, init="pca", random_state=args.seed,
        perplexity=args.perplexity, n_jobs=-1,
    ).fit_transform(reduced)
    print(f"[tsne] t-SNE in {(time.time() - started) / 60:.1f} min")

    out = args.out or os.path.join(
        DEFAULT_CACHE,
        f"tsne_{manifest.get('dataset', 'pool')}_s{manifest.get('seed', args.seed)}.npz",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(
        out,
        embedding=embedding.astype(np.float32),
        rows=rows,
        labels=labels[rows],
        class_names=np.array(class_names),
        train_fingerprint=fingerprint,
        perplexity=args.perplexity,
        pca_components=args.pca,
        seed=args.seed,
    )
    print(f"[tsne] wrote {out}")


if __name__ == "__main__":
    main()
