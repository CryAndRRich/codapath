"""graph_sargraph — SARGraphAL-faithful graph AL on a single fused embedding.

This is "Tổ hợp A": one reducer over `concat(visual, cell)`, ONE graph built with
SARGraphAL's own kernel, and SARGraphAL's own sequential acquisition loop. It is
deliberately separate from `sampling/graph_deuce.py` ("Tổ hợp B": two independent
VAEs -> two graphs -> DEUCE merge) so the two can be compared and developed
without touching each other; every acquisition helper is imported from there
rather than duplicated, so the SCORING code is literally identical between the
two and only the graph differs.

Differences from `graph_deuce`, all deliberate
----------------------------------------------
1. Fusion: `concat(visual, cell)` -> ONE reducer, instead of two separate VAEs
   merged as two graphs. `reducer` selects pca (default) | vae | none.
2. Graph: `graph_al.graph.knn_graph` — the paper's own self-tuning kernel
   `w_ij = exp(-4 d_ij^2 / d_k(i)^2)` symmetrised as `(W+W^T)/2` — instead of
   DEUCE's GRAPHNORM/fuzzy-union `knn_graph_umap`. `graph.py`'s own docstring
   reserves `knn_graph` for exactly this sampler.

KNOWN DEVIATION from the official code (deliberate, decided 2026-08-18 — RECORD
THIS AS AN EXPERIMENTAL LIMITATION): `knn_graph` runs with `normalize=True`, so
the reducer output is projected onto the unit sphere before the kNN search.
`jwcalder/MSTAR-Active-Learning` calls `gl.weightmatrix.knn` on the RAW,
un-normalized 32-d CNNVAE latent instead. Under PCA the discarded vector length
is the distance from the data mean, which is real signal. It is kept normalized
for consistency with every other kernel in this repo (CLAUDE.md "always
L2-normalize before treating a dot product as cosine"); flip `normalize=False`
in the `knn_graph` call here to measure what it costs.
3. Adds `laplace_plus_resistance`: effective-resistance / commute-time coverage
   (`graph_al/resistance.py`), the graph-intrinsic alternative to the PPR
   coverage `graph_deuce` already offers.

`reducer="pca"` is the default
------------------------------
SARGraphAL trains a CNNVAE on RAW SAR imagery, where the VAE is genuine
unsupervised representation learning. Here the reducer instead consumes DINOv2
embeddings that are ALREADY a learned representation, so it is only doing
dimensionality reduction — which PCA does with no hyperparameters, no training
loop, and no collapse mode. PCA is therefore the current experimental path, not
a control; `reducer="vae"` is kept for a later ablation answering "does the VAE
buy anything over plain dimensionality reduction?".

`_pca` deliberately avoids `torch.linalg.svd` on the full `(N, D)` matrix (it
would allocate `U` at ~600 MB for N=100k) and goes through the `D x D`
covariance matrix instead — same subspace, ~10 MB. It prints the explained
variance ratio, which is the number that says whether `vae_latent_dim`
components actually retain the signal in `concat(visual, cell)`.

On `vae_beta` (only relevant if `reducer="vae"` is switched back on)
-------------------------------------------------------------------
The default ELBO in `graph_al/vae.py` is sum-reduced with `beta=1.0`, and
`_fuse` L2-normalizes its input, which caps the per-sample reconstruction term
at `||x||^2 = 1` while the KL of a 32-d latent runs to tens of nats. At
`beta=1.0` the KL therefore outweighs reconstruction by one to two orders of
magnitude and the optimiser's cheapest move is to drive the posterior to the
prior — textbook posterior collapse, giving a latent that carries no
information and a kNN graph built on noise. `vae_beta` is exposed here (default
`1e-2`) and `_report_latent_collapse` prints the active-unit count after
training so collapse is VISIBLE instead of hiding behind a total-loss curve
that falls smoothly either way.
"""

from typing import Dict, List, Optional, Set

import numpy as np
import scipy.sparse as sps
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from graph_al.graph import knn_graph
from graph_al.laplace import laplace_learning, laplace_margin
from graph_al.resistance import resistance_embedding, resistance_to_set
from graph_al.sparse_coverage import greedy_coverage_sparse
from graph_al.vae import MLPVAE, train_vae
from . import register_sampler
from .graph_deuce import (
    _bootstrap_dense_sigma,
    _has_two_classes,
    _rank_normalize,
    _round_scores_laplace_margin,
    _round_scores_laplace_plus_ppr,
    _round_uncertainty_uherding_original,
    _run_uherding_swap_uncertainty_round,
    _select_batch_with_discount,
    _select_points_per_point_laplace_margin,
    _select_points_per_point_laplace_plus_ppr,
    _unlabeled_array,
)

VALID_VARIANTS = {
    "laplace_margin",
    "laplace_plus_ppr",
    "laplace_plus_resistance",
    "uherding_swap_uncertainty",
    "uherding_swap_coverage",
}

VALID_REDUCERS = {"vae", "pca", "none"}


# ---------------------------------------------------------------------------
# Fusion + reduction + graph (label-free: built once, reused every round/budget)
# ---------------------------------------------------------------------------

def _fuse(dino_np: np.ndarray, cell_np: Optional[np.ndarray],
          reliability: Optional[np.ndarray], device: torch.device) -> torch.Tensor:
    """`concat(L2(visual), L2(cell))`, with unreliable patches imputed.

    Patches where CellViT found no nucleus have an undefined cell vector. They
    are filled with the MEAN of the reliable cell vectors (the same convention
    `nucleus_coverage`'s `missing_impute="mean"` uses) rather than zeros: a zero
    row is not a neutral element after concatenation, it is a point far from
    every real cell vector, which would make empty-background patches look
    mutually similar and cluster together in the graph for an artefactual
    reason.

    `cell_np=None` gives visual-only — which is what SARGraphAL itself does (one
    embedding), and lets the whole pipeline run before any nucleus cache exists.
    """
    x_vis = F.normalize(torch.as_tensor(dino_np, device=device, dtype=torch.float32), p=2, dim=1)
    if cell_np is None:
        return x_vis

    cell = np.array(cell_np, dtype=np.float32, copy=True)
    if reliability is not None:
        reliable = np.asarray(reliability) > 0
        if not reliable.any():
            raise ValueError(
                "graph_sargraph got 0 reliable cell patches — the nucleus cache "
                "is empty or reliability_mode filtered everything out"
            )
        if not reliable.all():
            cell[~reliable] = cell[reliable].mean(axis=0)
    x_cell = F.normalize(torch.as_tensor(cell, device=device, dtype=torch.float32), p=2, dim=1)
    return torch.cat([x_vis, x_cell], dim=1)


def _report_latent_collapse(z: torch.Tensor, tag: str) -> None:
    """Print per-dimension latent activity. A VAE whose latent has collapsed
    produces near-identical std across the pool on every dimension; `active`
    counts dimensions whose std exceeds a small threshold."""
    std = z.std(dim=0)
    active = int((std > 1e-2).sum().item())
    print(f"[{tag}] latent dim={z.shape[1]} active(std>1e-2)={active}/{z.shape[1]} "
          f"| std min={std.min().item():.4f} med={std.median().item():.4f} "
          f"max={std.max().item():.4f}")
    if active <= 1:
        print(f"[{tag}] WARNING: latent looks COLLAPSED — the kNN graph below is "
              f"built on ~no information. Lower `vae_beta` or use reducer='pca'.")


def _pca(x: torch.Tensor, latent_dim: int, chunk_size: int) -> torch.Tensor:
    """Top-`latent_dim` principal components, via the DxD covariance matrix.

    NOT `torch.linalg.svd(xc, full_matrices=False)`: that allocates `U` at the
    full `(N, D)` size — ~600 MB at N=100k, D~1500 — on top of `xc` and the
    DINOv2 features already resident on the GPU, plus cuSOLVER workspace, and
    gesvd is slow on very tall matrices. The eigenvectors of `X^T X` ARE the
    right singular vectors of `X`, so accumulating the `D x D` Gram matrix
    chunk-by-chunk gives the identical subspace for ~`D^2` floats (~10 MB).

    The Gram accumulator is float64: summing 100k outer products in float32
    loses precision in exactly the small trailing eigenvalues that decide which
    components make the cut.
    """
    mean = x.mean(dim=0, keepdim=True)
    D = x.shape[1]
    gram = torch.zeros((D, D), device=x.device, dtype=torch.float64)
    for start in range(0, x.shape[0], chunk_size):
        blk = (x[start:start + chunk_size] - mean).to(torch.float64)
        gram += blk.T @ blk

    evals, evecs = torch.linalg.eigh(gram)          # ascending
    total = float(torch.clamp(evals, min=0.0).sum().item())
    top = evecs[:, -latent_dim:].flip(dims=(1,))    # descending eigenvalue order
    kept = float(torch.clamp(evals[-latent_dim:], min=0.0).sum().item())
    ratio = kept / total if total > 1e-12 else float("nan")
    print(f"[graph_sargraph PCA] latent_dim={latent_dim}/{D} "
          f"explained variance={ratio:.4f}")
    if ratio < 0.5:
        print(f"[graph_sargraph PCA] NOTE: {latent_dim} components keep under half "
              f"the variance of concat(visual, cell); raise `vae_latent_dim` before "
              f"trusting the acquisition results built on this graph.")

    top = top.to(x.dtype)
    z = torch.empty((x.shape[0], latent_dim), device=x.device, dtype=x.dtype)
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        z[start:end] = (x[start:end] - mean) @ top
    return z


def _reduce(x: torch.Tensor, reducer: str, latent_dim: int, hidden, epochs: int,
            lr: float, batch_size: int, beta: float, chunk_size: int,
            device: torch.device) -> torch.Tensor:
    if reducer not in VALID_REDUCERS:
        raise ValueError(f"Unknown reducer={reducer!r}, expected one of {sorted(VALID_REDUCERS)}")

    if reducer == "none":
        return x

    if reducer == "pca":
        return _pca(x, latent_dim, chunk_size)

    vae = MLPVAE(input_dim=x.shape[1], hidden_dims=hidden, latent_dim=latent_dim).to(device)
    train_vae(vae, x, epochs=epochs, lr=lr, batch_size=batch_size, beta=beta,
              device=device, desc="graph_sargraph VAE")
    z = vae.latent(x)
    _report_latent_collapse(z, "graph_sargraph VAE")
    del vae
    clear_memory()
    return z


_GRAPH_CACHE: Dict[tuple, tuple] = {}


def _build_graph_cached(dino_np, cell_np, reliability, device, cfg_key, **kw):
    """Graph + resistance embedding are fully unsupervised, so they depend on
    neither the budget nor the acquisition variant. `run.py` calls this sampler
    once per entry of `cumulative_budget` with the SAME arrays, so without this
    cache the VAE, the kNN search and the 200 resistance solves would all be
    repeated for every budget and every variant in a session, for no benefit.
    Keyed on `id()` of the input arrays (never reassigned inside run.py's loop)
    plus every setting that changes the result.
    """
    key = (id(dino_np), id(cell_np), cfg_key)
    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]

    x = _fuse(dino_np, cell_np, reliability, device)
    z = _reduce(x, kw["reducer"], kw["latent_dim"], kw["hidden"], kw["vae_epochs"],
                kw["vae_lr"], kw["vae_batch_size"], kw["vae_beta"], kw["chunk_size"],
                device)
    del x
    clear_memory()

    W = knn_graph(z, k=kw["k"], kernel=kw["kernel"], chunk_size=kw["chunk_size"])
    n_comp = sps.csgraph.connected_components(W, directed=False, return_labels=False)
    print(f"[graph_sargraph] graph N={W.shape[0]} k={kw['k']} kernel={kw['kernel']} "
          f"nnz={W.nnz} components={n_comp}")
    if n_comp > 1:
        print(f"[graph_sargraph] NOTE: graph is disconnected ({n_comp} components). "
              f"Laplace learning cannot propagate labels into a component holding "
              f"no labeled node; raise `k` if this number is large.")

    Z_res = None
    if kw["need_resistance"]:
        Z_res = resistance_embedding(W, n_sketch=kw["n_sketch"], seed=kw["seed"])
        print(f"[graph_sargraph] resistance embedding: {Z_res.shape} "
              f"(n_sketch={kw['n_sketch']})")

    del z
    clear_memory()
    _GRAPH_CACHE[key] = (W, Z_res)
    return W, Z_res


# ---------------------------------------------------------------------------
# New acquisition variant: Laplace margin (uncertainty) + effective-resistance
# coverage. Mirrors `_round_scores_laplace_plus_ppr` exactly, swapping the
# coverage term, so the two coverage mechanisms are compared on equal footing
# (same blend, same rank-normalisation, same selection loop).
# ---------------------------------------------------------------------------

def _round_scores_laplace_plus_resistance(
    W: sps.spmatrix,
    Z_res: np.ndarray,
    selected_indices: List[int],
    oracle_labels: np.ndarray,
    num_classes: int,
    N: int,
    device: torch.device,
    alpha: float,
    reduction: str,
) -> torch.Tensor:
    if not _has_two_classes(oracle_labels, selected_indices):
        return torch.ones(N, device=device, dtype=torch.float32)
    u = laplace_learning(
        W, np.asarray(selected_indices), oracle_labels[selected_indices], num_classes
    )
    uncertainty = laplace_margin(u)
    # HIGH resistance to the labeled set = poorly covered = worth acquiring, so
    # unlike PPR (where -pi is needed) this needs no sign flip.
    coverage = resistance_to_set(Z_res, np.asarray(selected_indices), reduction=reduction)
    score = (1.0 - alpha) * _rank_normalize(uncertainty) + alpha * _rank_normalize(coverage)
    return torch.as_tensor(score, device=device, dtype=torch.float32)


def _select_points_per_point_laplace_plus_resistance(
    W, Z_res, selected_indices, selected_set, oracle_labels, num_classes, N, device,
    n_select, alpha, reduction,
) -> List[int]:
    picks: List[int] = []
    for _ in tqdm(range(n_select), desc="graph_sargraph laplace+resistance (per_point)"):
        scores = _round_scores_laplace_plus_resistance(
            W, Z_res, selected_indices + picks, oracle_labels, num_classes, N, device,
            alpha, reduction,
        )
        unlabeled = _unlabeled_array(N, selected_set)
        unlabeled_t = torch.as_tensor(unlabeled, device=device, dtype=torch.long)
        best_idx = int(unlabeled[int(torch.argmax(scores[unlabeled_t]).item())])
        picks.append(best_idx)
        selected_set.add(best_idx)
    return picks


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

@register_sampler("graph_sargraph")
def graph_sargraph_sampling(**kwargs) -> List[int]:
    dino_np = np.asarray(kwargs["image_embeddings"], dtype=np.float32)
    cell_raw = kwargs.get("nucleus_embeddings")
    cell_np = None if cell_raw is None else np.asarray(cell_raw, dtype=np.float32)
    rel_raw = kwargs.get("nucleus_reliability")
    reliability = None if rel_raw is None else np.asarray(rel_raw, dtype=np.float32)
    oracle_labels = np.asarray(kwargs["oracle_labels"])
    num_classes = int(kwargs["num_classes"])
    max_budget = int(kwargs["max_budget"])
    device = kwargs["device"]

    acquisition_variant = kwargs.get("acquisition_variant", "laplace_margin")
    if acquisition_variant not in VALID_VARIANTS:
        raise ValueError(
            f"Unknown acquisition_variant={acquisition_variant!r}, "
            f"expected one of {sorted(VALID_VARIANTS)}"
        )

    reducer = kwargs.get("reducer", "vae")
    k = int(kwargs.get("k", 20))
    kernel = kwargs.get("kernel", "asymmetric")
    chunk_size = int(kwargs.get("chunk_size", 2000))
    latent_dim = int(kwargs.get("vae_latent_dim", 32))
    hidden = tuple(kwargs.get("vae_hidden", (512, 128)))
    vae_epochs = int(kwargs.get("vae_epochs", 100))
    vae_lr = float(kwargs.get("vae_lr", 1e-3))
    vae_batch_size = int(kwargs.get("vae_batch_size", 512))
    vae_beta = float(kwargs.get("vae_beta", 1e-2))
    probe_epochs = int(kwargs.get("probe_epochs", 50))
    probe_lr = float(kwargs.get("probe_lr", 1e-3))
    ppr_damping = float(kwargs.get("ppr_damping", 0.85))
    ppr_alpha = float(kwargs.get("ppr_alpha", 0.5))
    res_alpha = float(kwargs.get("resistance_alpha", 0.5))
    res_reduction = kwargs.get("resistance_reduction", "min")
    n_sketch = int(kwargs.get("resistance_n_sketch", 200))
    seed = int(kwargs.get("resistance_seed", 0))
    per_point = bool(kwargs.get("per_point", False))

    N = dino_np.shape[0]
    B = min(max_budget, N)
    step_budget = max(1, int(0.2 * B))

    need_resistance = acquisition_variant == "laplace_plus_resistance"
    cfg_key = (reducer, k, kernel, latent_dim, hidden, vae_epochs, vae_lr,
               vae_batch_size, vae_beta, need_resistance, n_sketch, seed)
    W, Z_res = _build_graph_cached(
        dino_np, cell_np, reliability, device, cfg_key,
        reducer=reducer, latent_dim=latent_dim, hidden=hidden, vae_epochs=vae_epochs,
        vae_lr=vae_lr, vae_batch_size=vae_batch_size, vae_beta=vae_beta, k=k,
        kernel=kernel, chunk_size=chunk_size, need_resistance=need_resistance,
        n_sketch=n_sketch, seed=seed,
    )

    dino_features_norm = F.normalize(
        torch.as_tensor(dino_np, device=device, dtype=torch.float32), p=2, dim=1
    )
    dino_norm_np = (
        dino_features_norm.cpu().numpy()
        if acquisition_variant == "uherding_swap_coverage" else None
    )

    selected_indices: List[int] = []
    selected_set: Set[int] = set()

    uherding_state = None
    if acquisition_variant == "uherding_swap_uncertainty":
        uherding_state = {
            "sigma": _bootstrap_dense_sigma(dino_features_norm, device),
            "k_running": torch.zeros(N, device=device, dtype=torch.float32),
        }

    laplace_family = ("laplace_margin", "laplace_plus_ppr", "laplace_plus_resistance")

    while len(selected_indices) < B:
        current_need = min(step_budget, B - len(selected_indices))

        if acquisition_variant in laplace_family and not _has_two_classes(
            oracle_labels, selected_indices
        ):
            # Cold start: Laplace learning needs >=2 labeled classes. Use the
            # same genuine facility-location greedy over the real graph that
            # graph_deuce uses for its round 1, NOT an all-tied argmax (which
            # would silently reduce to "pick the lowest index").
            ones = torch.ones(N, device=device, dtype=torch.float32)
            chosen = greedy_coverage_sparse(W, ones, current_need, selected_set, device)

        elif acquisition_variant == "laplace_margin" and per_point:
            chosen = _select_points_per_point_laplace_margin(
                W, selected_indices, selected_set, oracle_labels, num_classes, N,
                device, current_need,
            )

        elif acquisition_variant == "laplace_margin":
            unlabeled_indices = _unlabeled_array(N, selected_set).tolist()
            scores = _round_scores_laplace_margin(
                W, selected_indices, oracle_labels, num_classes, N, device,
            )
            chosen = _select_batch_with_discount(scores, unlabeled_indices, current_need, W, device)

        elif acquisition_variant == "laplace_plus_ppr" and per_point:
            chosen = _select_points_per_point_laplace_plus_ppr(
                W, selected_indices, selected_set, oracle_labels, num_classes, N,
                device, current_need, ppr_damping, ppr_alpha,
            )

        elif acquisition_variant == "laplace_plus_ppr":
            unlabeled_indices = _unlabeled_array(N, selected_set).tolist()
            scores = _round_scores_laplace_plus_ppr(
                W, selected_indices, oracle_labels, num_classes, N, device,
                ppr_damping, ppr_alpha,
            )
            chosen = _select_batch_with_discount(scores, unlabeled_indices, current_need, W, device)

        elif acquisition_variant == "laplace_plus_resistance" and per_point:
            chosen = _select_points_per_point_laplace_plus_resistance(
                W, Z_res, selected_indices, selected_set, oracle_labels, num_classes,
                N, device, current_need, res_alpha, res_reduction,
            )

        elif acquisition_variant == "laplace_plus_resistance":
            unlabeled_indices = _unlabeled_array(N, selected_set).tolist()
            scores = _round_scores_laplace_plus_resistance(
                W, Z_res, selected_indices, oracle_labels, num_classes, N, device,
                res_alpha, res_reduction,
            )
            chosen = _select_batch_with_discount(scores, unlabeled_indices, current_need, W, device)

        elif acquisition_variant == "uherding_swap_uncertainty":
            chosen = _run_uherding_swap_uncertainty_round(
                dino_features_norm, W, oracle_labels, num_classes,
                selected_indices, selected_set, uherding_state,
                current_need, chunk_size, device,
            )

        else:  # "uherding_swap_coverage"
            # L2-NORMALIZED features, not raw `dino_np`: `uncertainty_herding.py`
            # trains its probe on `norm_embeddings = F.normalize(features)`
            # (lines 133, 191-201). This variant's whole claim is that the
            # uncertainty side is UNCHANGED from UHerding, and feeding raw
            # features changes the logit scale, which moves the ECE-calibrated
            # temperature and therefore the margin ranking.
            U = _round_uncertainty_uherding_original(
                dino_norm_np, oracle_labels, selected_indices, num_classes,
                probe_epochs, probe_lr, device,
            )
            chosen = greedy_coverage_sparse(W, U, current_need, selected_set, device)

        if not chosen:
            break
        selected_indices.extend(chosen)
        selected_set.update(chosen)

    del dino_features_norm
    clear_memory()
    return selected_indices
