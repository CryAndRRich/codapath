"""Nucleus-aware iterative AL with CellViT embedding used for COVERAGE.

Companion to `sampling/nucleus_al.py`, which uses CellViT for uncertainty on a
different branch of this project. Here uncertainty is fixed and simple (plain
margin on a DINOv2+Linear probe); the experimental axis is which feature space
the coverage kernel operates in: `dino` | `cellvit` | `concat`. See DESIGN.md
for the full rationale.

Two departures from UHerding (arXiv:2412.20644) are deliberate and easy to
mistake for bugs:

1. Sigma for rounds >= 2 is NOT `_adaptive_sigma` from scalpel.py. That
   function is a pool-subsample bootstrap called once per run; it is not
   UHerding's radius adaptation (`sigma* = min pairwise distance on the
   LABELED set`, recomputed every round, shrinking as the labeled set grows —
   Proposition 4 of the paper). `_labeled_min_sigma` implements that. Round 1
   has no labeled set, so it still bootstraps via `_adaptive_sigma`.
2. `Score(i) = minmax(Uncertainty)(i) * minmax(Coverage)(i)` departs from
   UHerding's UCoverage (`sum_n U(x_n) * max(K(n,i) - K_n, 0)`, where
   uncertainty weights every pool point n INSIDE the sum, not the candidate i
   itself). Only the coverage machinery (kernel, greedy, radius adaptation) is
   faithful UHerding; the final combination is a self-designed hybrid.

Invariants this module must preserve. Every one of them was violated by an
earlier revision, and every violation was SILENT — no crash, full budget
returned, results that merely looked plausible:

* Every coverage feature row is L2-normalized. `_k_gaussian` and
  `_adaptive_sigma` read `A @ B.T` AS a cosine, so a non-unit row makes
  `1 - cos` negative (clamped to 0 -> kernel pinned at 1 for most pairs) and
  drives `_adaptive_sigma` toward its 1e-4 floor.
* Sigma is always a distance (`sqrt(2 - 2cos)`), never a squared distance
  (`1 - cos`), because the kernel squares it again. Feeding `1 - cos` sharpens
  the exponent by a factor `2 / (1 - cos)` and collapses coverage to zero.
* `K_n` is recomputed from scratch whenever sigma changes, i.e. every round.
  A running max carried across rounds mixes kernel values from different
  bandwidths and systematically over-subtracts the marginal gain.
* Patches with no detected nucleus are imputed, not left as zero vectors. A
  zero row has cosine 0 with EVERY other row, including other zero rows, so the
  whole missing group degenerates into one artificial point sitting at a
  constant kernel value `exp(-1/sigma^2)` from the entire pool. That is not a
  neutral placeholder — it distorts the greedy in a direction that depends on
  sigma and on the pool. Mean imputation puts them at the centre of the valid
  vectors instead, the defensible reading of "this patch carries no nucleus
  information", and under `concat` it keeps the DINO half discriminating
  normally.
* A constant score vector is detected instead of being handed to `_minmax`,
  which returns all-zeros for constant input, which would make `argmax` return
  the lowest remaining index — selection by index order, dressed as coverage.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from trainer import train_linear
from nucleus.uncertainty import margin_uncertainty_from_logits
from . import register_sampler
from .scalpel import _adaptive_sigma, _k_gaussian, _k_col, _minmax


VALID_COVERAGE_SOURCES = {"dino", "cellvit", "concat"}
VALID_MISSING_IMPUTE = {"mean", "zero"}

# A span below this is treated as "carries no ordering information" rather than
# being pushed through _minmax (which maps a constant vector to all-zeros).
_DEGENERATE_EPS = 1e-12


def _labeled_min_sigma(labeled_features: torch.Tensor) -> float:
    """UHerding radius adaptation: sigma* = min pairwise distance over the
    CURRENT labeled set, recomputed every round.

    Uses the SAME distance convention as `_adaptive_sigma` — Euclidean
    `||u - v|| = sqrt(2 - 2cos)` on L2-normalized rows. `1 - cos` is
    `||u - v||^2 / 2`, a squared distance; the kernel squares sigma again, so
    passing a squared distance here over-sharpens the exponent by `2/(1-cos)`
    and zeroes out coverage.

    Unlike `_adaptive_sigma` this is not a pool subsample and not fixed for the
    whole run — see module docstring point 1.
    """
    L = labeled_features.shape[0]
    if L < 2:
        raise ValueError("_labeled_min_sigma requires at least 2 labeled points")
    sim = torch.matmul(labeled_features, labeled_features.T)
    dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sim, min=0.0))
    dist.fill_diagonal_(float("inf"))
    return max(dist.min().item(), 1e-4)


def _build_coverage_features(
    dino_np: np.ndarray,
    nucleus_np: np.ndarray,
    reliability: np.ndarray,
    coverage_source: str,
    missing_impute: str,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """Return `(features, missing_fraction)` with EVERY row L2-normalized.

    - "dino": the DINOv2 CLS token.
    - "cellvit": pooled CellViT cell embedding.
    - "concat": DINO and CellViT each L2-normalized INDEPENDENTLY first (so
      neither backbone's raw magnitude or dimensionality dominates), then
      concatenated and re-normalized. For a patch with both halves present the
      re-normalization is exactly a `1/sqrt(2)` rescale, so
      `cos = (cos_dino + cos_cellvit) / 2` — the intended equal contribution.
      It is NOT optional: without it every row has norm sqrt(2), the dot
      product reaches 2.0, and the kernel stops being a cosine kernel.

    Patches with no detected nucleus (`reliability == 0`) are handled by
    `missing_impute`:
    - "mean" (default): the CellViT half becomes the mean direction of all
      valid CellViT vectors. Such a patch is then maximally typical rather
      than maximally novel, so it competes on its DINO half alone.
    - "zero": the CellViT half stays zero. Kept for ablation only. Under
      "cellvit" this makes those patches orthogonal to everything (cosine 0
      even with each other, so picking one does not cover the next) and under
      "concat" it drops their similarity to full patches by 1/sqrt(2). Either
      way the greedy over-selects them; always read `missing=` in the
      diagnostics before trusting a "zero" run.
    """
    if coverage_source not in VALID_COVERAGE_SOURCES:
        raise ValueError(
            f"Unknown coverage_source={coverage_source!r}, "
            f"expected one of {sorted(VALID_COVERAGE_SOURCES)}"
        )
    if missing_impute not in VALID_MISSING_IMPUTE:
        raise ValueError(
            f"Unknown missing_impute={missing_impute!r}, "
            f"expected one of {sorted(VALID_MISSING_IMPUTE)}"
        )

    dino_t = F.normalize(
        torch.as_tensor(dino_np, device=device, dtype=torch.float32), p=2, dim=1
    )
    if coverage_source == "dino":
        return dino_t, 0.0

    reliability_t = torch.as_tensor(reliability, device=device, dtype=torch.float32)
    valid = reliability_t > 0.0
    missing = ~valid
    missing_frac = float(missing.float().mean().item())

    nucleus_raw = torch.as_tensor(nucleus_np, device=device, dtype=torch.float32)
    nucleus_norm = torch.zeros_like(nucleus_raw)
    if valid.any():
        nucleus_norm[valid] = F.normalize(nucleus_raw[valid], p=2, dim=1)
        if missing_impute == "mean" and missing.any():
            mean_vec = nucleus_norm[valid].mean(dim=0, keepdim=True)
            mean_norm = float(mean_vec.norm(p=2).item())
            if mean_norm > 1e-8:
                nucleus_norm[missing] = mean_vec / mean_norm
            # A ~zero mean direction means the valid vectors cancel out; there
            # is nothing meaningful to impute, so leave the zero rows as-is.

    if coverage_source == "cellvit":
        return nucleus_norm, missing_frac
    return (
        F.normalize(torch.cat([dino_t, nucleus_norm], dim=1), p=2, dim=1),
        missing_frac,
    )


def _running_max_coverage(
    features: torch.Tensor,
    selected: List[int],
    sigma: float,
    chunk_size: int,
) -> torch.Tensor:
    """`K_n = max_{j in selected} k_sigma(x_n, x_j)` for the CURRENT sigma.

    Called once per round because sigma changes per round (radius adaptation).
    Carrying the previous round's running max forward would mix kernel values
    computed at different bandwidths: the old values were produced by a larger
    sigma, so they are larger, and subtracting them under the new sigma
    under-reports every marginal gain. Cost is O(N * |selected|), negligible
    next to the O(N^2) greedy step.
    """
    N = features.shape[0]
    K_n = torch.zeros(N, device=features.device, dtype=torch.float32)
    if not selected:
        return K_n
    sel_idx = torch.as_tensor(selected, device=features.device, dtype=torch.long)
    sel_feat = features[sel_idx]
    for ns in range(0, N, chunk_size):
        ne = min(ns + chunk_size, N)
        K_n[ns:ne] = _k_gaussian(features[ns:ne], sel_feat, sigma).max(dim=1).values
    del sel_feat
    clear_memory()
    return K_n


def _norm_or_none(values_np: np.ndarray) -> Optional[np.ndarray]:
    """minmax to [0,1], or None when the input carries no ordering at all.

    `_minmax` maps a constant vector to all-zeros. Multiplying that into the
    score zeroes every candidate and makes `argmax` return the first remaining
    index, i.e. selection by index order with no error raised. Returning None
    lets the caller treat that factor as neutral instead.
    """
    span = float(values_np.max() - values_np.min())
    if span < _DEGENERATE_EPS:
        return None
    return _minmax(values_np)


def _greedy_coverage_times_uncertainty(
    features: torch.Tensor,
    uncertainty_raw: Optional[np.ndarray],
    K_n: torch.Tensor,
    sigma: float,
    n_select: int,
    selected_set: set,
    chunk_size: int,
) -> Tuple[List[int], Dict[str, float]]:
    """Pick `n_select` points one at a time.

    Each step: compute the PURE coverage marginal gain
    `Coverage(i) = sum_n max(K(n,i) - K_n, 0)` for every candidate (no
    per-pool-point weighting — that is MaxHerding, UHerding with U=1),
    minmax-normalize it over the REMAINING (not-yet-selected) candidates only,
    and — unless `uncertainty_raw` is None (round 1 cold start, no probe yet) —
    multiply elementwise by minmax(uncertainty) over the same remaining set.
    Pick argmax, update the running max `K_n`, repeat.

    Both normalizations are recomputed at every step: coverage's raw values
    genuinely change every step (`K_n` grows), and while uncertainty's raw
    values are fixed within a round, its remaining-candidate reference set is
    not. A factor whose remaining values are constant is dropped (treated as
    neutral) rather than normalized to all-zeros; the count of such steps is
    reported back so a degenerate round cannot pass unnoticed.
    """
    N = features.shape[0]
    uncertainty_t = (
        None if uncertainty_raw is None
        else torch.as_tensor(uncertainty_raw, device=features.device, dtype=torch.float32)
    )
    picks: List[int] = []
    degenerate_cov = 0
    degenerate_unc = 0
    first_span = float("nan")
    last_span = float("nan")

    for _ in range(n_select):
        delta_cov = torch.zeros(N, device=features.device, dtype=torch.float32)
        for cs in range(0, N, chunk_size):
            ce = min(cs + chunk_size, N)
            cand = features[cs:ce]
            gain_sum = torch.zeros(ce - cs, device=features.device, dtype=torch.float32)
            for ns in range(0, N, chunk_size):
                ne = min(ns + chunk_size, N)
                k = _k_gaussian(features[ns:ne], cand, sigma)
                gain = torch.clamp(k - K_n[ns:ne].unsqueeze(1), min=0.0)
                gain_sum += gain.sum(0)
                del k, gain
            delta_cov[cs:ce] = gain_sum
            del cand, gain_sum
        clear_memory()

        # Drop already-selected points from the candidate set with a boolean
        # mask, NOT by writing -inf into delta_cov: -inf would become the min
        # of the array and destroy the minmax rescaling of everything else.
        remaining_mask = torch.ones(N, dtype=torch.bool, device=features.device)
        if selected_set:
            idx = torch.tensor(list(selected_set), device=features.device, dtype=torch.long)
            remaining_mask[idx] = False
        if not remaining_mask.any():
            break
        remaining_indices = remaining_mask.nonzero(as_tuple=True)[0]

        cov_remaining_np = delta_cov[remaining_indices].cpu().numpy()
        span = float(cov_remaining_np.max() - cov_remaining_np.min())
        if not picks:
            first_span = span
        last_span = span

        cov_norm_np = _norm_or_none(cov_remaining_np)
        if cov_norm_np is None:
            degenerate_cov += 1
            score_remaining = torch.ones(
                len(remaining_indices), device=features.device, dtype=torch.float32
            )
        else:
            score_remaining = torch.as_tensor(
                cov_norm_np, device=features.device, dtype=torch.float32
            )

        if uncertainty_t is not None:
            unc_norm_np = _norm_or_none(
                uncertainty_t[remaining_indices].cpu().numpy()
            )
            if unc_norm_np is None:
                degenerate_unc += 1
            else:
                score_remaining = score_remaining * torch.as_tensor(
                    unc_norm_np, device=features.device, dtype=torch.float32
                )

        best_local = int(torch.argmax(score_remaining).item())
        best_idx = int(remaining_indices[best_local].item())

        picks.append(best_idx)
        selected_set.add(best_idx)
        best_k_col = _k_col(features, features[best_idx].unsqueeze(0), sigma, chunk_size)
        K_n.copy_(torch.maximum(K_n, best_k_col))
        del best_k_col
        clear_memory()

    return picks, {
        "degenerate_cov_steps": degenerate_cov,
        "degenerate_unc_steps": degenerate_unc,
        "cov_span_first": first_span,
        "cov_span_last": last_span,
    }


@register_sampler("nucleus_coverage")
def nucleus_coverage_sampling(**kwargs) -> List[int]:
    dino_np = np.asarray(kwargs["image_embeddings"], dtype=np.float32)
    nucleus_np = np.asarray(kwargs["nucleus_embeddings"], dtype=np.float32)
    reliability = np.asarray(kwargs["nucleus_reliability"], dtype=np.float32)
    oracle_labels = np.asarray(kwargs["oracle_labels"])
    num_classes = int(kwargs["num_classes"])
    max_budget = int(kwargs["max_budget"])
    device = kwargs["device"]

    coverage_source = kwargs.get("coverage_source", "dino")
    missing_impute = kwargs.get("missing_impute", "mean")
    num_rounds = int(kwargs.get("num_rounds", 5))
    chunk_size = int(kwargs.get("chunk_size", 2000))
    n_sigma = int(kwargs.get("n_sigma", 2000))
    sigma_floor_ratio = float(kwargs.get("sigma_floor_ratio", 0.25))
    probe_epochs = int(kwargs.get("probe_epochs", 50))
    probe_lr = float(kwargs.get("probe_lr", 1e-3))
    probe_weight_decay = float(kwargs.get("probe_weight_decay", 1e-4))
    diag = bool(kwargs.get("diag", True))

    if coverage_source not in VALID_COVERAGE_SOURCES:
        raise ValueError(
            f"Unknown coverage_source={coverage_source!r}, "
            f"expected one of {sorted(VALID_COVERAGE_SOURCES)}"
        )
    if missing_impute not in VALID_MISSING_IMPUTE:
        raise ValueError(
            f"Unknown missing_impute={missing_impute!r}, "
            f"expected one of {sorted(VALID_MISSING_IMPUTE)}"
        )
    if len(dino_np) != len(nucleus_np) or len(dino_np) != len(reliability):
        raise ValueError("DINO, nucleus, and reliability arrays must align by patch")
    if not 0.0 <= sigma_floor_ratio < 1.0:
        raise ValueError("sigma_floor_ratio must lie in [0, 1)")

    N = len(dino_np)
    B = min(max_budget, N)
    T = max(1, min(num_rounds, B))
    base, rem = divmod(B, T)
    sizes = [base + (1 if r < rem else 0) for r in range(T)]

    coverage_features, missing_frac = _build_coverage_features(
        dino_np, nucleus_np, reliability, coverage_source, missing_impute, device
    )
    dino_feat_np = F.normalize(
        torch.as_tensor(dino_np, device=device, dtype=torch.float32), p=2, dim=1
    ).cpu().numpy()

    # Bootstrap radius: the only sigma available while the labeled set is empty,
    # and afterwards the reference scale for the floor below.
    sigma_pool = _adaptive_sigma(coverage_features, n_ref=n_sigma)
    sigma_floor = sigma_floor_ratio * sigma_pool

    selected: List[int] = []
    selected_set: set = set()

    for r in tqdm(range(T), desc="NUCLEUS-COVERAGE Rounds"):
        n_select = sizes[r]
        if n_select <= 0:
            continue
        round_start = time.time()

        if r == 0 or len(selected) < 2:
            sigma = sigma_pool
            sigma_labeled = float("nan")
            uncertainty_raw = None
            probe_acc = float("nan")
        else:
            sel = selected
            y = oracle_labels[sel]
            probe = train_linear(
                dino_feat_np[sel], y, num_classes, probe_epochs, probe_lr, device,
                weight_decay=probe_weight_decay,
            )
            logits = probe.predict_logits(dino_feat_np, device)
            uncertainty_raw = margin_uncertainty_from_logits(logits)
            probe_acc = float((logits.argmax(1) == oracle_labels).mean()) if diag else float("nan")
            del probe, logits
            sel_idx = torch.as_tensor(sel, device=device, dtype=torch.long)
            sigma_labeled = _labeled_min_sigma(coverage_features[sel_idx])
            # min-pairwise collapses toward 0 as soon as two labeled patches are
            # near-duplicates (common in tiled pathology pools); the kernel then
            # returns 0 everywhere and coverage stops ranking anything.
            sigma = max(sigma_labeled, sigma_floor)
            clear_memory()

        # Sigma just changed, so the previous round's running max is stale.
        K_n = _running_max_coverage(coverage_features, selected, sigma, chunk_size)

        if diag:
            floored = (
                not math.isnan(sigma_labeled) and sigma > sigma_labeled + 1e-12
            )
            unc_txt = (
                "off (cold start)" if uncertainty_raw is None
                else f"mean={float(np.mean(uncertainty_raw)):.3f} probe_acc={probe_acc:.3f}"
            )
            sigma_txt = (
                f"sigma={sigma:.4f} (pool bootstrap)" if uncertainty_raw is None
                else f"sigma={sigma:.4f} (labeled_min={sigma_labeled:.4f} "
                     f"floor={sigma_floor:.4f}{' [FLOORED]' if floored else ''})"
            )
            print(
                f"[NUC-COV b={B} r={r}] source={coverage_source} impute={missing_impute} "
                f"missing={missing_frac:.1%} | {sigma_txt} | unc {unc_txt}"
            )

        picks, step_diag = _greedy_coverage_times_uncertainty(
            coverage_features, uncertainty_raw, K_n, sigma, n_select, selected_set, chunk_size,
        )
        selected.extend(picks)

        if diag:
            print(
                f"[NUC-COV b={B} r={r}] picked={len(picks)} in {time.time() - round_start:.1f}s "
                f"| cov span first={step_diag['cov_span_first']:.3e} "
                f"last={step_diag['cov_span_last']:.3e} "
                f"| degenerate steps: cov={step_diag['degenerate_cov_steps']} "
                f"unc={step_diag['degenerate_unc_steps']}"
            )
        if step_diag["degenerate_cov_steps"] > 0:
            print(
                f"[NUC-COV WARNING b={B} r={r}] coverage was constant over the "
                f"remaining candidates on {step_diag['degenerate_cov_steps']}/{n_select} "
                "steps and was ignored there. Sigma is probably far too small "
                "(kernel underflow) or every row is the zero vector."
            )

        del K_n
        clear_memory()
        if len(picks) < n_select:        # pool exhausted
            break

    del coverage_features
    clear_memory()
    return selected
