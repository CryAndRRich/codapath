"""
DCoM: Dynamic Coverage & Margin mix.

Selection score: R(x) = S_L · (1 − margin(x)) + (1 − S_L) · ODR_norm(x)

  S_L = competence score = logistic(P_cov; a, k)
  ODR(x) = number of still-uncovered unlabeled points within δ of x
  margin(x) = 1 − (p_1 − p_2)  (from LinearProbe; fixed after warm-up)
  P_cov = fraction of unlabeled pool covered by selected set

Cold-start: S_L = 0 → pure ODR (ProbCover-like diversity).
After warm-up: S_L > 0 → increasingly uncertainty-driven.

Reference: Mishal & Weinshall, arXiv:2407.01804 (2024)
GitHub:    https://github.com/avihu111/TypiClust
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from . import register_sampler


def _estimate_delta(features_np: np.ndarray,
                    num_classes: int,
                    alpha: float = 0.95) -> float:
    """
    Estimate initial δ_0 as the highest δ where purity π(δ) ≥ alpha.
    Purity: fraction of points where all neighbors within δ have same pseudo-label.
    Uses K-means pseudo-labels for efficiency.
    """
    from sklearn.cluster import MiniBatchKMeans

    n = features_np.shape[0]
    km = MiniBatchKMeans(n_clusters=num_classes, n_init="auto",
                         batch_size=5000, random_state=42)
    pseudo = km.fit_predict(features_np)

    # For each point find its nearest differently-labelled neighbour
    # (approximate via per-class distance sampling to avoid O(N^2))
    feats = torch.tensor(features_np, dtype=torch.float32)
    feats = F.normalize(feats, p=2, dim=1)

    purity_radius = np.full(n, float("inf"), dtype=np.float32)

    # Process in class pairs to find cross-class NN distances
    unique_labels = np.unique(pseudo)
    for cls in unique_labels:
        idx_cls = np.where(pseudo == cls)[0]
        idx_other = np.where(pseudo != cls)[0]
        if len(idx_other) == 0:
            continue
        f_cls = feats[idx_cls]      # (n_c, D)
        f_other = feats[idx_other]  # (n_o, D)

        # chunk to avoid OOM
        chunk = 2000
        min_dist_cls = torch.full((len(idx_cls),), float("inf"))
        for s in range(0, len(idx_other), chunk):
            e = min(s + chunk, len(idx_other))
            sim = torch.matmul(f_cls, f_other[s:e].T)          # (n_c, e-s)
            dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sim, min=0.0))
            min_dist_cls = torch.minimum(min_dist_cls, dist.min(dim=1).values)
        purity_radius[idx_cls] = min_dist_cls.numpy()

    # π(δ) = fraction of points where purity_radius > δ
    # Binary search for highest δ where π(δ) ≥ alpha
    sorted_r = np.sort(purity_radius)
    # δ_0 = percentile that keeps ≥ alpha fraction of points pure
    thresh_idx = int(np.floor((1.0 - alpha) * n))
    thresh_idx = min(thresh_idx, n - 1)
    delta = float(sorted_r[thresh_idx])
    delta = max(delta, 1e-3)
    return delta


def _competence_score(p_cov: float, a: float = 0.9, k: float = 30.0) -> float:
    """S_L = logistic(P_cov; a, k) — Eq.(6) from DCoM paper."""
    numerator = 1.0 + np.exp(-k * (1.0 - a))
    denominator = 1.0 + np.exp(-k * (p_cov - a))
    return numerator / denominator


@register_sampler("dcom")
def dcom_sampling(**kwargs) -> List[int]:
    """
    DCoM greedy sampler — sliceable (run once at max_budget).

    Uses oracle_labels for an internal warm-up LinearProbe to compute margin scores.
    δ_0 is estimated from data purity or provided explicitly via 'delta' kwarg.
    """
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 50)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]
    # competence logistic parameters (paper defaults differ by dataset size)
    a = kwargs.get("competence_midpoint", 0.9)
    k_steep = kwargs.get("competence_steepness", 30.0)
    # δ can be provided directly or estimated automatically
    delta = kwargs.get("delta", None)

    from trainer import train_linear

    num_samples = image_embeddings.shape[0]
    step_budget = max(num_classes, int(0.2 * max_budget))

    features = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features = F.normalize(features, p=2, dim=1)

    # ── δ estimation ──────────────────────────────────────────────────────────
    if delta is None:
        delta = _estimate_delta(image_embeddings, num_classes, alpha=0.95)

    # ── Build adjacency (sparse): for each point, precompute neighbours within δ ─
    # Store as list of arrays to avoid O(N^2) dense matrix.
    # We use cosine distance: d(x,y) = sqrt(2 - 2·cos) ≤ δ  ↔  cos ≥ 1 - δ^2/2
    cos_thresh = 1.0 - (delta ** 2) / 2.0
    chunk_size = 2000

    # neighbours[i] = set of j where d(i,j) < δ  (excluding i)
    neighbours: List[np.ndarray] = [np.empty(0, dtype=np.int32)] * num_samples
    for cs in tqdm(range(0, num_samples, chunk_size), desc="DCoM: build graph", leave=False):
        ce = min(cs + chunk_size, num_samples)
        chunk = features[cs:ce]                           # (C, D)
        sim = torch.matmul(chunk, features.T)             # (C, N)
        sim[:, cs:ce].fill_diagonal_(-1.0)                # exclude self
        for local_i in range(ce - cs):
            nbrs = torch.where(sim[local_i] >= cos_thresh)[0].cpu().numpy()
            neighbours[cs + local_i] = nbrs.astype(np.int32)
        del chunk, sim
        clear_memory()

    # ── Margin scores (fixed after warm-up) ───────────────────────────────────
    # Initialise to 0 (unknown → no uncertainty signal)
    margin_scores = np.zeros(num_samples, dtype=np.float32)
    probe_trained = False

    # ── Greedy selection ───────────────────────────────────────────────────────
    selected_indices: List[int] = []
    selected_set: set = set()
    covered = np.zeros(num_samples, dtype=bool)   # True once any selected point covers it

    # ODR[i] = |{j ∈ U_uncovered : j ∈ neighbours[i]}|
    odr = np.array([len(nb) for nb in neighbours], dtype=np.float32)

    for step in tqdm(range(max_budget), desc="DCoM Selection"):

        # ── Warm-up: train probe, fix margin scores ───────────────────────────
        if step == step_budget and not probe_trained:
            probe = train_linear(
                image_embeddings[selected_indices],
                oracle_labels[selected_indices],
                num_classes, probe_epochs, probe_lr, device,
            )
            probs = probe.predict_proba(image_embeddings, device)
            sp = np.sort(probs, axis=1)
            margin_scores = sp[:, -1] - sp[:, -2]   # [0,1]: small = uncertain
            del probe
            clear_memory()
            probe_trained = True

        # ── Competence score from current coverage ────────────────────────────
        p_cov = float(covered.sum()) / num_samples
        S_L = _competence_score(p_cov, a=a, k=k_steep)

        # ── Normalised ODR ────────────────────────────────────────────────────
        odr_max = odr.max()
        odr_norm = odr / (odr_max + 1e-8)

        # ── Ranking score R(x) ────────────────────────────────────────────────
        # S_L · (1 - margin)  +  (1 - S_L) · ODR_norm
        uncertainty_term = 1.0 - margin_scores     # high = uncertain
        scores = S_L * uncertainty_term + (1.0 - S_L) * odr_norm

        # Mask already-selected
        for si in selected_set:
            scores[si] = -float("inf")

        best_idx = int(np.argmax(scores))

        selected_indices.append(best_idx)
        selected_set.add(best_idx)

        # ── Update coverage and ODR ───────────────────────────────────────────
        newly_covered = neighbours[best_idx][~covered[neighbours[best_idx]]]
        covered[newly_covered] = True
        covered[best_idx] = True

        # Decrease ODR for all points that had newly_covered as neighbours
        # (approximate: re-count from neighbours lists is expensive, so we
        # track a "covered count" subtraction)
        for nc_idx in newly_covered:
            for i in neighbours[nc_idx]:
                odr[i] = max(0.0, odr[i] - 1.0)
        odr[best_idx] = 0.0

    del features
    clear_memory()
    return selected_indices
