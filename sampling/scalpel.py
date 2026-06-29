"""SCALPEL v7 — Stain-Perturbation Instability (pathology-specific).

Why this version (2026-06-23)
-----------------------------
v6 tried a stain<->morphology cross-view *conflict*. Diagnostics killed it: the
stain probe classifies at 0.10-0.24 acc (chance 0.07) — stain is a NUISANCE, so a
"stain classifier" is inherently weak and can never form a confident, meaningful
second opinion. Cross-view conflict therefore degenerated to ~0.

v7 uses stain the way its nature demands — as a NUISANCE to perturb, not a signal
to classify (this is DropQuery's feature-perturbation idea, but the perturbation
is realistic H&E stain variation):

  * Morphology probe p_M: DINOv2 linear probe trained on revealed labels (eval space).
  * For each image we precompute DINOv2 features of K stain-PERTURBED versions
    (HED stain augmentation: jitter H&E concentrations, re-encode).
  * Stain-instability(x) = how much the probe's prediction DISAGREES across the
    stain-perturbed views (BALD-style mutual information). High instability = the
    model's decision flips with the stain => it relies on a stain shortcut for x
    => labelling x teaches stain-invariant morphology. This needs NO stain
    classifier, so it sidesteps v6's paradox.

Round-based (T rounds): round 1 = pure coverage; rounds 2..T schedule
explore (morphology vacuity) -> reconcile (stain-instability), over a DINOv2
submodular-coverage objective. Stain-instability is meaningless for natural
images (no stain) -> the novelty is genuinely pathology-specific.

Ablation: `use_instability: false` -> vacuity-only weighting.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from . import register_sampler


# ---------------------------------------------------------------------------
# H&E stain augmentation + DINOv2 features of stain-perturbed images
# ---------------------------------------------------------------------------

_HE_OD = np.array([[0.65, 0.70, 0.29],
                   [0.07, 0.99, 0.11],
                   [0.27, 0.57, 0.78]], dtype=np.float64)
_HE_OD = _HE_OD / np.linalg.norm(_HE_OD, axis=1, keepdims=True)
_DECONV = np.linalg.inv(_HE_OD).astype(np.float32)       # OD(P,3) @ _DECONV -> concentrations
_HE_OD_F = _HE_OD.astype(np.float32)                     # concentrations(P,3) @ _HE_OD -> OD

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@torch.inference_mode()
def extract_stain_perturbed_features(
    loader, extractor, device: torch.device,
    K: int = 3, sigma_alpha: float = 0.15, sigma_beta: float = 0.08,
    mean=_IMAGENET_MEAN, std=_IMAGENET_STD,
) -> np.ndarray:
    """DINOv2 features of K independently stain-augmented versions of the pool.

    HED stain augmentation (Tellez et al.): recover raw RGB, deconvolve into H&E
    concentrations, jitter each stain channel multiplicatively (1±σ_α) and
    additively (±σ_β), re-compose to RGB, re-encode with the frozen encoder.
    Returns array of shape (K, N, feat_dim).
    """
    extractor = extractor.to(device).eval()
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    M = torch.tensor(_HE_OD_F, device=device)
    D = torch.tensor(_DECONV, device=device)

    passes: List[np.ndarray] = []
    for k in range(K):
        feats: List[np.ndarray] = []
        for imgs, _ in tqdm(loader, desc=f"Stain-aug DINOv2 pass {k+1}/{K}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            rgb = (imgs * std_t + mean_t).clamp(0.0, 1.0)
            B, _, H, W = rgb.shape

            I255 = rgb * 255.0
            od = -torch.log10((I255 + 1.0) / 256.0).clamp(min=0.0)
            odf = od.permute(0, 2, 3, 1).reshape(B, -1, 3)            # (B,P,3)
            C = odf @ D                                               # concentrations

            alpha = (torch.rand(B, 1, 3, device=device) * 2 - 1) * sigma_alpha + 1.0
            beta = (torch.rand(B, 1, 3, device=device) * 2 - 1) * sigma_beta
            Cp = C * alpha + beta

            odp = torch.clamp(Cp @ M, min=0.0)
            Ip = (256.0 * torch.pow(10.0, -odp) - 1.0).clamp(0.0, 255.0)
            rgbp = (Ip / 255.0).reshape(B, H, W, 3).permute(0, 3, 1, 2)
            normp = (rgbp - mean_t) / std_t

            f = extractor(normp)
            feats.append(f.cpu().numpy().astype(np.float32))
            del imgs, rgb, od, odf, C, Cp, odp, Ip, rgbp, normp, f
        passes.append(np.vstack(feats))
        clear_memory()
    return np.stack(passes, axis=0)                                   # (K, N, feat_dim)


# ---------------------------------------------------------------------------
# Evidential / uncertainty helpers
# ---------------------------------------------------------------------------

def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _norm_rows(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def _vacuity(p: np.ndarray, kappa: float) -> np.ndarray:
    """Dirichlet vacuity from a posterior; λ=κ·L/(H+0.1), u=L/(λ+L)."""
    p = _norm_rows(p)
    L = p.shape[1]
    lam = kappa * L / (_entropy(p) + 0.1)
    return (L / (lam + L)).astype(np.float32)


def _stain_instability(views: List[np.ndarray]) -> np.ndarray:
    """BALD-style disagreement of the probe across stain-perturbed views.

    I = H[mean_v p_v] - mean_v H[p_v]  >= 0. High when each view is confident but
    the prediction CHANGES across stain perturbations (stain-dependent decision).
    """
    P = np.stack([_norm_rows(p) for p in views], axis=0)   # (V, N, L)
    p_bar = P.mean(axis=0)                                  # (N, L)
    h_bar = _entropy(p_bar)                                 # (N,)
    h_mean = np.stack([_entropy(P[v]) for v in range(P.shape[0])], 0).mean(0)
    return np.clip(h_bar - h_mean, 0.0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# Coverage kernel — Gaussian RBF on DINOv2 (the probe / evaluation space)
# ---------------------------------------------------------------------------

def _adaptive_sigma(features: torch.Tensor, n_ref: int = 2000) -> float:
    N = features.shape[0]
    n_ref = min(n_ref, N)
    idx = np.random.choice(N, n_ref, replace=False)
    ref = features[idx]
    sim = torch.matmul(ref, ref.T)
    dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sim, min=0.0))
    dist.fill_diagonal_(float("inf"))
    tril_idx = torch.tril_indices(n_ref, n_ref, offset=-1)
    lower = dist[tril_idx[0], tril_idx[1]]
    return max(lower.median().item(), 1e-4)


def _k_gaussian(row: torch.Tensor, col: torch.Tensor, sigma: float) -> torch.Tensor:
    cos_sim = torch.matmul(row, col.T)
    return torch.exp(-torch.clamp(1.0 - cos_sim, min=0.0) / (sigma ** 2))


def _k_col(features: torch.Tensor, best_feat: torch.Tensor, sigma: float, chunk_size: int) -> torch.Tensor:
    N = features.shape[0]
    col = torch.empty(N, device=features.device, dtype=torch.float32)
    for ns in range(0, N, chunk_size):
        ne = min(ns + chunk_size, N)
        col[ns:ne] = _k_gaussian(features[ns:ne], best_feat, sigma).squeeze(1)
    return col


def _greedy_coverage_batch(
    features: torch.Tensor, W: torch.Tensor, K_n: torch.Tensor, sigma: float,
    n_select: int, selected_set: set, chunk_size: int,
) -> List[int]:
    """Pick `n_select` points maximising Σ_n W_n·max(K(n,i) − K_n[n], 0)."""
    N = features.shape[0]
    picks: List[int] = []
    for _ in range(n_select):
        best_idx, best_score = -1, -float("inf")
        for cs in range(0, N, chunk_size):
            ce = min(cs + chunk_size, N)
            cand = features[cs:ce]
            delta_cov = torch.zeros(ce - cs, device=features.device, dtype=torch.float32)
            for ns in range(0, N, chunk_size):
                ne = min(ns + chunk_size, N)
                k = _k_gaussian(features[ns:ne], cand, sigma)
                gain = torch.clamp(k - K_n[ns:ne].unsqueeze(1), min=0.0)
                delta_cov += (W[ns:ne].unsqueeze(1) * gain).sum(0)
                del k, gain
            for si in selected_set:
                if cs <= si < ce:
                    delta_cov[si - cs] = -float("inf")
            local_best = int(torch.argmax(delta_cov).item())
            if delta_cov[local_best].item() > best_score:
                best_score = delta_cov[local_best].item()
                best_idx = cs + local_best
            del cand, delta_cov
            clear_memory()

        if best_idx < 0 or best_idx in selected_set:
            break
        picks.append(best_idx)
        selected_set.add(best_idx)
        best_k_col = _k_col(features, features[best_idx].unsqueeze(0), sigma, chunk_size)
        K_n.copy_(torch.maximum(K_n, best_k_col))
        del best_k_col
        clear_memory()
    return picks


# ---------------------------------------------------------------------------
# Main sampling function — iterative, round-based, stain-instability
# ---------------------------------------------------------------------------

@register_sampler("scalpel")
def scalpel_sampling(**kwargs) -> List[int]:
    dino_np: np.ndarray   = kwargs["image_embeddings"]    # (N, 768) DINOv2 (morphology + coverage)
    aug_np: np.ndarray    = kwargs["stain_aug_features"]  # (K, N, 768) stain-perturbed DINOv2
    oracle_labels         = np.asarray(kwargs["oracle_labels"])
    num_classes: int      = kwargs["num_classes"]
    max_budget: int       = kwargs["max_budget"]
    device: torch.device  = kwargs["device"]
    chunk_size: int       = kwargs.get("chunk_size", 2000)
    num_rounds: int       = kwargs.get("num_rounds", 5)
    kappa: float          = kwargs.get("kappa", 1.0)
    probe_epochs: int     = kwargs.get("probe_epochs", 50)
    probe_lr: float       = kwargs.get("probe_lr", 1e-3)
    use_instability: bool = kwargs.get("use_instability", True)
    diag: bool            = kwargs.get("diag", True)
    n_sigma: int          = kwargs.get("n_sigma", 2000)

    from trainer import train_linear

    N = dino_np.shape[0]
    L = num_classes
    B = min(max_budget, N)
    T = max(1, min(num_rounds, B))

    base, rem = divmod(B, T)
    sizes = [base + (1 if r < rem else 0) for r in range(T)]

    # ---- DINOv2 morphology / coverage space (L2-normalised) ----
    features = F.normalize(
        torch.tensor(dino_np, device=device, dtype=torch.float32), p=2, dim=1
    )
    feat_np = features.cpu().numpy()
    sigma = _adaptive_sigma(features, n_ref=n_sigma)
    K_n = torch.zeros(N, device=device, dtype=torch.float32)

    # stain-perturbed feature sets, normalised the same way as feat_np
    K_aug = aug_np.shape[0]
    aug_norm = [
        (aug_np[k] / (np.linalg.norm(aug_np[k], axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        for k in range(K_aug)
    ]

    selected_indices: List[int] = []
    selected_set: set = set()

    for r in tqdm(range(T), desc="SCALPEL Rounds"):
        n_select = sizes[r]
        if n_select <= 0:
            continue

        if r == 0 or len(selected_indices) < 2:
            W = torch.ones(N, device=device, dtype=torch.float32)        # cold: pure coverage
        else:
            sel = selected_indices
            probe = train_linear(feat_np[sel], oracle_labels[sel], L, probe_epochs, probe_lr, device)
            p0 = probe.predict_proba(feat_np, device)                    # original prediction
            p_views = [p0] + [probe.predict_proba(aug_norm[k], device) for k in range(K_aug)]
            del probe
            clear_memory()

            vac = _vacuity(p0, kappa)                                    # explore (morphology)
            instab = _stain_instability(p_views)                        # reconcile (stain-dependent)

            if diag:
                acc = float((p0.argmax(1) == oracle_labels).mean())
                print(f"[DIAG b={B} r={r}] probe_M acc={acc:.3f} | u_M={vac.mean():.3f} "
                      f"| stain-instab mean={instab.mean():.4f} max={instab.max():.4f}")

            t = r / max(1, T - 1)                                        # explore -> reconcile
            if use_instability:
                w_np = (1.0 - t) * _minmax(vac) + t * _minmax(instab)
            else:
                w_np = _minmax(vac)
            W = torch.tensor(w_np, device=device, dtype=torch.float32)
            if float(W.max()) <= 0.0:
                W = torch.ones(N, device=device, dtype=torch.float32)

        picks = _greedy_coverage_batch(
            features, W, K_n, sigma, n_select, selected_set, chunk_size,
        )
        selected_indices.extend(picks)
        del W
        clear_memory()
        if len(picks) < n_select:        # pool exhausted
            break

    del features, K_n
    clear_memory()
    return selected_indices
