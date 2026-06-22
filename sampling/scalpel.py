"""SCALPEL v6 — Stain<->Morphology Cross-View Evidential Conflict (pathology-specific).

Motivation (2026-06-23)
-----------------------
v1-v5 borrowed SaE (VLM evidence) + UHerding (coverage). But (a) raw/zero-shot
PLIP evidence lives in the wrong space and is weak on pathology, and (b) the
"PLIP vs DINOv2" idea is NOT pathology-specific — it transfers verbatim to
natural images (CLIP vs DINOv2). A reviewer asks: why pathology?

v6 anchors the novelty on the ONE property unique to H&E histopathology: STAIN.
Stain is a nuisance factor *confounded* with class — a good cold-start sample is
one where the model has not yet separated biology (morphology) from stain (color).

Two complementary, label-trained evidential views of each patch:
  * Morphology view (M): DINOv2 features -> linear probe -> posterior p_M.
      "What does the tissue STRUCTURE look like?"  (this is the eval space)
  * Stain view (S): H&E color-deconvolution descriptor -> linear probe -> p_S.
      "What does the COLOR/STAIN look like?"

New acquisition signal = Dempster-Shafer CONFLICT between the two evidential
opinions:  C(x) = (1-u_M)(1-u_S) - <b_M, b_S>.  C is high exactly when BOTH views
are confident yet point to DIFFERENT classes — i.e. color says one thing but
structure says another. Labelling those teaches the model to stop relying on
stain shortcuts. This signal is meaningless for natural images (no stain) -> it
does not transfer, unlike PLIP-vs-DINOv2.

Round-based (T rounds, like SaE T=5):
  * Round 1 (cold): pure MaxHerding coverage in DINOv2.
  * Rounds 2..T: train both probes on revealed labels; schedule
        explore (morphology vacuity, cover unknowns)  ->  reconcile (cross-view conflict).
    Coverage stays as submodular batch-diversity (DINOv2).

Ablation: `use_conflict: false` -> vacuity-only weighting.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from . import register_sampler


# ---------------------------------------------------------------------------
# Stain feature extraction — H&E color deconvolution (Ruifrok-Johnston)
# ---------------------------------------------------------------------------

# Standard H&E (+ residual) stain optical-density vectors in RGB (rows = stains).
_HE_OD = np.array([[0.65, 0.70, 0.29],
                   [0.07, 0.99, 0.11],
                   [0.27, 0.57, 0.78]], dtype=np.float64)
_HE_OD = _HE_OD / np.linalg.norm(_HE_OD, axis=1, keepdims=True)
_DECONV = np.linalg.inv(_HE_OD).astype(np.float32)   # OD(P,3) @ _DECONV -> concentrations(P,3)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def extract_stain_features(loader, device: torch.device,
                           mean=_IMAGENET_MEAN, std=_IMAGENET_STD,
                           down: int = 64) -> np.ndarray:
    """Per-image H&E stain descriptor from a (ImageNet-normalised) image loader.

    Recovers raw RGB by inverting the loader normalisation, deconvolves into
    Hematoxylin / Eosin concentration channels, and summarises each patch by a
    compact stain/appearance descriptor (concentration + colour moments).
    """
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    deconv = torch.tensor(_DECONV, device=device)            # (3,3)

    feats: List[np.ndarray] = []
    for imgs, _ in tqdm(loader, desc="Stain feature extraction"):
        imgs = imgs.to(device, non_blocking=True)
        rgb = (imgs * std_t + mean_t).clamp(0.0, 1.0)        # raw RGB in [0,1]
        if down and rgb.shape[-1] > down:
            rgb = F.interpolate(rgb, size=(down, down), mode="bilinear", align_corners=False)
        B = rgb.shape[0]

        I255 = rgb * 255.0
        od = -torch.log10((I255 + 1.0) / 256.0).clamp(min=0.0)   # (B,3,H,W)
        od_flat = od.permute(0, 2, 3, 1).reshape(B, -1, 3)        # (B,P,3)
        C = torch.clamp(od_flat @ deconv, min=0.0)                # (B,P,3) [H,E,res]

        rgb_flat = rgb.permute(0, 2, 3, 1).reshape(B, -1, 3)      # (B,P,3)
        sat = rgb_flat.max(-1).values - rgb_flat.min(-1).values   # (B,P)
        od_mag = od_flat.sum(-1)                                  # (B,P)
        tissue = (od_mag > 0.15).float().mean(1, keepdim=True)    # (B,1)

        desc = torch.cat([
            C.mean(1), C.std(1),               # H/E/res concentration mean+std (6)
            rgb_flat.mean(1), rgb_flat.std(1), # colour mean+std (6)
            sat.mean(1, keepdim=True),         # mean saturation (1)
            tissue,                            # tissue fraction (1)
        ], dim=1)                              # -> (B, 14)
        feats.append(desc.cpu().numpy().astype(np.float32))
        del imgs, rgb, od, od_flat, C, rgb_flat, sat, od_mag, desc
    clear_memory()
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Evidential helpers — Dirichlet opinion + Dempster-Shafer cross-view conflict
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


def _opinion(p: np.ndarray, kappa: float) -> Tuple[np.ndarray, np.ndarray]:
    """From a calibrated posterior, build a Dirichlet opinion.

    λ(x)=κ/(H[p]+0.1) ties evidence to probe certainty; α_k=λ·p_k+1.
    Returns (vacuity u=L/S, belief masses b_k=(α_k-1)/S).
    """
    p = _norm_rows(p)
    L = p.shape[1]
    lam = kappa / (_entropy(p) + 0.1)
    S = lam + L
    u = (L / S).astype(np.float32)
    b = (lam[:, None] * p) / S[:, None]
    return u, b.astype(np.float32)


def _cross_conflict(u_m, b_m, u_s, b_s) -> np.ndarray:
    """Dempster-Shafer conflict between two opinions: mass on contradictory classes.

    C = Σ_{i≠j} b_m_i·b_s_j = (1-u_m)(1-u_s) − Σ_k b_m_k·b_s_k  ∈ [0,1).
    High only when BOTH views are confident (low vacuity) and point to
    DIFFERENT classes.
    """
    agree = (b_m * b_s).sum(axis=1)
    conflict = (1.0 - u_m) * (1.0 - u_s) - agree
    return np.clip(conflict, 0.0, None).astype(np.float32)


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
    """Pick `n_select` points maximising Σ_n W_n·max(K(n,i) − K_n[n], 0).

    Mutates K_n and selected_set in place; returns the new picks.
    """
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
# Main sampling function — iterative, round-based, cross-view conflict
# ---------------------------------------------------------------------------

@register_sampler("scalpel")
def scalpel_sampling(**kwargs) -> List[int]:
    dino_np: np.ndarray   = kwargs["image_embeddings"]   # (N, 768) DINOv2 — morphology + coverage
    stain_np: np.ndarray  = kwargs["stain_features"]     # (N, d_s) H&E stain descriptor
    oracle_labels         = np.asarray(kwargs["oracle_labels"])
    num_classes: int      = kwargs["num_classes"]
    max_budget: int       = kwargs["max_budget"]
    device: torch.device  = kwargs["device"]
    chunk_size: int       = kwargs.get("chunk_size", 2000)
    num_rounds: int       = kwargs.get("num_rounds", 5)
    kappa: float          = kwargs.get("kappa", 1.0)
    probe_epochs: int     = kwargs.get("probe_epochs", 50)
    probe_lr: float       = kwargs.get("probe_lr", 1e-3)
    use_conflict: bool    = kwargs.get("use_conflict", True)
    diag: bool            = kwargs.get("diag", True)
    n_sigma: int          = kwargs.get("n_sigma", 2000)

    from trainer import train_linear

    N = dino_np.shape[0]
    L = num_classes
    B = min(max_budget, N)
    T = max(1, min(num_rounds, B))

    base, rem = divmod(B, T)
    sizes = [base + (1 if r < rem else 0) for r in range(T)]

    # ---- DINOv2 morphology / coverage space ----
    features = F.normalize(
        torch.tensor(dino_np, device=device, dtype=torch.float32), p=2, dim=1
    )
    feat_np = features.cpu().numpy()
    sigma = _adaptive_sigma(features, n_ref=n_sigma)
    K_n = torch.zeros(N, device=device, dtype=torch.float32)

    # ---- stain descriptor: z-score so the linear probe sees comparable scales ----
    stain_z = stain_np.astype(np.float32)
    mu = stain_z.mean(0, keepdims=True)
    sd = stain_z.std(0, keepdims=True) + 1e-6
    stain_z = (stain_z - mu) / sd

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
            y = oracle_labels[sel]

            # Morphology view (eval space) — DINOv2 probe
            probe_m = train_linear(feat_np[sel], y, L, probe_epochs, probe_lr, device)
            p_m = probe_m.predict_proba(feat_np, device)
            u_m, b_m = _opinion(p_m, kappa)
            del probe_m

            # Stain view — H&E colour-deconvolution probe
            probe_s = train_linear(stain_z[sel], y, L, probe_epochs, probe_lr, device)
            p_s = probe_s.predict_proba(stain_z, device)
            u_s, b_s = _opinion(p_s, kappa)
            del probe_s
            clear_memory()

            conflict = _cross_conflict(u_m, b_m, u_s, b_s)

            if diag:
                acc_m = float((p_m.argmax(1) == oracle_labels).mean())
                acc_s = float((p_s.argmax(1) == oracle_labels).mean())
                print(f"[DIAG b={B} r={r}] probe_M acc={acc_m:.3f} | probe_S(stain) acc={acc_s:.3f} "
                      f"| mean_conflict={conflict.mean():.4f} max={conflict.max():.4f}")

            t = r / max(1, T - 1)                                         # explore -> reconcile
            if use_conflict:
                w_np = (1.0 - t) * _minmax(u_m) + t * _minmax(conflict)
            else:
                w_np = _minmax(u_m)
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
