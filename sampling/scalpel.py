"""SCALPEL v8 — Stain-Shortcut-Aware Active Learning (pathology-specific).

Gap this addresses (2026-06-23)
-------------------------------
Every AL baseline treats H&E patches as generic images and selects on feature
geometry or prediction uncertainty — both BLIND to the defining nuisance of
histopathology: H&E stain, which is (i) highly variable across labs/scanners and
(ii) confounded with class. Models exploit stain as a SHORTCUT (well-known in
pathology DL), so a probe can be confident for the wrong reason. No acquisition
function accounts for this.

Idea
----
Two probes trained each round on revealed labels:
  * Morphology probe p_M : DINOv2 features (the eval space). Its margin gives the
    strong uncertainty signal that wins at mid budget (cf. Margin/BADGE).
  * Stain probe p_S : cheap H&E colour-deconvolution descriptor. It classifies
    weakly (stain is a nuisance) — we use it as a STAIN-SHORTCUT DETECTOR.

Stain-shortcut score  s(x) = p_S(x)[argmax_c p_M(x)]  = how much the stain view
supports the morphology prediction. High s = the model can get x "for free" via
the stain shortcut -> low teaching value. So we weight acquisition by
uncertainty that stain does NOT explain:

    W_reconcile(x) = unc_M(x) * (1 - s(x))          (unc_M = 1 - margin(p_M))

Labelling high-W samples forces the model to learn stain-invariant morphology,
which should generalise better on the stain-heterogeneous test set. Round-based
(T rounds): round 1 pure coverage; rounds 2..T schedule explore (morphology
vacuity) -> reconcile (stain-adjusted uncertainty), over DINOv2 submodular
coverage.

Built-in diagnostic tests the core hypothesis: is s(x) higher for samples the
model classifies CORRECTLY than incorrectly? (i.e. is "stain-easy" == "model-easy"?)

Ablation: `use_stain_discount: false` -> plain margin uncertainty (measures the
stain contribution). This is pathology-specific: stain has no meaning for
natural images, so the discount does not transfer.
"""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from . import register_sampler


# ---------------------------------------------------------------------------
# Stain descriptor — H&E colour deconvolution (Ruifrok-Johnston)
# ---------------------------------------------------------------------------

_HE_OD = np.array([[0.65, 0.70, 0.29],
                   [0.07, 0.99, 0.11],
                   [0.27, 0.57, 0.78]], dtype=np.float64)
_HE_OD = _HE_OD / np.linalg.norm(_HE_OD, axis=1, keepdims=True)
_DECONV = np.linalg.inv(_HE_OD).astype(np.float32)       # OD(P,3) @ _DECONV -> concentrations

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def extract_stain_features(loader, device: torch.device,
                           mean=_IMAGENET_MEAN, std=_IMAGENET_STD,
                           down: int = 64) -> np.ndarray:
    """Per-image H&E stain descriptor from an (ImageNet-normalised) image loader.

    Recovers raw RGB, deconvolves into Hematoxylin/Eosin concentration channels,
    and summarises each patch by a compact stain/appearance descriptor.
    """
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    deconv = torch.tensor(_DECONV, device=device)

    feats: List[np.ndarray] = []
    for imgs, _ in tqdm(loader, desc="Stain feature extraction", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        rgb = (imgs * std_t + mean_t).clamp(0.0, 1.0)
        if down and rgb.shape[-1] > down:
            rgb = F.interpolate(rgb, size=(down, down), mode="bilinear", align_corners=False)
        B = rgb.shape[0]

        I255 = rgb * 255.0
        od = -torch.log10((I255 + 1.0) / 256.0).clamp(min=0.0)
        od_flat = od.permute(0, 2, 3, 1).reshape(B, -1, 3)
        C = torch.clamp(od_flat @ deconv, min=0.0)                # (B,P,3) H/E/res

        rgb_flat = rgb.permute(0, 2, 3, 1).reshape(B, -1, 3)
        sat = rgb_flat.max(-1).values - rgb_flat.min(-1).values
        od_mag = od_flat.sum(-1)
        tissue = (od_mag > 0.15).float().mean(1, keepdim=True)

        desc = torch.cat([
            C.mean(1), C.std(1),                 # H/E/res concentration mean+std (6)
            rgb_flat.mean(1), rgb_flat.std(1),   # colour mean+std (6)
            sat.mean(1, keepdim=True),           # mean saturation (1)
            tissue,                              # tissue fraction (1)
        ], dim=1)
        feats.append(desc.cpu().numpy().astype(np.float32))
        del imgs, rgb, od, od_flat, C, rgb_flat, sat, od_mag, desc
    clear_memory()
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Uncertainty helpers
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
    """Dirichlet vacuity; λ=κ·L/(H+0.1), u=L/(λ+L). Explore signal (cold-start)."""
    p = _norm_rows(p)
    L = p.shape[1]
    lam = kappa * L / (_entropy(p) + 0.1)
    return (L / (lam + L)).astype(np.float32)


def _margin_uncertainty(p: np.ndarray) -> np.ndarray:
    """1 - (top1 - top2) of the posterior. High = decision boundary."""
    ps = np.sort(_norm_rows(p), axis=1)
    return (1.0 - (ps[:, -1] - ps[:, -2])).astype(np.float32)


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
# Main sampling function — iterative, round-based, stain-shortcut-aware
# ---------------------------------------------------------------------------

@register_sampler("scalpel")
def scalpel_sampling(**kwargs) -> List[int]:
    dino_np: np.ndarray   = kwargs["image_embeddings"]    # (N, 768) DINOv2 (morphology + coverage)
    stain_np: np.ndarray  = kwargs["stain_features"]      # (N, d_s) H&E stain descriptor
    oracle_labels         = np.asarray(kwargs["oracle_labels"])
    num_classes: int      = kwargs["num_classes"]
    max_budget: int       = kwargs["max_budget"]
    device: torch.device  = kwargs["device"]
    chunk_size: int       = kwargs.get("chunk_size", 2000)
    num_rounds: int       = kwargs.get("num_rounds", 5)
    kappa: float          = kwargs.get("kappa", 1.0)
    probe_epochs: int     = kwargs.get("probe_epochs", 50)
    probe_lr: float       = kwargs.get("probe_lr", 1e-3)
    use_stain_discount: bool = kwargs.get("use_stain_discount", True)
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

    # ---- stain descriptor: z-scored ----
    stain_z = stain_np.astype(np.float32)
    stain_z = (stain_z - stain_z.mean(0, keepdims=True)) / (stain_z.std(0, keepdims=True) + 1e-6)

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

            probe_m = train_linear(feat_np[sel], y, L, probe_epochs, probe_lr, device)
            p_m = probe_m.predict_proba(feat_np, device)
            del probe_m

            probe_s = train_linear(stain_z[sel], y, L, probe_epochs, probe_lr, device)
            p_s = probe_s.predict_proba(stain_z, device)
            del probe_s
            clear_memory()

            pred_m = p_m.argmax(1)
            vac = _vacuity(p_m, kappa)                                   # explore
            unc = _margin_uncertainty(p_m)                              # morphology uncertainty
            s_shortcut = p_s[np.arange(N), pred_m].astype(np.float32)   # stain support for p_M's class
            reconcile = unc * (1.0 - s_shortcut) if use_stain_discount else unc

            if diag:
                acc_m = float((pred_m == oracle_labels).mean())
                acc_s = float((p_s.argmax(1) == oracle_labels).mean())
                correct = (pred_m == oracle_labels)
                s_c = float(s_shortcut[correct].mean()) if correct.any() else 0.0
                s_w = float(s_shortcut[~correct].mean()) if (~correct).any() else 0.0
                print(f"[DIAG b={B} r={r}] probe_M acc={acc_m:.3f} probe_S acc={acc_s:.3f} "
                      f"| s_shortcut(correct)={s_c:.3f} (wrong)={s_w:.3f} "
                      f"| mean unc={unc.mean():.3f} reconcile={reconcile.mean():.3f}")

            t = r / max(1, T - 1)                                        # explore -> reconcile
            w_np = (1.0 - t) * _minmax(vac) + t * _minmax(reconcile)
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
