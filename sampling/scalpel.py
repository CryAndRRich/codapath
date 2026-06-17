"""SCALPEL v4 — Iterative Evidential Coverage (round-based, VLM + label-aware).

Design rationale (2026-06-12)
-----------------------------
Earlier SCALPEL versions computed EDL vacuity directly from *raw* zero-shot PLIP
similarity. The parent paper SaE shows precisely that raw VLM similarity is
overconfident/miscalibrated — its core contribution, the Similarity Evidence
Head (SEH), *recalibrates similarity using labels*. Skipping the SEH (v1-v3) is
the exact failure mode SaE was written to fix, and it kept SCALPEL below UHerding
and CODAPath.

v4 is faithful to SaE and adds the missing batch-diversity SaE itself lists as
future work ("combination with coverage/clustering"):

  Per target budget B, selection runs in T rounds of ~B/T (standard pool AL,
  like SaE's T=5):
    * Round 1 (cold, no labels): pure MaxHerding coverage in DINOv2 — diverse seed.
    * Rounds 2..T: reveal labels of everything selected so far and TRAIN AN SEH
      (SaE Eq. 3) to map PLIP similarity -> calibrated evidence strength lambda(x).
      Build the Dirichlet alpha_k = lambda(x)*p_k + 1 (SaE Eq. 4) from the PLIP
      zero-shot posterior p, and decompose it into VACUITY (explore rare/under-
      covered classes) and DISSONANCE (refine decision boundaries, SaE Eqs. 5-6).
      The next batch maximises evidence-weighted submodular coverage gain in
      DINOv2, with SaE's dynamic explore->refine schedule across rounds.

Two spaces, each where it belongs:
  * PLIP (image + text)  -> semantic evidence (SEH-calibrated vacuity/dissonance).
  * DINOv2               -> coverage kernel (the space the linear probe lives in).

When too few labels exist to fit an SEH reliably (very low budgets), lambda falls
back to the uncalibrated proxy kappa/(H+eps) so the round still runs.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

from set_up import clear_memory
from . import register_sampler


# ---------------------------------------------------------------------------
# PLIP feature extraction (called by run.py)
# ---------------------------------------------------------------------------

class PLIPExtractor(nn.Module):
    def __init__(self, model_name: str = "vinid/plip") -> None:
        super().__init__()
        clip = CLIPModel.from_pretrained(model_name)
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection
        del clip
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.vision_model(pixel_values=x).pooler_output
        return self.visual_projection(pooled)  # (B, projection_dim=512)


def extract_plip_text_features(
    class_descriptions: Dict[str, str],
    prompt_templates: List[str],
    class_names: List[str],
    device: torch.device,
    plip_model: str = "vinid/plip",
) -> np.ndarray:
    detailed_descriptions = [class_descriptions.get(cls, cls) for cls in class_names]
    list_prompts = [
        template.format(desc)
        for desc in detailed_descriptions
        for template in prompt_templates
    ]
    num_classes = len(class_names)
    num_templates = len(prompt_templates)

    tokenizer = CLIPTokenizer.from_pretrained(plip_model)
    clip_model = CLIPModel.from_pretrained(plip_model).to(device).eval()
    tokens = tokenizer(
        list_prompts, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        text_out = clip_model.text_model(**tokens)
        emb = clip_model.text_projection(text_out.pooler_output).cpu().numpy()
    del clip_model, tokens
    clear_memory()

    text_embeddings = emb.reshape(num_classes, num_templates, -1).mean(axis=1).astype(np.float32)
    norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    text_embeddings /= np.maximum(norms, 1e-8)
    return text_embeddings


# ---------------------------------------------------------------------------
# Similarity Evidence Head (SaE 3.2.2) — calibrates PLIP similarity with labels
# ---------------------------------------------------------------------------

class _SEH(nn.Module):
    """Dual-branch MLP: (PLIP image emb, similarity vector) -> evidence strength λ>0."""

    def __init__(self, img_dim: int, n_classes: int, hidden: int = 128, p_drop: float = 0.3) -> None:
        super().__init__()
        self.img_branch = nn.Sequential(
            nn.Linear(img_dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p_drop),
        )
        self.sim_branch = nn.Sequential(
            nn.Linear(n_classes, hidden), nn.ReLU(), nn.Dropout(p_drop),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_img: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.img_branch(x_img), self.sim_branch(s)], dim=1)
        return F.softplus(self.fuse(z)).squeeze(-1)  # λ > 0


def _train_seh(
    img_lab: np.ndarray,   # (n, d) PLIP image emb of labeled
    sim_lab: np.ndarray,   # (n, L) similarity vectors of labeled
    p_lab: np.ndarray,     # (n, L) VLM posterior of labeled
    y_lab: np.ndarray,     # (n,)   oracle labels
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
    beta: float = 0.5,
) -> _SEH:
    eps, eps_h = 1e-6, 0.1
    n, d = img_lab.shape
    L = p_lab.shape[1]

    # detached targets from the frozen VLM posterior (SaE Eq. 3)
    l_cls = -np.log(np.clip(p_lab[np.arange(n), y_lab], eps, 1.0))          # difficulty
    H = -(p_lab * np.log(np.clip(p_lab, eps, 1.0))).sum(axis=1)            # entropy
    tgt_inv = torch.tensor(l_cls, device=device, dtype=torch.float32)
    tgt_lam = torch.tensor(1.0 / (H + eps_h), device=device, dtype=torch.float32)

    Xi = torch.tensor(img_lab, device=device, dtype=torch.float32)
    Si = torch.tensor(sim_lab, device=device, dtype=torch.float32)

    seh = _SEH(d, L).to(device)
    opt = torch.optim.Adam(seh.parameters(), lr=lr, weight_decay=1e-3)
    seh.train()
    for _ in range(epochs):
        opt.zero_grad()
        lam = seh(Xi, Si).clamp(1e-3, 50.0)
        loss = F.mse_loss(1.0 / (lam + eps), tgt_inv) + beta * F.mse_loss(lam, tgt_lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(seh.parameters(), 5.0)
        opt.step()
    seh.eval()
    return seh


@torch.no_grad()
def _seh_lambda(seh: _SEH, img_all: np.ndarray, sim_all: np.ndarray,
                device: torch.device, chunk: int = 8192) -> np.ndarray:
    N = img_all.shape[0]
    out = np.empty(N, dtype=np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        xi = torch.tensor(img_all[s:e], device=device, dtype=torch.float32)
        si = torch.tensor(sim_all[s:e], device=device, dtype=torch.float32)
        out[s:e] = seh(xi, si).clamp(1e-3, 50.0).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Evidential uncertainty: Dirichlet from (λ, posterior) — vacuity & dissonance
# ---------------------------------------------------------------------------

def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _evidence(p: np.ndarray, lam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """alpha_k = lam*p_k + 1 (SaE Eq.4); return (vacuity, dissonance) (SaE Eqs.5-6)."""
    eps = 1e-8
    N, L = p.shape
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)

    S = lam + L                                       # Dirichlet strength (N,)
    vacuity = L / S                                   # (N,)

    b = (lam[:, None] * p) / S[:, None]               # belief masses (N, L)
    bi = b[:, :, None]
    bj = b[:, None, :]
    bal = 1.0 - np.abs(bi - bj) / (bi + bj + eps)
    offdiag = 1.0 - np.eye(L, dtype=p.dtype)[None]
    num = (bj * bal * offdiag).sum(axis=2)
    den = (bj * offdiag).sum(axis=2) + eps
    dissonance = (b * num / den).sum(axis=1)          # (N,)

    return vacuity.astype(np.float32), dissonance.astype(np.float32)


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
# Main sampling function — iterative, round-based
# ---------------------------------------------------------------------------

@register_sampler("scalpel")
def scalpel_sampling(**kwargs) -> List[int]:
    dino_np: np.ndarray   = kwargs["image_embeddings"]        # (N, 768) DINOv2 — coverage space
    vlm_np:  np.ndarray   = kwargs["vlm_image_embeddings"]    # (N, 512) PLIP image — evidence
    text_np: np.ndarray   = kwargs["text_embeddings"]         # (L, 512) PLIP text prototypes
    oracle_labels         = np.asarray(kwargs["oracle_labels"])
    num_classes: int      = kwargs["num_classes"]
    max_budget: int       = kwargs["max_budget"]              # target budget B for this call
    device: torch.device  = kwargs["device"]
    chunk_size: int       = kwargs.get("chunk_size", 2000)
    num_rounds: int       = kwargs.get("num_rounds", 5)
    tau: float            = kwargs.get("tau", 0.1)            # VLM posterior temperature
    kappa: float          = kwargs.get("kappa", 1.0)         # fallback λ scale
    beta: float           = kwargs.get("beta", 0.5)          # SEH loss balance
    seh_epochs: int       = kwargs.get("seh_epochs", 200)
    seh_lr: float         = kwargs.get("seh_lr", 1e-3)
    use_dissonance: bool  = kwargs.get("use_dissonance", True)
    n_sigma: int          = kwargs.get("n_sigma", 2000)

    if tau <= 0.0:
        tau = 0.1
    N = dino_np.shape[0]
    L = num_classes
    B = min(max_budget, N)
    T = max(1, min(num_rounds, B))
    seh_min = kwargs.get("seh_min", max(2 * L, 16))

    base, rem = divmod(B, T)
    sizes = [base + (1 if r < rem else 0) for r in range(T)]

    # ---- DINOv2 coverage space ----
    features = F.normalize(
        torch.tensor(dino_np, device=device, dtype=torch.float32), p=2, dim=1
    )
    sigma = _adaptive_sigma(features, n_ref=n_sigma)
    K_n = torch.zeros(N, device=device, dtype=torch.float32)

    # ---- PLIP semantic space: zero-shot posterior p and similarity vectors ----
    vlm = F.normalize(torch.tensor(vlm_np, device=device, dtype=torch.float32), p=2, dim=1)
    txt = F.normalize(torch.tensor(text_np, device=device, dtype=torch.float32), p=2, dim=1)
    sim_all = torch.matmul(vlm, txt.T)                       # (N, L) cosine similarity
    p_all = F.softmax(sim_all / tau, dim=1).cpu().numpy().astype(np.float32)
    sim_np = sim_all.cpu().numpy().astype(np.float32)
    vlm_np_norm = vlm.cpu().numpy().astype(np.float32)
    H_all = -(np.clip(p_all, 1e-8, 1.0) * np.log(np.clip(p_all, 1e-8, 1.0))).sum(axis=1)
    del vlm, txt, sim_all
    clear_memory()

    selected_indices: List[int] = []
    selected_set: set = set()

    for r in tqdm(range(T), desc="SCALPEL Rounds"):
        n_select = sizes[r]
        if n_select <= 0:
            continue

        # ---- evidence weights ------------------------------------------------
        if r == 0 or len(selected_indices) < 2:
            W = torch.ones(N, device=device, dtype=torch.float32)        # cold: pure coverage
        else:
            if len(selected_indices) >= seh_min:
                seh = _train_seh(
                    vlm_np_norm[selected_indices], sim_np[selected_indices],
                    p_all[selected_indices], oracle_labels[selected_indices],
                    device, epochs=seh_epochs, lr=seh_lr, beta=beta,
                )
                lam = _seh_lambda(seh, vlm_np_norm, sim_np, device)
                del seh
                clear_memory()
            else:
                lam = kappa / (H_all + 0.1)                              # uncalibrated fallback

            vac, dis = _evidence(p_all, lam)
            t = r / max(1, T - 1)                                        # 0 -> 1 across rounds
            if use_dissonance:
                w_np = (1.0 - t) * _minmax(vac) + t * _minmax(dis)       # explore -> refine
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
