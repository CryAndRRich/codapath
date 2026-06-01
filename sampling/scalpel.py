"""
SCALPEL: Semantic and Coverage Active Learning with Pathology Evidential Logic

Pipeline (all cold-start, no labeled data required):
  Phase 1 — EDL Vacuity from VLM (semantic prior)
    alpha_{i,c} = exp(cos(v_i^VLM, t_c) / tau)      tau = 0.05
    U_i = L / sum_c alpha_{i,c}                      vacuity in (0, 1]
    P_{i,c} = alpha_{i,c} / S_i                      Dirichlet mean

  Phase 2 — Adaptive sigma for Gaussian kernel (from DINOv2 features)
    sigma = median pairwise distance on subsample

  Phase 3 — Greedy UW-Coverage with K_joint
    K_struct(i,j) = exp(-(1 - cos(v_i^ViT, v_j^ViT)) / sigma^2)
    D_sem(i,j)    = JSD(P_i || P_j) in [0, 1]  (log-2 normalised)
    K_joint(i,j)  = K_struct(i,j) * (1 - D_sem(i,j))  in [0, 1]
    Score(x_i)    = U_i * sum_n U_n * max(K_joint(x_n, x_i) - K_n, 0)

Novelty vs literature:
  K_joint is a product kernel combining DINOv2 structural similarity with
  VLM-JSD semantic divergence — not present in UHerding, MaxHerding, or CODAPath.

References:
  EDL        — Sensoy et al. (2018)
  SaE        — arXiv:2602.18867 (EDL from VLM, warm-start)
  UHerding   — arXiv:2412.20644 (UCoverage framework, Gaussian kernel)
  CODAPath   — dual VLM, cold-start pathology AL
"""
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPTokenizer, CLIPTextModel

from set_up import clear_memory
from . import register_sampler

_LN2 = float(np.log(2.0))


# ---------------------------------------------------------------------------
# PLIP single-VLM extractors (SCALPEL's semantic prior)
# ---------------------------------------------------------------------------

class PLIPExtractor(nn.Module):
    """Frozen PLIP visual encoder → 512-dim pooler_output."""

    def __init__(self, model_name: str = "vinid/plip") -> None:
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values=x).pooler_output  # (B, 512)


def extract_plip_text_features(
    class_descriptions: Dict[str, str],
    prompt_templates: List[str],
    class_names: List[str],
    device: torch.device,
    plip_model: str = "vinid/plip",
) -> np.ndarray:
    """
    Single-VLM text prototype extraction for SCALPEL.
    Each class: K templates → encode → average → L2-normalise.
    Returns: (num_classes, d_plip), float32, L2-normalised.

    Differs from CODAPath's extract_text_features which concatenates
    PLIP + BiomedBERT — SCALPEL uses a single embedding space so that
    cosine(v_i^VLM, t_c) is meaningful for Dirichlet evidence scaling.
    """
    detailed_descriptions = [class_descriptions.get(cls, cls) for cls in class_names]
    list_prompts = [
        template.format(desc)
        for desc in detailed_descriptions
        for template in prompt_templates
    ]
    num_classes = len(class_names)
    num_templates = len(prompt_templates)

    tokenizer = CLIPTokenizer.from_pretrained(plip_model)
    text_model = CLIPTextModel.from_pretrained(plip_model).to(device).eval()

    tokens = tokenizer(
        list_prompts, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        emb = text_model(**tokens).pooler_output.cpu().numpy()  # (num_classes*K, d)

    del text_model, tokens
    clear_memory()

    # Reshape → (num_classes, K, d) → mean over K → L2-normalise
    text_embeddings = emb.reshape(num_classes, num_templates, -1).mean(axis=1).astype(np.float32)
    norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    text_embeddings /= np.maximum(norms, 1e-8)
    return text_embeddings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_edl(
    vlm_img: torch.Tensor,   # (N, d) L2-normalised
    text_emb: torch.Tensor,  # (L, d) L2-normalised
    tau: float,
) -> tuple:
    """
    Returns U (N,) and P (N, L) — vacuity and Dirichlet means.
    Uses temperature scaling alpha = exp(cos/tau) to avoid the
    range compression of naive max(cos,0)+1 (alpha in [1,2]).
    """
    sim = torch.matmul(vlm_img, text_emb.T)        # (N, L)
    alpha = torch.exp(sim / tau)                    # (N, L), > 0
    S = alpha.sum(dim=1, keepdim=True)              # (N, 1)
    L = text_emb.shape[0]
    U = L / S.squeeze(1)                            # (N,), in (0, 1]
    P = alpha / S                                   # (N, L)
    return U, P


def _adaptive_sigma(features: torch.Tensor, n_ref: int = 2000) -> float:
    """
    Median pairwise L2 distance on a random subsample.
    For L2-normalised v: ||v_i - v_j||^2 = 2(1 - cos(v_i, v_j)).
    """
    N = features.shape[0]
    n_ref = min(n_ref, N)
    idx = np.random.choice(N, n_ref, replace=False)
    ref = features[idx]                              # (n_ref, D)
    sim = torch.matmul(ref, ref.T)                   # (n_ref, n_ref)
    dist_sq = torch.clamp(2.0 - 2.0 * sim, min=0.0)
    dist = torch.sqrt(dist_sq)
    # Exclude diagonal (self-distance = 0)
    dist.fill_diagonal_(float("inf"))
    # Lower-triangle values for median (avoid counting each pair twice)
    tril_idx = torch.tril_indices(n_ref, n_ref, offset=-1)
    lower = dist[tril_idx[0], tril_idx[1]]
    sigma = lower.median().item()
    return max(sigma, 1e-4)


def _k_struct(
    row_dino: torch.Tensor,  # (n, D)
    col_dino: torch.Tensor,  # (c, D)
    sigma: float,
) -> torch.Tensor:
    """Gaussian structural kernel: exp(-(1-cos)/sigma^2) → (n, c) in [0,1]."""
    cos_sim = torch.matmul(row_dino, col_dino.T)           # (n, c)
    return torch.exp(-torch.clamp(1.0 - cos_sim, min=0.0) / (sigma ** 2))


def _d_sem(
    row_P: torch.Tensor,  # (n, L)
    col_P: torch.Tensor,  # (c, L)
) -> torch.Tensor:
    """
    Jensen-Shannon Divergence normalised to [0, 1] (log-2 base).
    JSD(p||q) = 0.5*KL(p||M) + 0.5*KL(q||M),  M = (p+q)/2.

    Computed in natural log then divided by ln(2) to get bits ∈ [0,1].
    Uses (n,1,L) and (1,c,L) broadcasting → (n,c,L) intermediate.
    Keep chunk sizes modest (≤1000 each) to stay within VRAM.
    """
    eps = 1e-10
    P_n = row_P.unsqueeze(1)   # (n, 1, L)
    P_c = col_P.unsqueeze(0)   # (1, c, L)
    M = (P_n + P_c) * 0.5      # (n, c, L)
    kl_n = (P_n * (torch.log(P_n + eps) - torch.log(M + eps))).sum(-1)  # (n, c)
    kl_c = (P_c * (torch.log(P_c + eps) - torch.log(M + eps))).sum(-1)  # (n, c)
    return torch.clamp(0.5 * (kl_n + kl_c) / _LN2, min=0.0, max=1.0)   # (n, c)


def _k_joint(
    row_dino: torch.Tensor, col_dino: torch.Tensor, sigma: float,
    row_P: torch.Tensor, col_P: torch.Tensor,
) -> torch.Tensor:
    """K_joint = K_struct * (1 - D_sem) ∈ [0,1]."""
    ks = _k_struct(row_dino, col_dino, sigma)
    ds = _d_sem(row_P, col_P)
    return ks * (1.0 - ds)


def _k_joint_col(
    dino: torch.Tensor,      # (N, D)
    P: torch.Tensor,         # (N, L)
    best_dino: torch.Tensor, # (1, D)
    best_P: torch.Tensor,    # (1, L)
    sigma: float,
    chunk_size: int,
) -> torch.Tensor:
    """
    Compute K_joint(all N, single candidate) in row-chunks.
    Returns (N,) tensor — used to update K_n after each selection.
    """
    N = dino.shape[0]
    col = torch.empty(N, device=dino.device, dtype=torch.float32)
    for ns in range(0, N, chunk_size):
        ne = min(ns + chunk_size, N)
        kj = _k_joint(dino[ns:ne], best_dino, sigma, P[ns:ne], best_P)  # (chunk,1)
        col[ns:ne] = kj.squeeze(1)
        del kj
    return col


# ---------------------------------------------------------------------------
# SCALPEL sampler
# ---------------------------------------------------------------------------

@register_sampler("scalpel")
def scalpel_sampling(**kwargs) -> List[int]:
    """
    SCALPEL greedy UW-Coverage sampler — SLICEABLE (run once at max_budget).

    Required kwargs:
      image_embeddings     : np.ndarray (N, 768) — DINOv2 structural features
      vlm_image_embeddings : np.ndarray (N, d)   — PLIP semantic features
      text_embeddings      : np.ndarray (L, d)   — PLIP text prototypes
      max_budget           : int
      device               : torch.device

    Optional kwargs (from config.yaml samplers.scalpel):
      tau        : float = 0.05   EDL temperature
      chunk_size : int   = 500    rows processed per kernel call (VRAM tuning)
      n_sigma    : int   = 2000   subsample size for sigma estimation
    """
    dino_np: np.ndarray      = kwargs["image_embeddings"]
    vlm_np: np.ndarray       = kwargs["vlm_image_embeddings"]
    text_np: np.ndarray      = kwargs["text_embeddings"]
    max_budget: int          = kwargs["max_budget"]
    device: torch.device     = kwargs["device"]
    tau: float               = kwargs.get("tau", 0.05)
    chunk_size: int          = kwargs.get("chunk_size", 500)
    n_sigma: int             = kwargs.get("n_sigma", 2000)

    N = dino_np.shape[0]
    L = text_np.shape[0]

    # ── Load features onto device ────────────────────────────────────────────
    dino = F.normalize(
        torch.tensor(dino_np, device=device, dtype=torch.float32), p=2, dim=1
    )
    vlm_img = F.normalize(
        torch.tensor(vlm_np, device=device, dtype=torch.float32), p=2, dim=1
    )
    text_emb = F.normalize(
        torch.tensor(text_np, device=device, dtype=torch.float32), p=2, dim=1
    )

    # ── Phase 1: EDL vacuity + Dirichlet means ────────────────────────────────
    # alpha_{i,c} = exp(cos(v_i^VLM, t_c) / tau) — temperature scaling avoids
    # the range compression of alpha∈[1,2] in the naive max(cos,0)+1 version.
    U, P = _compute_edl(vlm_img, text_emb, tau)      # U:(N,), P:(N,L)
    del vlm_img, text_emb
    clear_memory()

    # ── Phase 2: Adaptive sigma (median pairwise distance on DINOv2 subsample) ─
    sigma = _adaptive_sigma(dino, n_ref=n_sigma)

    # ── Phase 3: Greedy UW-Coverage ──────────────────────────────────────────
    # K_n[n] = max_{s in S} K_joint(x_n, x_s)  — current coverage of point n
    # Initialised to 0 (cold-start: S=∅ → ΔCov = Σ_n U_n * K_joint(x_n, x_i))
    K_n = torch.zeros(N, device=device, dtype=torch.float32)
    selected_indices: List[int] = []
    selected_set: set = set()

    for step in tqdm(range(max_budget), desc="SCALPEL Selection"):

        best_idx   = -1
        best_score = -float("inf")

        # --- Outer loop: candidate chunks --------------------------------
        for cs in range(0, N, chunk_size):
            ce = min(cs + chunk_size, N)
            cand_dino = dino[cs:ce]    # (c, D)
            cand_P    = P[cs:ce]       # (c, L)
            cand_U    = U[cs:ce]       # (c,)

            # delta_cov[j] = sum_n U_n * max(K_joint(x_n, x_j) - K_n[n], 0)
            delta_cov = torch.zeros(ce - cs, device=device, dtype=torch.float32)

            # --- Inner loop: row chunks ----------------------------------
            for ns in range(0, N, chunk_size):
                ne = min(ns + chunk_size, N)
                kj = _k_joint(
                    dino[ns:ne], cand_dino, sigma,
                    P[ns:ne], cand_P,
                )                                               # (n, c)
                gain = torch.clamp(
                    kj - K_n[ns:ne].unsqueeze(1), min=0.0
                )                                               # (n, c)
                delta_cov += (U[ns:ne].unsqueeze(1) * gain).sum(0)  # (c,)
                del kj, gain
            clear_memory()

            # Score = U_i * ΔCov(x_i | S)  — multiplicative, no hyperparameter
            scores = cand_U * delta_cov                          # (c,)

            # Mask already-selected candidates
            for si in selected_set:
                if cs <= si < ce:
                    scores[si - cs] = -float("inf")

            local_best = int(torch.argmax(scores).item())
            if scores[local_best].item() > best_score:
                best_score = scores[local_best].item()
                best_idx   = cs + local_best

            del cand_dino, cand_P, cand_U, delta_cov, scores
            clear_memory()

        if best_idx < 0 or best_idx in selected_set:
            print(f"SCALPEL: no valid candidate at step {step}, stopping early.")
            break

        selected_indices.append(best_idx)
        selected_set.add(best_idx)

        # Update K_n: K_n[n] ← max(K_n[n], K_joint(x_n, x_best))
        best_k_col = _k_joint_col(
            dino, P,
            dino[best_idx].unsqueeze(0),
            P[best_idx].unsqueeze(0),
            sigma, chunk_size,
        )                                                        # (N,)
        K_n = torch.maximum(K_n, best_k_col)
        del best_k_col
        clear_memory()

    del dino, P, U, K_n
    clear_memory()
    return selected_indices
