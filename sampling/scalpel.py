import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

from set_up import clear_memory
from . import register_sampler


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
# PLIP zero-shot class prior (semantic partition + evidential vacuity)
# ---------------------------------------------------------------------------
#
# v3 — the VLM is used ONLY as a label-free CLASS PRIOR, not for coverage.
#   * pred_class : argmax_c cos(v_i^PLIP, t_c)  → semantic partition (round-robin)
#   * U (vacuity): EDL uncertainty of that zero-shot prediction → phase-2 weight
# Coverage itself runs in DINOv2 space (the backbone the linear probe uses),
# so the (1-1/e) submodular guarantee lives in the SAME space as evaluation.

def _adaptive_tau(sim_cal: torch.Tensor, L: int, target_u_conf: float = 0.2) -> float:
    s90 = torch.quantile(sim_cal.max(dim=1).values, 0.90).item()
    if s90 <= 0:
        return 0.1
    denom = math.log(max(L / target_u_conf, 1.001))
    return max(s90 / denom, 1e-4)


def _plip_class_prior(
    vlm_img: torch.Tensor,   # (N, 512) L2-normalised PLIP image features
    text_emb: torch.Tensor,  # (L, 512) L2-normalised PLIP text prototypes
    tau: float,              # 0.0 → auto-calibrate
):
    sim = torch.matmul(vlm_img, text_emb.T)            # (N, L)
    sim_cal = sim - sim.mean(dim=0, keepdim=True)      # class-mean calibration
    if tau <= 0.0:
        tau = _adaptive_tau(sim_cal, text_emb.shape[0])
    alpha = torch.exp(sim_cal / tau)                   # Dirichlet evidence
    S = alpha.sum(dim=1)                               # Dirichlet strength
    U = text_emb.shape[0] / S                          # vacuity uncertainty (N,)
    pred = sim_cal.argmax(dim=1)                       # zero-shot predicted class (N,)
    return pred, U


def _rank_normalize(x: torch.Tensor) -> torch.Tensor:
    N = x.shape[0]
    ranks = torch.argsort(torch.argsort(x)).float() + 1.0
    return ranks / N


# ---------------------------------------------------------------------------
# Coverage kernel — Gaussian RBF on the DINOv2 space (aligned with the probe)
# ---------------------------------------------------------------------------

def _adaptive_sigma(features: torch.Tensor, n_ref: int = 2000) -> float:
    N = features.shape[0]
    n_ref = min(n_ref, N)
    idx = np.random.choice(N, n_ref, replace=False)
    ref = features[idx]
    sim = torch.matmul(ref, ref.T)
    dist_sq = torch.clamp(2.0 - 2.0 * sim, min=0.0)
    dist = torch.sqrt(dist_sq)
    dist.fill_diagonal_(float("inf"))
    tril_idx = torch.tril_indices(n_ref, n_ref, offset=-1)
    lower = dist[tril_idx[0], tril_idx[1]]
    sigma = lower.median().item()
    return max(sigma, 1e-4)


def _k_gaussian(row: torch.Tensor, col: torch.Tensor, sigma: float) -> torch.Tensor:
    cos_sim = torch.matmul(row, col.T)
    return torch.exp(-torch.clamp(1.0 - cos_sim, min=0.0) / (sigma ** 2))


def _k_col(
    features: torch.Tensor,
    best_feat: torch.Tensor,
    sigma: float,
    chunk_size: int,
) -> torch.Tensor:
    N = features.shape[0]
    col = torch.empty(N, device=features.device, dtype=torch.float32)
    for ns in range(0, N, chunk_size):
        ne = min(ns + chunk_size, N)
        col[ns:ne] = _k_gaussian(features[ns:ne], best_feat, sigma).squeeze(1)
    return col


# ---------------------------------------------------------------------------
# Main sampling function — Class-Balanced Evidential Coverage (v3)
# ---------------------------------------------------------------------------

@register_sampler("scalpel")
def scalpel_sampling(**kwargs) -> List[int]:
    # Coverage runs in DINOv2 space (image_embeddings); PLIP only supplies the
    # label-free class prior (semantic partition + evidential vacuity).
    dino_np:  np.ndarray = kwargs["image_embeddings"]      # (N, 768) DINOv2 — coverage space
    vlm_np:   np.ndarray = kwargs["vlm_image_embeddings"]  # (N, 512) PLIP image — class prior
    text_np:  np.ndarray = kwargs["text_embeddings"]       # (L, 512) PLIP text prototypes
    max_budget: int      = kwargs["max_budget"]
    device: torch.device = kwargs["device"]
    tau: float           = kwargs.get("tau", 0.0)
    chunk_size: int      = kwargs.get("chunk_size", 500)
    n_sigma: int         = kwargs.get("n_sigma", 2000)

    N = dino_np.shape[0]
    L = text_np.shape[0]
    # Phase 1: pure submodular coverage (MaxHerding) per class — clean class reps.
    # Phase 2: uncertainty-weighted coverage (UHerding) — refine boundary regions.
    step_budget = max(L, int(0.2 * max_budget))

    dino = F.normalize(
        torch.tensor(dino_np, device=device, dtype=torch.float32), p=2, dim=1
    )
    vlm_img = F.normalize(
        torch.tensor(vlm_np, device=device, dtype=torch.float32), p=2, dim=1
    )
    text_emb = F.normalize(
        torch.tensor(text_np, device=device, dtype=torch.float32), p=2, dim=1
    )

    # ---- PLIP class prior (label-free) ------------------------------------
    pred, U = _plip_class_prior(vlm_img, text_emb, tau)
    del vlm_img, text_emb
    clear_memory()

    U_norm   = _rank_normalize(U)                                  # high = uncertain
    W_phase1 = torch.ones(N, device=device, dtype=torch.float32)   # pure coverage
    del U

    pred_np   = pred.cpu().numpy()
    pool_size = np.bincount(pred_np, minlength=L)
    counts    = np.zeros(L, dtype=np.int64)                        # selected per class

    # ---- DINOv2 coverage state --------------------------------------------
    sigma = _adaptive_sigma(dino, n_ref=n_sigma)
    K_n = torch.zeros(N, device=device, dtype=torch.float32)       # current coverage of each point

    selected_indices: List[int] = []
    selected_set: set = set()

    for step in tqdm(range(max_budget), desc="SCALPEL Selection"):

        # Round-robin: target the least-represented predicted class that still
        # has unselected members (ties → larger class pool, like TypiClust).
        avail = [c for c in range(L) if pool_size[c] - counts[c] > 0]
        if not avail:
            print(f"SCALPEL: pool exhausted at step {step}, stopping early.")
            break
        c_star = min(avail, key=lambda c: (counts[c], -pool_size[c]))

        # Per-point coverage weight (importance of covering point n).
        W = W_phase1 if step < step_budget else U_norm

        best_idx   = -1
        best_score = -float("inf")

        for cs in range(0, N, chunk_size):
            ce = min(cs + chunk_size, N)
            cls_chunk = pred[cs:ce]
            in_class = (cls_chunk == c_star)
            if not bool(in_class.any()):
                continue
            cand_dino = dino[cs:ce]

            # Marginal coverage gain ΔCov(i|S) = Σ_n W_n · max(K(n,i) - K_n[n], 0)
            delta_cov = torch.zeros(ce - cs, device=device, dtype=torch.float32)
            for ns in range(0, N, chunk_size):
                ne = min(ns + chunk_size, N)
                k = _k_gaussian(dino[ns:ne], cand_dino, sigma)
                gain = torch.clamp(k - K_n[ns:ne].unsqueeze(1), min=0.0)
                delta_cov += (W[ns:ne].unsqueeze(1) * gain).sum(0)
                del k, gain

            scores = delta_cov
            # Restrict selection to the target class; exclude already-selected.
            scores = torch.where(in_class, scores, torch.full_like(scores, -float("inf")))
            for si in selected_set:
                if cs <= si < ce:
                    scores[si - cs] = -float("inf")

            local_best = int(torch.argmax(scores).item())
            if scores[local_best].item() > best_score:
                best_score = scores[local_best].item()
                best_idx   = cs + local_best

            del cand_dino, delta_cov, scores
            clear_memory()

        if best_idx < 0 or best_idx in selected_set:
            print(f"SCALPEL: no valid candidate at step {step}, stopping early.")
            break

        selected_indices.append(best_idx)
        selected_set.add(best_idx)
        counts[pred_np[best_idx]] += 1

        best_k_col = _k_col(dino, dino[best_idx].unsqueeze(0), sigma, chunk_size)
        K_n = torch.maximum(K_n, best_k_col)
        del best_k_col
        clear_memory()

    del dino, U_norm, W_phase1, K_n
    clear_memory()
    return selected_indices
