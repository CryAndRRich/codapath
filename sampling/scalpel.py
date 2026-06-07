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
        emb = clip_model.text_projection(text_out.pooler_output).cpu().numpy()  # (N*T, 512)
    del clip_model, tokens
    clear_memory()

    text_embeddings = emb.reshape(num_classes, num_templates, -1).mean(axis=1).astype(np.float32)
    norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    text_embeddings /= np.maximum(norms, 1e-8)
    return text_embeddings


# ---------------------------------------------------------------------------
# EDL branch — returns only U (vacuity); P no longer needed after K_joint change
# ---------------------------------------------------------------------------

def _adaptive_tau(sim_cal: torch.Tensor, L: int, target_u_conf: float = 0.2) -> float:
    """
    Pick τ so the 90th-percentile-confident sample has U ≈ target_u_conf.
    After class-mean subtraction, s_cal_max ~ 0 for a uniform-uncertainty sample
    and > 0 for a confident one.  τ = s90 / log(L / target_u_conf).
    """
    s90 = torch.quantile(sim_cal.max(dim=1).values, 0.90).item()
    if s90 <= 0:
        return 0.1
    denom = math.log(max(L / target_u_conf, 1.001))
    return max(s90 / denom, 1e-4)


def _compute_edl(
    vlm_img: torch.Tensor,   # (N, d_vlm) L2-normalised PLIP image features
    text_emb: torch.Tensor,  # (L, d_vlm) L2-normalised PLIP text prototypes
    tau: float,              # 0.0 → auto-calibrate from pool statistics
) -> torch.Tensor:           # returns U (N,)
    sim = torch.matmul(vlm_img, text_emb.T)           # (N, L) cosine similarities
    sim_cal = sim - sim.mean(dim=0, keepdim=True)      # Option B: per-class mean subtraction
    if tau <= 0.0:
        tau = _adaptive_tau(sim_cal, text_emb.shape[0])
    alpha = torch.exp(sim_cal / tau)
    S = alpha.sum(dim=1, keepdim=True)
    return text_emb.shape[0] / S.squeeze(1)            # U = L / S


def _rank_normalize(x: torch.Tensor) -> torch.Tensor:
    """Map values to their fractional ranks in [1/N, 1]. Removes absolute scale sensitivity."""
    N = x.shape[0]
    ranks = torch.argsort(torch.argsort(x)).float() + 1.0
    return ranks / N


# ---------------------------------------------------------------------------
# Kernel functions — unified Gaussian kernel used for both DINOv2 and PLIP
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


def _k_gaussian(
    row: torch.Tensor,
    col: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    cos_sim = torch.matmul(row, col.T)
    return torch.exp(-torch.clamp(1.0 - cos_sim, min=0.0) / (sigma ** 2))


def _k_joint(
    row_dino: torch.Tensor, col_dino: torch.Tensor, sigma_dino: float,
    row_vlm:  torch.Tensor, col_vlm:  torch.Tensor, sigma_vlm:  float,
) -> torch.Tensor:
    """
    Product kernel: K_joint = K_dino × K_vlm.
    Two samples are redundant only when similar in BOTH structural (DINOv2)
    AND semantic (PLIP image) space.  Product of two PD kernels is PD,
    so submodular coverage guarantees carry through.
    """
    return _k_gaussian(row_dino, col_dino, sigma_dino) * _k_gaussian(row_vlm, col_vlm, sigma_vlm)


def _k_joint_col(
    dino:       torch.Tensor,
    vlm_img:    torch.Tensor,
    best_dino:  torch.Tensor,
    best_vlm:   torch.Tensor,
    sigma_dino: float,
    sigma_vlm:  float,
    chunk_size: int,
) -> torch.Tensor:
    N = dino.shape[0]
    col = torch.empty(N, device=dino.device, dtype=torch.float32)
    for ns in range(0, N, chunk_size):
        ne = min(ns + chunk_size, N)
        kj = _k_joint(
            dino[ns:ne], best_dino, sigma_dino,
            vlm_img[ns:ne], best_vlm, sigma_vlm,
        )
        col[ns:ne] = kj.squeeze(1)
        del kj
    return col


# ---------------------------------------------------------------------------
# Main sampling function
# ---------------------------------------------------------------------------

@register_sampler("scalpel")
def scalpel_sampling(**kwargs) -> List[int]:
    dino_np: np.ndarray      = kwargs["image_embeddings"]       # (N, 768) DINOv2
    vlm_np:  np.ndarray      = kwargs["vlm_image_embeddings"]   # (N, 512) PLIP image
    text_np: np.ndarray      = kwargs["text_embeddings"]        # (L, 512) PLIP text
    max_budget: int          = kwargs["max_budget"]
    device: torch.device     = kwargs["device"]
    tau: float               = kwargs.get("tau", 0.0)           # 0.0 = auto-adaptive
    chunk_size: int          = kwargs.get("chunk_size", 500)
    n_sigma: int             = kwargs.get("n_sigma", 2000)

    N = dino_np.shape[0]

    dino = F.normalize(
        torch.tensor(dino_np, device=device, dtype=torch.float32), p=2, dim=1
    )
    vlm_img = F.normalize(
        torch.tensor(vlm_np, device=device, dtype=torch.float32), p=2, dim=1
    )
    text_emb = F.normalize(
        torch.tensor(text_np, device=device, dtype=torch.float32), p=2, dim=1
    )

    # EDL: returns only U (P no longer needed after product-kernel change)
    U = _compute_edl(vlm_img, text_emb, tau)
    del text_emb
    clear_memory()

    # Rank-normalise U → removes outlier/domain-gap domination
    U_norm = _rank_normalize(U)
    del U

    # Kernel bandwidths — independent for each feature space
    sigma_dino = _adaptive_sigma(dino,    n_ref=n_sigma)
    sigma_vlm  = _adaptive_sigma(vlm_img, n_ref=n_sigma)

    K_n = torch.zeros(N, device=device, dtype=torch.float32)
    selected_indices: List[int] = []
    selected_set: set = set()

    for step in tqdm(range(max_budget), desc="SCALPEL Selection"):

        best_idx   = -1
        best_score = -float("inf")

        for cs in range(0, N, chunk_size):
            ce = min(cs + chunk_size, N)
            cand_dino   = dino[cs:ce]
            cand_vlm    = vlm_img[cs:ce]
            cand_U_norm = U_norm[cs:ce]

            delta_cov = torch.zeros(ce - cs, device=device, dtype=torch.float32)

            for ns in range(0, N, chunk_size):
                ne = min(ns + chunk_size, N)
                kj = _k_joint(
                    dino[ns:ne], cand_dino, sigma_dino,
                    vlm_img[ns:ne], cand_vlm, sigma_vlm,
                )
                gain = torch.clamp(kj - K_n[ns:ne].unsqueeze(1), min=0.0)
                delta_cov += (U_norm[ns:ne].unsqueeze(1) * gain).sum(0)
                del kj, gain

            scores = cand_U_norm * delta_cov

            for si in selected_set:
                if cs <= si < ce:
                    scores[si - cs] = -float("inf")

            local_best = int(torch.argmax(scores).item())
            if scores[local_best].item() > best_score:
                best_score = scores[local_best].item()
                best_idx   = cs + local_best

            del cand_dino, cand_vlm, cand_U_norm, delta_cov, scores
            clear_memory()

        if best_idx < 0 or best_idx in selected_set:
            print(f"SCALPEL: no valid candidate at step {step}, stopping early.")
            break

        selected_indices.append(best_idx)
        selected_set.add(best_idx)

        best_k_col = _k_joint_col(
            dino, vlm_img,
            dino[best_idx].unsqueeze(0),
            vlm_img[best_idx].unsqueeze(0),
            sigma_dino, sigma_vlm, chunk_size,
        )
        K_n = torch.maximum(K_n, best_k_col)
        del best_k_col
        clear_memory()

    del dino, vlm_img, U_norm, K_n
    clear_memory()
    return selected_indices
