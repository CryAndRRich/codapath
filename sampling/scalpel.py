from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPTokenizer, CLIPTextModel

from set_up import clear_memory
from . import register_sampler

_LN2 = float(np.log(2.0))


class PLIPExtractor(nn.Module):
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
        emb = text_model(**tokens).pooler_output.cpu().numpy() 
    del text_model, tokens
    clear_memory()

    text_embeddings = emb.reshape(num_classes, num_templates, -1).mean(axis=1).astype(np.float32)
    norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    text_embeddings /= np.maximum(norms, 1e-8)
    return text_embeddings


def _compute_edl(
    vlm_img: torch.Tensor,  
    text_emb: torch.Tensor, 
    tau: float,
) -> tuple:
    sim = torch.matmul(vlm_img, text_emb.T)      
    alpha = torch.exp(sim / tau)                   
    S = alpha.sum(dim=1, keepdim=True)             
    L = text_emb.shape[0]
    U = L / S.squeeze(1)                            
    P = alpha / S                              
    return U, P


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


def _k_struct(
    row_dino: torch.Tensor, 
    col_dino: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    cos_sim = torch.matmul(row_dino, col_dino.T)        
    return torch.exp(-torch.clamp(1.0 - cos_sim, min=0.0) / (sigma ** 2))


def _d_sem(
    row_P: torch.Tensor,  
    col_P: torch.Tensor,  
) -> torch.Tensor:
    eps = 1e-10
    P_n = row_P.unsqueeze(1)  
    P_c = col_P.unsqueeze(0)  
    M = (P_n + P_c) * 0.5     
    kl_n = (P_n * (torch.log(P_n + eps) - torch.log(M + eps))).sum(-1) 
    kl_c = (P_c * (torch.log(P_c + eps) - torch.log(M + eps))).sum(-1) 
    return torch.clamp(0.5 * (kl_n + kl_c) / _LN2, min=0.0, max=1.0)  


def _k_joint(
    row_dino: torch.Tensor, col_dino: torch.Tensor, sigma: float,
    row_P: torch.Tensor, col_P: torch.Tensor,
) -> torch.Tensor:
    ks = _k_struct(row_dino, col_dino, sigma)
    ds = _d_sem(row_P, col_P)
    return ks * (1.0 - ds)


def _k_joint_col(
    dino: torch.Tensor,     
    P: torch.Tensor,         
    best_dino: torch.Tensor, 
    best_P: torch.Tensor,    
    sigma: float,
    chunk_size: int,
) -> torch.Tensor:
    N = dino.shape[0]
    col = torch.empty(N, device=dino.device, dtype=torch.float32)
    for ns in range(0, N, chunk_size):
        ne = min(ns + chunk_size, N)
        kj = _k_joint(dino[ns:ne], best_dino, sigma, P[ns:ne], best_P)  # (chunk,1)
        col[ns:ne] = kj.squeeze(1)
        del kj
    return col


@register_sampler("scalpel")
def scalpel_sampling(**kwargs) -> List[int]:
    dino_np: np.ndarray      = kwargs["image_embeddings"]
    vlm_np: np.ndarray       = kwargs["vlm_image_embeddings"]
    text_np: np.ndarray      = kwargs["text_embeddings"]
    max_budget: int          = kwargs["max_budget"]
    device: torch.device     = kwargs["device"]
    tau: float               = kwargs.get("tau", 0.05)
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

    U, P = _compute_edl(vlm_img, text_emb, tau)      
    del vlm_img, text_emb
    clear_memory()

    sigma = _adaptive_sigma(dino, n_ref=n_sigma)

    K_n = torch.zeros(N, device=device, dtype=torch.float32)
    selected_indices: List[int] = []
    selected_set: set = set()

    for step in tqdm(range(max_budget), desc="SCALPEL Selection"):

        best_idx   = -1
        best_score = -float("inf")
        for cs in range(0, N, chunk_size):
            ce = min(cs + chunk_size, N)
            cand_dino = dino[cs:ce]   
            cand_P    = P[cs:ce]     
            cand_U    = U[cs:ce]     

            delta_cov = torch.zeros(ce - cs, device=device, dtype=torch.float32)

            for ns in range(0, N, chunk_size):
                ne = min(ns + chunk_size, N)
                kj = _k_joint(
                    dino[ns:ne], cand_dino, sigma,
                    P[ns:ne], cand_P,
                )                                           
                gain = torch.clamp(
                    kj - K_n[ns:ne].unsqueeze(1), min=0.0
                )                                             
                delta_cov += (U[ns:ne].unsqueeze(1) * gain).sum(0)  
                del kj, gain
            clear_memory()

            scores = cand_U * delta_cov                     

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

        best_k_col = _k_joint_col(
            dino, P,
            dino[best_idx].unsqueeze(0),
            P[best_idx].unsqueeze(0),
            sigma, chunk_size,
        )                                                       
        K_n = torch.maximum(K_n, best_k_col)
        del best_k_col
        clear_memory()

    del dino, P, U, K_n
    clear_memory()
    return selected_indices