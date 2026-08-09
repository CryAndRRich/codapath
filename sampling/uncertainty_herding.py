from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from . import register_sampler


def _uherding_sampling_with_type(uncertainty_type: str, **kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 50)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]
    chunk_size = kwargs.get("chunk_size", 2000)
    
    # CEC specific
    cec_n = kwargs.get("cec_n", 30)
    cec_k = kwargs.get("cec_k", 20)
    cec_beta = kwargs.get("cec_beta", 1.0)

    from trainer import train_linear

    num_samples = image_embeddings.shape[0]
    
    # 5 rounds -> train at 20%, 40%, 60%, 80% of max_budget
    train_steps = [int((i + 1) * max_budget / 5) for i in range(4)]

    features = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features = F.normalize(features, p=2, dim=1)

    # Initial Sigma estimation
    n_ref = min(1000, num_samples)
    ref_idx = np.random.choice(num_samples, n_ref, replace=False)
    ref = features[ref_idx]                             
    sim_ref = torch.matmul(ref, features.T)           
    for i, gi in enumerate(ref_idx):
        sim_ref[i, gi] = -2.0
    nn_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sim_ref.max(dim=1).values, min=0.0))
    sigma = nn_dist.mean().item()
    sigma = max(sigma, 1e-3)
    del ref, sim_ref, nn_dist
    clear_memory()

    U = torch.ones(num_samples, device=device, dtype=torch.float32)
    k_running = torch.zeros(num_samples, device=device, dtype=torch.float32)

    selected_indices: List[int] = []
    selected_set: set = set()

    for step in tqdm(range(max_budget), desc=f"UHerding {uncertainty_type.upper()} Selection"):

        if step in train_steps and len(selected_indices) >= 2:
            # Update Sigma based on selected points
            sel_feats = features[selected_indices]     
            sel_sim = torch.matmul(sel_feats, sel_feats.T)
            sel_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sel_sim, min=0.0))
            sel_dist.fill_diagonal_(float("inf"))
            new_sigma = sel_dist.min().item()
            sigma = max(new_sigma, 1e-3)
            del sel_feats, sel_sim, sel_dist
            clear_memory()

            # Recalculate k_running with new sigma
            k_running.zero_()
            for si in selected_indices:
                si_feat = features[si].unsqueeze(0)     
                for cs in range(0, num_samples, chunk_size):
                    ce = min(cs + chunk_size, num_samples)
                    chunk = features[cs:ce]
                    sim_c = torch.matmul(chunk, si_feat.T).squeeze(1)
                    dist_sq_c = torch.clamp(2.0 - 2.0 * sim_c, min=0.0)
                    k_c = torch.exp(-dist_sq_c / (sigma ** 2))
                    k_running[cs:ce] = torch.maximum(k_running[cs:ce], k_c)
                    del chunk, sim_c, dist_sq_c, k_c
            clear_memory()

            # Train linear probe and update U
            norm_embeddings = features.cpu().numpy()
            probe = train_linear(
                norm_embeddings[selected_indices],
                oracle_labels[selected_indices],
                num_classes, probe_epochs, probe_lr, device,
            )
            probs = probe.predict_proba(norm_embeddings, device)
            
            if uncertainty_type == "margin":
                s_probs = np.sort(probs, axis=1)
                margin = s_probs[:, -1] - s_probs[:, -2]
                U = torch.tensor(1.0 - margin, device=device, dtype=torch.float32)
                
            elif uncertainty_type == "entropy":
                entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1) / np.log(num_classes)
                U = torch.tensor(entropy, device=device, dtype=torch.float32)
                
            elif uncertainty_type == "cec":
                probs_t = torch.tensor(probs, device=device, dtype=torch.float32)
                
                # Contextual Prior
                P_c = torch.zeros(num_classes, device=device)
                for c in range(num_classes):
                    c_probs = probs_t[:, c]
                    top_N = min(cec_n, num_samples)
                    top_val, _ = torch.topk(c_probs, top_N)
                    P_c[c] = top_val.mean()
                P_c = torch.clamp(P_c, min=1e-6)
                
                # Calibrated Entropy
                cal_probs = probs_t / P_c.unsqueeze(0)
                cal_probs = cal_probs / cal_probs.sum(dim=1, keepdim=True)
                H_cal = -torch.sum(cal_probs * torch.log(cal_probs + 1e-9), dim=1) / np.log(num_classes)
                
                # Neighbor uncertainty using cosine similarity
                U_neighbor = torch.zeros(num_samples, device=device)
                for cs in range(0, num_samples, chunk_size):
                    ce = min(cs + chunk_size, num_samples)
                    chunk = features[cs:ce]
                    sim = torch.matmul(chunk, features.T)
                    # Get top k+1 (including self), then exclude self
                    top_k_sim, top_k_idx = torch.topk(sim, cec_k + 1, dim=1)
                    for i in range(ce - cs):
                        neighbors_idx = top_k_idx[i, 1:]
                        weights = top_k_sim[i, 1:]
                        weights = torch.clamp(weights, min=0.0)
                        if weights.sum() > 0:
                            weights = weights / weights.sum()
                        else:
                            weights = torch.ones_like(weights) / cec_k
                        U_neighbor[cs + i] = torch.sum(H_cal[neighbors_idx] * weights)
                
                U = H_cal + cec_beta * U_neighbor
            
            else:
                raise ValueError(f"Unknown uncertainty_type: {uncertainty_type}")

            del probe
            clear_memory()

        # Score = U * Coverage (both normalized)
        coverage_gains = []
        for cs in range(0, num_samples, chunk_size):
            ce = min(cs + chunk_size, num_samples)
            cand = features[cs:ce]                     

            sim = torch.matmul(features, cand.T)      
            dist_sq = torch.clamp(2.0 - 2.0 * sim, min=0.0)
            k_vals = torch.exp(-dist_sq / (sigma ** 2)) 
            gain = torch.clamp(k_vals - k_running.unsqueeze(1), min=0.0) 
            coverage_gains.append(gain.sum(dim=0))
            
            del cand, sim, dist_sq, k_vals, gain
            clear_memory()
            
        C = torch.cat(coverage_gains) # shape: num_samples
        
        # Normalize U and C
        U_min, U_max = U.min(), U.max()
        U_norm = (U - U_min) / (U_max - U_min) if U_max > U_min else U - U_min
        
        C_min, C_max = C.min(), C.max()
        C_norm = (C - C_min) / (C_max - C_min) if C_max > C_min else C - C_min
        
        scores = U_norm * C_norm

        # Mask out selected
        for si in selected_set:
            scores[si] = -float("inf")

        best_idx = torch.argmax(scores).item()
        
        # Recalculate k_col for best_idx to update k_running
        cand = features[best_idx].unsqueeze(0)
        sim = torch.matmul(features, cand.T)
        dist_sq = torch.clamp(2.0 - 2.0 * sim, min=0.0)
        best_k_col = torch.exp(-dist_sq / (sigma ** 2)).squeeze(1)

        if best_idx >= 0 and best_idx not in selected_set:
            selected_indices.append(best_idx)
            selected_set.add(best_idx)
            k_running = torch.maximum(k_running, best_k_col)
            del best_k_col
            clear_memory()
        else:
            break

    del features, U, k_running
    clear_memory()
    return selected_indices


@register_sampler("uncertainty_herding")
def uncertainty_herding_sampling(**kwargs) -> List[int]:
    return _uherding_sampling_with_type("margin", **kwargs)

@register_sampler("uherding_margin")
def uherding_margin_sampling(**kwargs) -> List[int]:
    return _uherding_sampling_with_type("margin", **kwargs)

@register_sampler("uherding_entropy")
def uherding_entropy_sampling(**kwargs) -> List[int]:
    return _uherding_sampling_with_type("entropy", **kwargs)

@register_sampler("uherding_cec")
def uherding_cec_sampling(**kwargs) -> List[int]:
    return _uherding_sampling_with_type("cec", **kwargs)