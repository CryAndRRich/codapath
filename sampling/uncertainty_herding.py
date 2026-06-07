from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from . import register_sampler


@register_sampler("uncertainty_herding")
def uncertainty_herding_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 50)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]
    chunk_size = kwargs.get("chunk_size", 2000)

    from trainer import train_linear

    num_samples = image_embeddings.shape[0]
    step_budget = max(num_classes, int(0.2 * max_budget))

    features = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features = F.normalize(features, p=2, dim=1)

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

    for step in tqdm(range(max_budget), desc="UHerding Selection"):

        if step == step_budget and len(selected_indices) >= 2:
            sel_feats = features[selected_indices]     
            sel_sim = torch.matmul(sel_feats, sel_feats.T)
            sel_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sel_sim, min=0.0))
            sel_dist.fill_diagonal_(float("inf"))
            new_sigma = sel_dist.min().item()
            sigma = max(new_sigma, 1e-6)
            del sel_feats, sel_sim, sel_dist
            clear_memory()

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

            probe = train_linear(
                image_embeddings[selected_indices],
                oracle_labels[selected_indices],
                num_classes, probe_epochs, probe_lr, device,
            )
            probs = probe.predict_proba(image_embeddings, device)  
            s_probs = np.sort(probs, axis=1)
            margin = s_probs[:, -1] - s_probs[:, -2]
            U = torch.tensor(1.0 - margin, device=device, dtype=torch.float32)
            del probe
            clear_memory()

        best_idx = -1
        best_score = -float("inf")
        best_k_col = None

        for cs in range(0, num_samples, chunk_size):
            ce = min(cs + chunk_size, num_samples)
            cand = features[cs:ce]                     

            sim = torch.matmul(features, cand.T)      
            dist_sq = torch.clamp(2.0 - 2.0 * sim, min=0.0)
            k_vals = torch.exp(-dist_sq / (sigma ** 2)) 
            gain = torch.clamp(k_vals - k_running.unsqueeze(1), min=0.0) 
            scores = (U.unsqueeze(1) * gain).sum(dim=0) 

            for si in selected_set:
                if cs <= si < ce:
                    scores[si - cs] = -float("inf")

            local_best = torch.argmax(scores).item()
            if scores[local_best].item() > best_score:
                best_score = scores[local_best].item()
                best_idx = cs + local_best
                best_k_col = k_vals[:, local_best].clone()

            del cand, sim, dist_sq, k_vals, gain, scores
            clear_memory()

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