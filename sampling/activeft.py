from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from . import register_sampler


@register_sampler("activeft")
def activeft_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    max_budget = kwargs["max_budget"]
    device = kwargs["device"]

    num_samples = image_embeddings.shape[0]

    lr = kwargs.get("lr", 0.01)
    tau = kwargs.get("temperature", 0.07)
    iterations = kwargs.get("iterations", 100)
    lambda_reg = kwargs.get("balance", 1.0)

    features = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features = F.normalize(features, p=2, dim=1)

    init_indices = np.random.choice(num_samples, max_budget, replace=False)
    theta = features[init_indices].detach().clone()
    theta.requires_grad_(True)

    optimizer = torch.optim.Adam([theta], lr=lr)

    for _ in tqdm(range(iterations), desc="ActiveFT Optimization"):
        optimizer.zero_grad()

        theta_norm = F.normalize(theta, p=2, dim=1)

        # FIX (major): rewrite loss to match official ActiveFT log-ratio formulation.
        # Previous code used two SEPARATE additive terms (loss_dist + lambda_reg * loss_reg),
        # which is a non-equivalent approximation of the official InfoNCE-style loss.
        #
        # Official loss (get_loss in the ActiveFT repo):
        #   J = -mean( log(max_exp_sim_i) - log(max_exp_sim_i + balance * cent_self_sim[argmax_i]) )
        # where:
        #   sim_matrix[i, k] = features[i] · theta_norm[k] / tau   (N x K)
        #   max_exp_sim_i    = exp(sim_matrix[i, argmax_k])
        #   cent_self_sim[k] = sum_{k'≠k} exp(theta_norm[k] · theta_norm[k'] / tau)
        #                      (self-similarity sum of the MATCHED centroid for sample i)
        #
        # The log-ratio naturally couples distribution-matching and diversity in one term.

        # Step 1: feature-centroid similarities.
        sim_matrix = torch.matmul(features, theta_norm.t()) / tau   # [N, K]
        max_sim_vals, argmax_cols = torch.max(sim_matrix, dim=1)     # [N], [N]
        max_exp_sim = torch.exp(max_sim_vals)                        # [N]

        # Step 2: centroid self-similarity sums (exclude self-pair for each centroid).
        theta_sim = torch.matmul(theta_norm, theta_norm.t()) / tau   # [K, K]
        mask_self = torch.eye(max_budget, device=device, dtype=torch.bool)
        theta_sim_no_self = theta_sim.masked_fill(mask_self, float("-inf"))
        # sum exp over non-self pairs, per centroid row → shape [K]
        cent_exp_sum = torch.sum(torch.exp(theta_sim_no_self), dim=1)

        # Step 3: for each feature i, look up cent_exp_sum of its matched centroid.
        cent_term = cent_exp_sum[argmax_cols]                        # [N]

        # Step 4: log-ratio loss (InfoNCE-style, couples both objectives in one term).
        loss = -torch.mean(
            torch.log(max_exp_sim + 1e-9)
            - torch.log(max_exp_sim + lambda_reg * cent_term + 1e-9)
        )
        loss.backward()
        optimizer.step()

    # FIX (minor): use an ordered list + set to guarantee deterministic insertion order.
    # Python sets are hash-ordered; list(set(...)) produces non-deterministic ordering
    # across runs even with the same seed, which breaks reproducibility.
    selected_indices = []   # preserves insertion order
    selected_set = set()    # O(1) membership check

    with torch.no_grad():
        theta_final = F.normalize(theta, p=2, dim=1)
        dist_to_real = torch.matmul(theta_final, features.t())
        _, ids_sort = torch.sort(dist_to_real, dim=1, descending=True)
        ids_sort = ids_sort.cpu().numpy()

        for i in tqdm(range(max_budget), desc="ActiveFT Selection"):
            for j in range(num_samples):
                candidate_idx = int(ids_sort[i, j])
                if candidate_idx not in selected_set:
                    selected_indices.append(candidate_idx)
                    selected_set.add(candidate_idx)
                    break

    del features, theta, sim_matrix, theta_sim, mask_self, dist_to_real, ids_sort
    clear_memory()

    return selected_indices
