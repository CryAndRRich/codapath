from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.runtime import clear_memory
from .. import register_sampler


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

        sim_matrix = torch.matmul(features, theta_norm.t()) / tau   
        max_sim_vals, argmax_cols = torch.max(sim_matrix, dim=1)     
        max_exp_sim = torch.exp(max_sim_vals)                        

        theta_sim = torch.matmul(theta_norm, theta_norm.t()) / tau
        # upstream ActiveFT keeps the self-similarity term (diagonal) in this sum —
        # it acts as an anti-collapse anchor in the partition function, not noise
        # to be masked out; removing it changes the strength of the diversity term.
        cent_exp_sum = torch.sum(torch.exp(theta_sim), dim=1)

        cent_term = cent_exp_sum[argmax_cols]                        

        loss = -torch.mean(
            torch.log(max_exp_sim + 1e-9)
            - torch.log(max_exp_sim + lambda_reg * cent_term + 1e-9)
        )
        loss.backward()
        optimizer.step()

    selected_indices = []   
    selected_set = set()    

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

    del features, theta, sim_matrix, theta_sim, dist_to_real, ids_sort
    clear_memory()

    return selected_indices