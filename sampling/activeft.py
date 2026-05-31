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

        sim_matrix = torch.matmul(features, theta_norm.t()) / tau
        max_sim, _ = torch.max(sim_matrix, dim=1)
        loss_dist = -torch.mean(max_sim)

        theta_sim = torch.matmul(theta_norm, theta_norm.t()) / tau
        mask = ~torch.eye(max_budget, device=device, dtype=torch.bool)
        theta_sim_filtered = theta_sim[mask].view(max_budget, max_budget - 1)
        loss_reg = torch.mean(torch.log(torch.sum(torch.exp(theta_sim_filtered), dim=1)))

        loss = loss_dist + lambda_reg * loss_reg
        loss.backward()
        optimizer.step()

    selected_indices = set()
    with torch.no_grad():
        theta_final = F.normalize(theta, p=2, dim=1)
        dist_to_real = torch.matmul(theta_final, features.t())
        _, ids_sort = torch.sort(dist_to_real, dim=1, descending=True)
        ids_sort = ids_sort.cpu().numpy()

        for i in tqdm(range(max_budget), desc="ActiveFT Selection"):
            for j in range(num_samples):
                candidate_idx = ids_sort[i, j]
                if candidate_idx not in selected_indices:
                    selected_indices.add(candidate_idx)
                    break

    del features, theta, sim_matrix, theta_sim, mask, dist_to_real, ids_sort
    clear_memory()

    return list(selected_indices)
