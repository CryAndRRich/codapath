"""ActiveFT (Xie et al., CVPR 2023).

Verified against `repos/activeft/data_selection/sample_tools/ActiveFT_CIFAR.py`.
Optimise `k` continuous centroids on the unit sphere against

    J = -mean[ log s_pos - log(s_pos + balance * sum_j exp(theta_i . theta_j / tau)) ]

then snap each centroid to its nearest not-yet-taken real sample.

Official defaults: `temperature=0.07`, `lr=0.001`, `balance=1.0`, random init,
`max_iter=300` (CIFAR variant; the ImageNet variant uses 100).

Two subtleties in the diversity term, both reproduced below: its left factor is
detached, and the self-similarity diagonal stays in the sum.
"""

import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.runtime import clear_memory
from ..registry import register_sampler


@register_sampler("activeft")
def activeft_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    max_budget = kwargs["max_budget"]
    device = kwargs["device"]

    num_samples = image_embeddings.shape[0]

    lr = kwargs.get("lr", 0.001)
    tau = kwargs.get("temperature", 0.07)
    iterations = kwargs.get("iterations", 300)
    lambda_reg = kwargs.get("balance", 1.0)
    trace = kwargs.get("trace")

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

        # Two details of the official diversity term, both easy to get wrong:
        # 1. the LEFT factor is detached, so a gradient reaches each centroid
        #    through one factor of this product rather than two. Without it the
        #    diversity gradient is roughly doubled and the optimisation drifts
        #    away from the published trajectory.
        # 2. the self-similarity diagonal STAYS in the sum. It anchors the
        #    partition function against collapse; masking it out silently
        #    strengthens the diversity term.
        theta_sim = torch.matmul(theta_norm.detach(), theta_norm.t()) / tau
        cent_exp_sum = torch.sum(torch.exp(theta_sim), dim=0)

        cent_term = cent_exp_sum[argmax_cols]

        loss = -torch.mean(
            torch.log(max_exp_sim + 1e-9)
            - torch.log(max_exp_sim + lambda_reg * cent_term + 1e-9)
        )
        loss.backward()
        optimizer.step()

    selected_indices = []
    selected_set = set()

    started = time.time()
    if trace is not None:
        trace.start_round(0)

    with torch.no_grad():
        theta_final = F.normalize(theta, p=2, dim=1)
        dist_to_real = torch.matmul(theta_final, features.t())
        sorted_sims, ids_sort = torch.sort(dist_to_real, dim=1, descending=True)
        sorted_sims = sorted_sims.cpu().numpy()
        ids_sort = ids_sort.cpu().numpy()

        for i in tqdm(range(max_budget), desc="ActiveFT Selection"):
            for j in range(num_samples):
                candidate_idx = int(ids_sort[i, j])
                if candidate_idx not in selected_set:
                    if trace is not None:
                        # ActiveFT optimises `max_budget` centroids and then
                        # takes the real point nearest each one. The score is
                        # that cosine similarity to its own centroid -- a pure
                        # coverage quantity; this method never fits a
                        # classifier, so it has no uncertainty term at all.
                        trace.add_step(
                            candidate_idx,
                            score=float(sorted_sims[i, j]),
                            coverage=float(sorted_sims[i, j]),
                            centroid=float(i),
                        )
                    selected_indices.append(candidate_idx)
                    selected_set.add(candidate_idx)
                    break

    if trace is not None:
        trace.add_round(
            num_selected=len(selected_indices),
            seconds=time.time() - started,
            final_loss=float(loss.item()),
            iterations=float(iterations),
        )

    del features, theta, sim_matrix, theta_sim, dist_to_real, ids_sort
    clear_memory()

    return selected_indices
