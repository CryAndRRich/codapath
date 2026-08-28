"""Core-Set / k-center greedy (Sener & Savarese, ICLR 2018).

Verified against `repos/badge/query_strategies/kcenter_greedy.py`: seed a
center, then repeatedly take the point whose distance to the nearest chosen
center is largest, updating that nearest-center distance incrementally.

Distance here is `1 - cosine` on L2-normalized rows, which equals half the
squared Euclidean distance. k-center greedy only ever compares distances, and
a monotone transform leaves every argmax unchanged, so the selection is
identical to the reference while matching this project's kernel convention.
"""

import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.runtime import clear_memory
from ..registry import register_sampler


@register_sampler("coreset")
def coreset_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    max_budget = kwargs["max_budget"]
    device = kwargs["device"]
    chunk_size = kwargs["chunk_size"]
    trace = kwargs.get("trace")

    num_samples = image_embeddings.shape[0]

    unlabeled_tensor = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    unlabeled_tensor = F.normalize(unlabeled_tensor, p=2, dim=1)

    selected_indices = []

    first_idx = np.random.randint(0, num_samples)
    selected_indices.append(first_idx)

    started = time.time()
    if trace is not None:
        trace.start_round(0)
        # The seed centre is drawn at random, so it has no k-center distance of
        # its own -- recorded with no score, like every other random pick here.
        trace.add_step(int(first_idx))

    min_distances = torch.full((num_samples,), float("inf"), device=device)

    for _ in tqdm(range(max_budget - 1), desc="CoreSet Selection"):
        latest_idx = selected_indices[-1]
        latest_feature = unlabeled_tensor[latest_idx].unsqueeze(0)

        for chunk_start in range(0, num_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_samples)
            chunk_tensor = unlabeled_tensor[chunk_start:chunk_end]

            sim_chunk = torch.matmul(chunk_tensor, latest_feature.T).squeeze(1)
            dist_chunk = 1.0 - sim_chunk

            min_distances[chunk_start:chunk_end] = torch.minimum(
                min_distances[chunk_start:chunk_end],
                dist_chunk
            )

        min_distances[selected_indices] = -1.0

        furthest_idx = torch.argmax(min_distances).item()
        if trace is not None:
            # k-center greedy's acquisition value IS this distance to the
            # nearest already-chosen centre, which is a pure coverage quantity:
            # large means "far from everything selected so far". There is no
            # uncertainty term in this method at all, so none is recorded.
            best_distance = min_distances[furthest_idx].item()
            runner_up = torch.topk(min_distances, 2).values[1].item() \
                if min_distances.numel() > 1 else None
            trace.add_step(
                int(furthest_idx),
                score=best_distance,
                margin_to_runner_up=(
                    None if runner_up is None else best_distance - runner_up
                ),
                coverage=best_distance,
            )
        selected_indices.append(furthest_idx)

        clear_memory()

    if trace is not None:
        trace.add_round(
            num_selected=len(selected_indices),
            seconds=time.time() - started,
            scores=min_distances.detach().cpu().numpy(),
        )

    del unlabeled_tensor, min_distances
    clear_memory()

    return selected_indices
