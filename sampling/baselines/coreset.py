from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.runtime import clear_memory
from .. import register_sampler


@register_sampler("coreset")
def coreset_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    max_budget = kwargs["max_budget"]
    device = kwargs["device"]
    chunk_size = kwargs["chunk_size"]

    num_samples = image_embeddings.shape[0]

    unlabeled_tensor = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    unlabeled_tensor = F.normalize(unlabeled_tensor, p=2, dim=1)

    selected_indices = []

    first_idx = np.random.randint(0, num_samples)
    selected_indices.append(first_idx)

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
        selected_indices.append(furthest_idx)

        clear_memory()

    del unlabeled_tensor, min_distances
    clear_memory()

    return selected_indices