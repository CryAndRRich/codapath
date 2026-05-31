from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans

from set_up import clear_memory
from . import register_sampler


@register_sampler("typiclust")
def typiclust_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    max_budget = kwargs["max_budget"]
    device = kwargs["device"]
    chunk_size = kwargs["chunk_size"]

    num_samples = image_embeddings.shape[0]
    K_NN = kwargs.get("k_nn", 20)

    features_tensor = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features_tensor = F.normalize(features_tensor, p=2, dim=1)

    typicality = torch.zeros(num_samples, device=device)

    for chunk_start in range(0, num_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk = features_tensor[chunk_start:chunk_end]

        sim_matrix = torch.matmul(chunk, features_tensor.T)
        dist_matrix = 1.0 - sim_matrix

        k_val = min(K_NN + 1, num_samples)
        topk_dist, _ = torch.topk(dist_matrix, k=k_val, dim=1, largest=False)

        mean_dist = topk_dist[:, 1:].mean(dim=1)
        typicality[chunk_start:chunk_end] = 1.0 / (mean_dist + 1e-5)

        del sim_matrix, dist_matrix, topk_dist
        clear_memory()

    typicality = typicality.cpu().numpy()
    features_np = features_tensor.cpu().numpy()

    num_clusters = max_budget

    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters, n_init="auto", random_state=42)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000, n_init="auto", random_state=42)

    cluster_ids = km.fit_predict(features_np)
    cluster_sizes = np.bincount(cluster_ids, minlength=num_clusters)

    sorted_clusters = np.argsort(cluster_sizes)[::-1]
    valid_clusters = [c for c in sorted_clusters if cluster_sizes[c] > 0]

    selected_indices = []
    selected_set = set()

    i = 0
    with tqdm(total=max_budget, desc="TypiClust Selection") as pbar:
        while len(selected_indices) < max_budget:
            cluster = valid_clusters[i % len(valid_clusters)]

            in_cluster_indices = np.where(cluster_ids == cluster)[0]
            available_indices = [idx for idx in in_cluster_indices if idx not in selected_set]

            if len(available_indices) > 0:
                cluster_typ = typicality[available_indices]
                best_local_idx = np.argmax(cluster_typ)
                best_global_idx = available_indices[best_local_idx]

                selected_indices.append(best_global_idx)
                selected_set.add(best_global_idx)
                pbar.update(1)

            i += 1
            if i > len(valid_clusters) * max_budget:
                break

    if len(selected_indices) < max_budget:
        remaining = list(set(range(num_samples)) - selected_set)
        missing = max_budget - len(selected_indices)
        fallback = np.random.choice(remaining, missing, replace=False)
        selected_indices.extend(fallback.tolist())

    del features_tensor, typicality
    clear_memory()

    return selected_indices
