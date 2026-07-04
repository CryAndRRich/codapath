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
    existing_labeled_indices = kwargs.get("existing_labeled_indices", [])

    num_samples = image_embeddings.shape[0]
    K_NN = kwargs.get("k_nn", 20)

    features_tensor = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features_tensor = F.normalize(features_tensor, p=2, dim=1)
    features_np = features_tensor.cpu().numpy()

    num_clusters = min(len(existing_labeled_indices) + max_budget, 500)

    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters, n_init="auto", random_state=42)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000, n_init="auto", random_state=42)

    cluster_ids = km.fit_predict(features_np)
    cluster_sizes = np.bincount(cluster_ids, minlength=num_clusters)

    typicality = np.zeros(num_samples, dtype=np.float32)

    for c in range(num_clusters):
        in_cluster = np.where(cluster_ids == c)[0]
        if len(in_cluster) == 0:
            continue

        cluster_feats = features_tensor[in_cluster] 
        
        n_c = len(in_cluster)
        if n_c < 2:
            typicality[in_cluster] = 0.0
            continue
        mean_nn_dist = torch.zeros(n_c, device=device)

        for cs in range(0, n_c, chunk_size):
            ce = min(cs + chunk_size, n_c)
            chunk = cluster_feats[cs:ce]                   
            sim = torch.matmul(chunk, cluster_feats.T)     
            dist = 1.0 - sim                               

            k_val = min(K_NN + 1, n_c)
            topk_dist, _ = torch.topk(dist, k=k_val, dim=1, largest=False)
            mean_nn_dist[cs:ce] = topk_dist[:, 1:].mean(dim=1)

            del sim, dist, topk_dist

        typicality[in_cluster] = (1.0 / (mean_nn_dist.cpu().numpy() + 1e-5))
        del cluster_feats, mean_nn_dist
        clear_memory()

    existing_set = set(existing_labeled_indices)
    existing_count_per_cluster = np.array([
        np.sum(np.isin(np.where(cluster_ids == c)[0],
                       list(existing_set))) if cluster_sizes[c] > 0 else 0
        for c in range(num_clusters)
    ])

    sort_keys = [(existing_count_per_cluster[c], -cluster_sizes[c]) for c in range(num_clusters)]
    sorted_clusters = sorted(range(num_clusters), key=lambda c: sort_keys[c])
    valid_clusters = [c for c in sorted_clusters if cluster_sizes[c] > 5]  # upstream MIN_CLUSTER_SIZE

    selected_indices = []
    selected_set = set(existing_labeled_indices)

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
    del existing_count_per_cluster
    clear_memory()

    return selected_indices