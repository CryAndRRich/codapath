from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans

from utils.runtime import clear_memory
from .. import register_sampler


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

    existing_set = set(existing_labeled_indices)
    existing_count_per_cluster = np.array([
        np.sum(np.isin(np.where(cluster_ids == c)[0],
                       list(existing_set))) if cluster_sizes[c] > 0 else 0
        for c in range(num_clusters)
    ])

    sort_keys = [(existing_count_per_cluster[c], -cluster_sizes[c]) for c in range(num_clusters)]
    sorted_clusters = sorted(range(num_clusters), key=lambda c: sort_keys[c])
    valid_clusters = [c for c in sorted_clusters if cluster_sizes[c] > 5]  # upstream MIN_CLUSTER_SIZE
    if not valid_clusters:
        # MIN_CLUSTER_SIZE=5 assumes clusters carved out of a large pool; on a
        # small pool (e.g. a small subsample) every cluster can legitimately be
        # smaller than that. Fall back to any non-empty cluster instead of
        # crashing on `i % len(valid_clusters)` below.
        valid_clusters = [c for c in sorted_clusters if cluster_sizes[c] > 0]
        print(f"[TypiClust WARNING] no cluster exceeds MIN_CLUSTER_SIZE=5 "
              f"(pool={num_samples}, num_clusters={num_clusters}); "
              f"falling back to all non-empty clusters")

    # Per-cluster pool of still-unlabeled members, shrunk in place as points get picked —
    # upstream TypiClust recomputes typicality from this shrinking pool on every single
    # selection (not once up front), since density estimates among the remaining candidates
    # change as their nearest neighbors get removed.
    cluster_members = {
        c: [int(idx) for idx in np.where(cluster_ids == c)[0] if idx not in existing_set]
        for c in valid_clusters
    }

    def _typicality(indices: List[int]) -> np.ndarray:
        n_c = len(indices)
        if n_c <= 1:
            return np.zeros(n_c, dtype=np.float32)

        feats = features_tensor[indices]
        k_val = max(1, min(K_NN, n_c // 2))  # upstream: min(K_NN, len(indices) // 2)
        mean_nn_dist = torch.zeros(n_c, device=device)

        for cs in range(0, n_c, chunk_size):
            ce = min(cs + chunk_size, n_c)
            chunk = feats[cs:ce]
            sim = torch.matmul(chunk, feats.T)
            dist = 1.0 - sim

            k_search = min(k_val + 1, n_c)
            topk_dist, _ = torch.topk(dist, k=k_search, dim=1, largest=False)
            mean_nn_dist[cs:ce] = topk_dist[:, 1:].mean(dim=1)

            del sim, dist, topk_dist

        return (1.0 / (mean_nn_dist.cpu().numpy() + 1e-5))

    selected_indices = []
    selected_set = set(existing_labeled_indices)

    i = 0
    with tqdm(total=max_budget, desc="TypiClust Selection") as pbar:
        while len(selected_indices) < max_budget:
            cluster = valid_clusters[i % len(valid_clusters)]
            available_indices = cluster_members[cluster]

            if len(available_indices) > 0:
                cluster_typ = _typicality(available_indices)
                best_local_idx = int(np.argmax(cluster_typ))
                best_global_idx = available_indices[best_local_idx]

                selected_indices.append(best_global_idx)
                selected_set.add(best_global_idx)
                cluster_members[cluster] = [
                    idx for idx in available_indices if idx != best_global_idx
                ]
                pbar.update(1)

            i += 1
            if i > len(valid_clusters) * max_budget:
                break

    if len(selected_indices) < max_budget:
        remaining = list(set(range(num_samples)) - selected_set)
        missing = max_budget - len(selected_indices)
        fallback = np.random.choice(remaining, missing, replace=False)
        selected_indices.extend(fallback.tolist())

    del features_tensor
    del existing_count_per_cluster
    clear_memory()

    return selected_indices