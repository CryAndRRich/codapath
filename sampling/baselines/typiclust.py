"""TypiClust (Hacohen et al., ICML 2022).

Verified against `repos/typiclust/deep-al/pycls/al/typiclust.py`. Matching
constants: `MIN_CLUSTER_SIZE=5`, `MAX_NUM_CLUSTERS=500`, `K_NN=20`, KMeans
below 50 clusters and MiniBatchKMeans(batch_size=5000) above.

Two details that are easy to lose:

* typicality is recomputed on the SHRINKING set of still-unlabeled members of
  a cluster at every single pick, not once per cluster. Density among the
  remaining candidates changes as their neighbours get taken.
* `n_init` is pinned to the sklearn defaults upstream inherited (10 for KMeans,
  3 for MiniBatchKMeans). Today's `n_init="auto"` resolves to 1, which would
  quietly give the clustering this whole method rests on a single restart.

Upstream measures neighbour distance with `faiss.IndexFlatL2` (squared L2) on
L2-normalized features; `1 - cosine` here is the same quantity up to a factor
of two, so the argmax over typicality is unchanged.
"""

import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans

from utils.runtime import clear_memory
from ..registry import register_sampler


@register_sampler("typiclust")
def typiclust_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    max_budget = kwargs["max_budget"]
    device = kwargs["device"]
    chunk_size = kwargs["chunk_size"]
    existing_labeled_indices = kwargs.get("existing_labeled_indices", [])
    trace = kwargs.get("trace")

    num_samples = image_embeddings.shape[0]
    K_NN = kwargs.get("k_nn", 20)

    features_tensor = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features_tensor = F.normalize(features_tensor, p=2, dim=1)
    features_np = features_tensor.cpu().numpy()

    num_clusters = min(len(existing_labeled_indices) + max_budget, 500)

    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    else:
        km = MiniBatchKMeans(
            n_clusters=num_clusters, batch_size=5000, n_init=3, random_state=42
        )

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

    def _typicality(features: torch.Tensor, indices: List[int]) -> np.ndarray:
        n_c = len(indices)
        if n_c <= 1:
            return np.zeros(n_c, dtype=np.float32)

        feats = features[indices]
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

    started = time.time()
    if trace is not None:
        trace.start_round(0)

    i = 0
    with tqdm(total=max_budget, desc="TypiClust Selection") as pbar:
        while len(selected_indices) < max_budget:
            cluster = valid_clusters[i % len(valid_clusters)]
            available_indices = cluster_members[cluster]

            if len(available_indices) > 0:
                cluster_typ = _typicality(features_tensor, available_indices)
                best_local_idx = int(np.argmax(cluster_typ))
                best_global_idx = available_indices[best_local_idx]

                if trace is not None:
                    # Typicality is TypiClust's whole acquisition value: local
                    # density among the still-unlabeled members of this
                    # cluster. It is a coverage-family quantity (no model, no
                    # uncertainty), so it is recorded under `coverage`.
                    best_typ = float(cluster_typ[best_local_idx])
                    runner_up = (
                        float(np.sort(cluster_typ)[-2]) if cluster_typ.size > 1 else None
                    )
                    trace.add_step(
                        int(best_global_idx),
                        score=best_typ,
                        margin_to_runner_up=(
                            None if runner_up is None else best_typ - runner_up
                        ),
                        coverage=best_typ,
                        cluster=float(cluster),
                    )

                selected_indices.append(best_global_idx)
                selected_set.add(best_global_idx)
                cluster_members[cluster] = [
                    idx for idx in available_indices if idx != best_global_idx
                ]
                pbar.update(1)

            i += 1
            if i > len(valid_clusters) * max_budget:
                break

    num_random_fallback = 0
    if len(selected_indices) < max_budget:
        remaining = list(set(range(num_samples)) - selected_set)
        missing = max_budget - len(selected_indices)
        fallback = np.random.choice(remaining, missing, replace=False)
        selected_indices.extend(fallback.tolist())
        num_random_fallback = len(fallback)
        if trace is not None:
            # Random top-up when the clusters ran dry: no typicality behind
            # these, so they carry no score -- and the count is worth keeping,
            # since a large one means the clustering, not typicality, decided
            # most of this budget.
            for index in fallback.tolist():
                trace.add_step(int(index))

    if trace is not None:
        trace.add_round(
            num_selected=len(selected_indices),
            seconds=time.time() - started,
            num_clusters=float(num_clusters),
            valid_clusters=float(len(valid_clusters)),
            random_fallback=float(num_random_fallback),
        )

    del features_tensor
    del existing_count_per_cluster
    clear_memory()

    return selected_indices
