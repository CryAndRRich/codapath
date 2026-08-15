from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans

from set_up import clear_memory
from . import register_sampler


def _typicality(features: torch.Tensor, indices: List[int], k_nn: int,
                chunk_size: int) -> np.ndarray:
    # Recomputed on the current (shrinking) candidate pool of a single cluster —
    # matches upstream TypiClust, where density estimates are relative to the
    # still-unlabeled members only, not a stale global/whole-cluster snapshot.
    n_c = len(indices)
    if n_c <= 1:
        return np.zeros(n_c, dtype=np.float32)

    feats = features[indices]
    k_val = max(1, min(k_nn, n_c // 2))  # upstream: min(K_NN, len(indices) // 2)
    mean_nn_dist = torch.zeros(n_c, device=features.device)

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


def _typiclust_phase(features: torch.Tensor,
                     num_select: int,
                     k_nn: int = 20,
                     chunk_size: int = 5000) -> List[int]:
    num_samples = features.shape[0]
    feats_np = features.cpu().numpy()

    if num_select <= 50:
        km = KMeans(n_clusters=num_select, n_init="auto", random_state=42)
    else:
        km = MiniBatchKMeans(n_clusters=num_select, batch_size=5000,
                             n_init="auto", random_state=42)

    cluster_ids = km.fit_predict(feats_np)
    cluster_sizes = np.bincount(cluster_ids, minlength=num_select)
    sorted_clusters = np.argsort(cluster_sizes)[::-1]
    valid_clusters = [c for c in sorted_clusters if cluster_sizes[c] > 5]  # upstream MIN_CLUSTER_SIZE
    if not valid_clusters:
        # see typiclust.py: MIN_CLUSTER_SIZE=5 assumes a large pool; on a small
        # pool/subsample every cluster can legitimately be smaller than that.
        valid_clusters = [c for c in sorted_clusters if cluster_sizes[c] > 0]
        print(f"[TCM WARNING] no cluster exceeds MIN_CLUSTER_SIZE=5 "
              f"(pool={num_samples}, num_clusters={num_select}); "
              f"falling back to all non-empty clusters")

    cluster_members = {
        c: [int(idx) for idx in np.where(cluster_ids == c)[0]]
        for c in valid_clusters
    }

    selected: List[int] = []
    selected_set: set = set()
    i = 0
    with tqdm(total=num_select, desc="TCM Phase-1 (TypiClust)", leave=False) as pbar:
        while len(selected) < num_select:
            cluster = valid_clusters[i % len(valid_clusters)]
            available = cluster_members[cluster]
            if available:
                typ = _typicality(features, available, k_nn, chunk_size)
                best = available[int(np.argmax(typ))]
                selected.append(best)
                selected_set.add(best)
                cluster_members[cluster] = [idx for idx in available if idx != best]
                pbar.update(1)
            i += 1
            if i > len(valid_clusters) * num_select:
                break

    if len(selected) < num_select:
        remaining = list(set(range(num_samples)) - selected_set)
        missing = num_select - len(selected)
        selected.extend(np.random.choice(remaining, missing, replace=False).tolist())

    return selected


@register_sampler("tcm")
def tcm_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 50)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]
    k_nn = kwargs.get("k_nn", 20)
    chunk_size = kwargs.get("chunk_size", 5000)
    transition_budget = kwargs.get("transition_budget", min(2 * num_classes, max_budget))
    transition_budget = min(transition_budget, max_budget)

    from trainer import train_linear

    features = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features = F.normalize(features, p=2, dim=1)

    phase1 = _typiclust_phase(features, transition_budget, k_nn=k_nn, chunk_size=chunk_size)

    if transition_budget >= max_budget:
        del features
        clear_memory()
        return phase1[:max_budget]

    remaining_budget = max_budget - transition_budget
    phase1_set = set(phase1)
    unlabeled_indices = [i for i in range(len(image_embeddings)) if i not in phase1_set]

    # Paper (references/TCM.md, Sec. 4.1 "Setup"): "classifier training
    # following EACH sampling step" — Margin retrains on the FULL accumulated
    # labeled set every acquisition, exactly like basic_samplers.py's own
    # margin_sampling. A single train-once-then-freeze cut (the old behavior
    # here) is not what the paper describes for its Margin phase.
    #
    # Step size (Sec. 3.2: "we use a step size equal to the size of the
    # initial budget of each setting") is `transition_budget` itself, not a
    # fraction of `max_budget`.
    step_budget = max(1, transition_budget)
    phase2: List[int] = []

    while len(phase2) < remaining_budget:
        current_need = min(step_budget, remaining_budget - len(phase2))
        labeled_so_far = phase1 + phase2

        probe = train_linear(
            image_embeddings[labeled_so_far],
            oracle_labels[labeled_so_far],
            num_classes, probe_epochs, probe_lr, device,
        )
        probs = probe.predict_proba(image_embeddings[unlabeled_indices], device)
        s_probs = np.sort(probs, axis=1)
        margin = s_probs[:, -1] - s_probs[:, -2]
        order = np.argsort(margin)
        chosen = [unlabeled_indices[i] for i in order[:current_need]]
        del probe

        phase2.extend(chosen)
        chosen_set = set(chosen)
        unlabeled_indices = [i for i in unlabeled_indices if i not in chosen_set]

    del features
    clear_memory()

    return phase1 + phase2