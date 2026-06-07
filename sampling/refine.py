from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from . import register_sampler


def _coreset_mini(feats: np.ndarray, b: int, device: torch.device) -> List[int]:
    n = feats.shape[0]
    b = min(b, n)
    f = torch.tensor(feats, device=device, dtype=torch.float32)
    f = F.normalize(f, p=2, dim=1)

    selected = [int(np.random.randint(0, n))]
    min_dist = torch.full((n,), float("inf"), device=device)

    for _ in range(b - 1):
        last = f[selected[-1]].unsqueeze(0)
        sim = torch.matmul(f, last.T).squeeze(1)
        dist = 1.0 - sim
        min_dist = torch.minimum(min_dist, dist)
        min_dist[selected] = -1.0
        selected.append(int(torch.argmax(min_dist).item()))

    del f, min_dist
    return selected


def _typiclust_mini(feats: np.ndarray, b: int) -> List[int]:
    from sklearn.cluster import MiniBatchKMeans, KMeans

    n = feats.shape[0]
    b = min(b, n)

    feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    chunk = min(1000, n)
    typicality = np.zeros(n, dtype=np.float32)
    k_nn = min(20, n - 1)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sim = feats_norm[s:e] @ feats_norm.T         
        sim[:, s:e] -= 2 * np.eye(e - s, n, k=s)   
        topk_sim = np.sort(sim, axis=1)[:, :-1][:, -k_nn:]
        typicality[s:e] = topk_sim.mean(axis=1)

    if b <= 50:
        km = KMeans(n_clusters=b, n_init="auto", random_state=42)
    else:
        km = MiniBatchKMeans(n_clusters=b, batch_size=5000, n_init="auto", random_state=42)
    cluster_ids = km.fit_predict(feats_norm)
    cluster_sizes = np.bincount(cluster_ids, minlength=b)
    order = np.argsort(cluster_sizes)[::-1]

    selected: List[int] = []
    sel_set: set = set()
    i = 0
    while len(selected) < b:
        c = order[i % len(order)]
        members = [j for j in np.where(cluster_ids == c)[0] if j not in sel_set]
        if members:
            best = members[int(np.argmax(typicality[members]))]
            selected.append(best)
            sel_set.add(best)
        i += 1
        if i > b * len(order):
            break

    if len(selected) < b:
        remaining = list(set(range(n)) - sel_set)
        selected.extend(np.random.choice(remaining, b - len(selected), replace=False).tolist())
    return selected


def _margin_mini(feats: np.ndarray,
                 labels: np.ndarray,
                 b: int,
                 device: torch.device,
                 num_classes: int,
                 probe_epochs: int = 30,
                 probe_lr: float = 1e-3) -> List[int]:
    from trainer import train_linear

    n = feats.shape[0]
    b = min(b, n)
    warmup = max(num_classes, n // 5)
    warmup_idx = np.random.choice(n, warmup, replace=False)

    probe = train_linear(feats[warmup_idx], labels[warmup_idx],
                         num_classes, probe_epochs, probe_lr, device)
    probs = probe.predict_proba(feats, device)
    del probe
    sp = np.sort(probs, axis=1)
    margin = sp[:, -1] - sp[:, -2]

    selected_set = set()
    order = np.argsort(margin) 
    selected = []
    for idx in order:
        if len(selected) >= b:
            break
        if idx not in selected_set:
            selected.append(int(idx))
            selected_set.add(int(idx))
    return selected


def _maxherding(feats_np: np.ndarray,
                budget: int,
                device: torch.device,
                chunk_size: int = 2000) -> List[int]:
    n = feats_np.shape[0]
    budget = min(budget, n)
    feats = torch.tensor(feats_np, device=device, dtype=torch.float32)
    feats = F.normalize(feats, p=2, dim=1)
    max_sim = torch.full((n,), -float("inf"), device=device)

    selected: List[int] = []
    sel_set: set = set()

    for _ in range(budget):
        best_idx = -1
        best_gain = -float("inf")
        best_col = None

        for cs in range(0, n, chunk_size):
            ce = min(cs + chunk_size, n)
            chunk = feats[cs:ce]                         
            sim = torch.matmul(feats, chunk.T)           
            gain = torch.clamp(sim - max_sim.unsqueeze(1), min=0.0)
            scores = gain.sum(dim=0)                    
            for si in sel_set:
                if cs <= si < ce:
                    scores[si - cs] = -float("inf")
            local_best = int(torch.argmax(scores).item())
            if scores[local_best].item() > best_gain:
                best_gain = scores[local_best].item()
                best_idx = cs + local_best
                best_col = sim[:, local_best].clone()
            del chunk, sim, gain, scores
            clear_memory()

        if best_idx >= 0 and best_idx not in sel_set:
            selected.append(best_idx)
            sel_set.add(best_idx)
            max_sim = torch.maximum(max_sim, best_col)
            del best_col
            clear_memory()
        else:
            break

    del feats, max_sim
    clear_memory()
    return selected


@register_sampler("refine")
def refine_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    device = kwargs["device"]
    R = kwargs.get("filter_rounds", 3)       
    alpha = kwargs.get("filter_alpha", 0.4)    
    J = kwargs.get("filter_batches", 5)       
    probe_epochs = kwargs.get("probe_epochs", 30)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    chunk_size = kwargs.get("chunk_size", 2000)

    num_samples = image_embeddings.shape[0]
    b = max_budget

    pool_indices = list(range(num_samples)) 

    for r in tqdm(range(R), desc="REFINE Filtering", leave=True):
        if len(pool_indices) <= b:
            break

        next_pool_set: set = set()

        strategies = ["coreset", "typiclust", "margin"]

        for s_name in strategies:
            for _ in range(J):
                sample_size = max(b + 1, int(alpha * len(pool_indices)))
                sample_size = min(sample_size, len(pool_indices))
                sub_local = np.random.choice(len(pool_indices), sample_size, replace=False)
                sub_global = [pool_indices[i] for i in sub_local]

                sub_feats = image_embeddings[sub_global] 
                sub_labels = oracle_labels[sub_global]

                if s_name == "coreset":
                    local_sel = _coreset_mini(sub_feats, b, device)
                elif s_name == "typiclust":
                    local_sel = _typiclust_mini(sub_feats, b)
                else:  
                    local_sel = _margin_mini(sub_feats, sub_labels, b,
                                             device, num_classes,
                                             probe_epochs, probe_lr)

                for li in local_sel:
                    next_pool_set.add(sub_global[li])

        if len(next_pool_set) < b:
            missing = b - len(next_pool_set)
            extras = [i for i in pool_indices if i not in next_pool_set]
            next_pool_set.update(np.random.choice(extras, missing, replace=False).tolist())

        pool_indices = list(next_pool_set)

    pool_feats = image_embeddings[pool_indices]    
    local_order = _maxherding(pool_feats, max_budget, device, chunk_size=chunk_size)

    selected_indices = [pool_indices[li] for li in local_order]

    if len(selected_indices) < max_budget:
        used = set(selected_indices)
        remaining = [i for i in range(num_samples) if i not in used]
        n_fill = max_budget - len(selected_indices)
        selected_indices.extend(
            np.random.choice(remaining, min(n_fill, len(remaining)), replace=False).tolist()
        )

    return selected_indices