from typing import List

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans, KMeans

from set_up import clear_memory
from . import register_sampler


@register_sampler("dropquery")
def dropquery_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 50)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    dropout_ratio = kwargs.get("dropout_ratio", 0.75)
    n_dropout = kwargs.get("n_dropout", 3) 
    device = kwargs["device"]

    from trainer import train_linear

    num_samples = image_embeddings.shape[0]
    warmup_size = max(num_classes, min(int(0.2 * max_budget), num_samples // 5))
    warmup_idx = np.random.choice(num_samples, warmup_size, replace=False).tolist()

    probe = train_linear(
        image_embeddings[warmup_idx],
        oracle_labels[warmup_idx],
        num_classes, probe_epochs, probe_lr, device,
    )

    feats = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    with torch.no_grad():
        orig_logits = probe(feats)
        orig_preds = torch.argmax(orig_logits, dim=1).cpu().numpy()  
        
    inconsistency = np.zeros(num_samples, dtype=np.int32)
    for _ in range(n_dropout):
        mask = (torch.rand_like(feats) > dropout_ratio).float()
        feats_dropped = feats * mask / (1.0 - dropout_ratio + 1e-8)
        with torch.no_grad():
            drop_logits = probe(feats_dropped)
            drop_preds = torch.argmax(drop_logits, dim=1).cpu().numpy()
        inconsistency += (drop_preds != orig_preds).astype(np.int32)
        del mask, feats_dropped, drop_logits, drop_preds

    del feats
    clear_memory()
    del probe

    candidate_mask = inconsistency > (n_dropout * 0.5)
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) < max_budget:
        candidate_indices = np.argsort(inconsistency)[::-1][:max(max_budget * 3, 300)]

    cand_features = image_embeddings[candidate_indices]  
    
    B = max_budget
    if B <= 50:
        km = KMeans(n_clusters=B, n_init="auto", random_state=42)
    else:
        km = MiniBatchKMeans(n_clusters=B, batch_size=5000, n_init="auto", random_state=42)

    km.fit(cand_features)
    centroids = km.cluster_centers_        
    
    cand_t = torch.tensor(cand_features, dtype=torch.float32)
    cent_t = torch.tensor(centroids, dtype=torch.float32)
    dists = (
        (cent_t ** 2).sum(dim=1, keepdim=True)
        + (cand_t ** 2).sum(dim=1).unsqueeze(0)
        - 2.0 * torch.matmul(cent_t, cand_t.T)
    )

    selected_local = set()
    selected_indices: List[int] = []
    for b in range(B):
        sorted_local = torch.argsort(dists[b]).tolist()
        for li in sorted_local:
            if li not in selected_local:
                selected_local.add(li)
                selected_indices.append(int(candidate_indices[li]))
                break

    return selected_indices