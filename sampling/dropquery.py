"""
DropQuery: Feature-level dropout uncertainty + K-means diversity.

Algorithm (per budget B):
  1. Apply M=3 feature-level dropout perturbations (ρ=0.75) to each unlabeled embedding.
  2. Run the linear classifier on original and perturbed features; keep points where
     > 50% of perturbed predictions differ from the original → candidate set Z_c.
  3. K-means(B) on Z_c → select the point closest to each centroid.

Requires a LinearProbe trained on oracle-labelled warm-up data.  The probe is trained
ONCE on random warm-up samples and reused across all budgets (the K-means step is
re-run per budget, so this sampler is PER_BUDGET).

Reference: Gupte et al., arXiv:2401.14555 (2024)
GitHub:    https://github.com/sanketx/AL-foundation-models
"""
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans, KMeans

from set_up import clear_memory
from . import register_sampler


@register_sampler("dropquery")
def dropquery_sampling(**kwargs) -> List[int]:
    """
    DropQuery — PER_BUDGET (re-run per budget; K-means depends on B).

    For cold-start, a warm-up probe is trained on a small random subset of the pool
    using oracle_labels.  The same probe is then used for dropout scoring across all
    budget values within a run.
    """
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 50)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    dropout_ratio = kwargs.get("dropout_ratio", 0.75)
    n_dropout = kwargs.get("n_dropout", 3)   # M in the paper (robust to M∈{3,5,7,9})
    device = kwargs["device"]

    from trainer import train_linear

    num_samples = image_embeddings.shape[0]
    # Warm-up: random subset (≥ num_classes, ≤ 20% of budget)
    warmup_size = max(num_classes, min(int(0.2 * max_budget), num_samples // 5))
    warmup_idx = np.random.choice(num_samples, warmup_size, replace=False).tolist()

    probe = train_linear(
        image_embeddings[warmup_idx],
        oracle_labels[warmup_idx],
        num_classes, probe_epochs, probe_lr, device,
    )

    # ── Dropout uncertainty scoring ───────────────────────────────────────────
    feats = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    # Original predictions (no dropout)
    with torch.no_grad():
        orig_logits = probe(feats)
        orig_preds = torch.argmax(orig_logits, dim=1).cpu().numpy()  # (N,)

    # Count how many out of M dropout passes give a different prediction
    inconsistency = np.zeros(num_samples, dtype=np.int32)
    for _ in range(n_dropout):
        # Feature-level dropout: randomly zero ρ fraction of dimensions
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

    # ── Candidate set: inconsistency > 50% of M passes ───────────────────────
    candidate_mask = inconsistency > (n_dropout * 0.5)
    candidate_indices = np.where(candidate_mask)[0]

    # Fallback: if too few candidates, use all points sorted by inconsistency
    if len(candidate_indices) < max_budget:
        candidate_indices = np.argsort(inconsistency)[::-1][:max(max_budget * 3, 300)]

    cand_features = image_embeddings[candidate_indices]   # (|Z_c|, D)

    # ── K-means(B) diversity on candidates ───────────────────────────────────
    B = max_budget
    if B <= 50:
        km = KMeans(n_clusters=B, n_init="auto", random_state=42)
    else:
        km = MiniBatchKMeans(n_clusters=B, batch_size=5000, n_init="auto", random_state=42)

    km.fit(cand_features)
    centroids = km.cluster_centers_           # (B, D)

    # Select the candidate closest to each centroid
    cand_t = torch.tensor(cand_features, dtype=torch.float32)
    cent_t = torch.tensor(centroids, dtype=torch.float32)
    # Squared L2 distances: (B, |Z_c|)
    # ||c - z||^2 = ||c||^2 + ||z||^2 - 2 c·z
    dists = (
        (cent_t ** 2).sum(dim=1, keepdim=True)
        + (cand_t ** 2).sum(dim=1).unsqueeze(0)
        - 2.0 * torch.matmul(cent_t, cand_t.T)
    )  # (B, |Z_c|)

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
