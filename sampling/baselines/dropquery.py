"""DropQuery (TMLR 2024).

Verified against `repos/dropquery/ALFM/src/query_strategies/dropout.py` and
`ALFM/src/clustering/kmeans.py`.

Candidates are points whose prediction flips under dropout. The threshold is
NOT a fixed 50%: it starts at `n_dropout // 2` and is lowered until at least
`25 * acq_size` points qualify, or it reaches zero. Survivors are
L2-normalized, clustered into `acq_size` clusters, and one point is taken per
cluster -- the member closest to its OWN centroid, a Voronoi pick rather than a
global nearest-neighbour scan. `n_init=1` matches upstream's single faiss
k-means run with an explicit k-means++ init.

One forced adaptation: upstream perturbs an MLP head's hidden activations,
which a single linear probe does not have. Dropout is applied to the input
features instead -- the closest analogue under this project's frozen-backbone
protocol.
"""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans, KMeans

from utils.runtime import clear_memory
from utils.progress import progress
from ..registry import register_sampler


@register_sampler("dropquery")
def dropquery_sampling(**kwargs) -> List[int]:
    """DropQuery (dal_toolbox, TMLR 2024).

    Upstream (repos/refine/dal_toolbox/active_learning/strategies/dropquery.py
    + examples/active_learning.py) is a standard pool-based AL loop: cycle 0's
    labeled set comes from plain random init (DropQuery's own `query()` is
    NEVER called with an empty labeled set — it always assumes an
    already-trained model), then EVERY subsequent cycle retrains the model
    from scratch on the full accumulated labeled set before calling `query()`
    again. This mirrors that: round 0 is random, rounds 1..num_rounds-1 each
    retrain the probe on ALL selected indices so far before recomputing the
    dropout-inconsistency candidates for that round.
    """
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 50)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    dropout_ratio = kwargs.get("dropout_ratio", 0.75)
    n_dropout = kwargs.get("n_dropout", 3)
    num_rounds = kwargs.get("num_rounds", 5)
    device = kwargs["device"]

    from training.probe import train_probe

    num_samples = image_embeddings.shape[0]
    rounds = max(1, min(num_rounds, max_budget))
    base, remainder = divmod(max_budget, rounds)
    round_sizes = [base + (1 if r < remainder else 0) for r in range(rounds)]

    selected_indices: List[int] = []
    selected_set: set = set()

    for round_idx in progress(range(rounds), desc="DropQuery rounds", total=rounds):
        n_select = round_sizes[round_idx]
        if n_select <= 0:
            continue

        unlabeled_indices = [i for i in range(num_samples) if i not in selected_set]

        if round_idx == 0:
            chosen = np.random.choice(
                unlabeled_indices, min(n_select, len(unlabeled_indices)), replace=False
            ).tolist()
            selected_indices.extend(chosen)
            selected_set.update(chosen)
            continue

        probe = train_probe(
            image_embeddings[selected_indices],
            oracle_labels[selected_indices],
            num_classes, probe_epochs, probe_lr, device,
        )

        feats = torch.tensor(
            image_embeddings[unlabeled_indices], device=device, dtype=torch.float32
        )
        with torch.no_grad():
            orig_logits = probe(feats)
            orig_preds = torch.argmax(orig_logits, dim=1).cpu().numpy()

        inconsistency = np.zeros(len(unlabeled_indices), dtype=np.int32)
        for _ in range(n_dropout):
            mask = (torch.rand_like(feats) > dropout_ratio).float()
            feats_dropped = feats * mask / (1.0 - dropout_ratio + 1e-8)
            with torch.no_grad():
                drop_logits = probe(feats_dropped)
                drop_preds = torch.argmax(drop_logits, dim=1).cpu().numpy()
            inconsistency += (drop_preds != orig_preds).astype(np.int32)
            del mask, feats_dropped, drop_logits, drop_preds

        del feats, probe
        clear_memory()

        # upstream DropQuery (dal-toolbox): shrink the mismatch threshold from
        # n_dropout//2 down to (at worst) 0 until at least 25x this round's
        # query size qualifies as "uncertain" — NOT a fixed >50% cutoff.
        thresh = n_dropout // 2
        while (inconsistency > thresh).sum() < 25 * n_select and thresh > 0:
            thresh -= 1
        candidate_local = np.nonzero(inconsistency > thresh)[0]

        if len(candidate_local) < n_select:
            # too few uncertain points this round: take them all + random fill,
            # no clustering.
            remaining_local = np.setdiff1d(np.arange(len(unlabeled_indices)), candidate_local)
            delta = n_select - len(candidate_local)
            fill_local = np.random.choice(
                remaining_local, min(delta, len(remaining_local)), replace=False
            )
            chosen_local = np.concatenate([candidate_local, fill_local]).astype(int)
            chosen = [unlabeled_indices[i] for i in chosen_local]
        else:
            candidate_global = [unlabeled_indices[i] for i in candidate_local]
            cand_features = image_embeddings[candidate_global]
            cand_t = F.normalize(torch.tensor(cand_features, dtype=torch.float32), p=2, dim=1)
            cand_features = cand_t.numpy()

            if n_select <= 50:
                km = KMeans(n_clusters=n_select, n_init=1, random_state=42)
            else:
                km = MiniBatchKMeans(n_clusters=n_select, batch_size=5000, n_init=1, random_state=42)

            km.fit(cand_features)
            all_dist = km.transform(cand_features)  # (num_candidates, n_select)
            cluster_idx = np.argmin(all_dist, axis=1)
            dist_to_own_centroid = all_dist[np.arange(len(cand_features)), cluster_idx]

            # one point per Voronoi cluster (closest to its own centroid) —
            # matches upstream cluster_features(); empty clusters are
            # backfilled with random candidates rather than letting a
            # neighboring cluster's point double up.
            selected_local: List[int] = []
            for b in range(n_select):
                idx_in_cluster = np.nonzero(cluster_idx == b)[0]
                if len(idx_in_cluster) > 0:
                    best = idx_in_cluster[np.argmin(dist_to_own_centroid[idx_in_cluster])]
                    selected_local.append(int(best))

            selected_local_set = set(selected_local)
            remaining_local = [i for i in range(len(cand_features)) if i not in selected_local_set]
            np.random.shuffle(remaining_local)
            delta = n_select - len(selected_local)
            selected_local.extend(remaining_local[:delta])

            chosen = [candidate_global[i] for i in selected_local]

        selected_indices.extend(chosen)
        selected_set.update(chosen)

    return selected_indices
