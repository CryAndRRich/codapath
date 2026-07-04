from typing import List

import numpy as np

from . import register_sampler, get_sampler


@register_sampler("refine")
def refine_sampling(**kwargs) -> List[int]:
    """REFINE (CVPR 2026): progressive ensemble pool-filtering → UncertaintyHerding.

    Stage 1 — progressive filtering (R rounds). Each round, every base strategy
    in a DIVERSE ensemble (coreset = representative, typiclust = density,
    margin = uncertainty, badge = gradient diversity) queries J random subsamples
    of the current pool; the UNION of all their picks becomes the next pool.
    Round 1 draws a fixed-size subsample (`init_subset_size`), matching the paper.

    Stage 2 — the paper's acquisition head: UncertaintyHerding over the refined
    pool (uncertainty-weighted submodular coverage), NOT plain MaxHerding.
    """
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels    = kwargs["oracle_labels"]
    max_budget       = kwargs["max_budget"]
    num_classes      = kwargs["num_classes"]
    device           = kwargs["device"]
    R                = kwargs.get("filter_rounds", 5)
    alpha            = kwargs.get("filter_alpha", 0.4)
    J                = kwargs.get("filter_batches", 5)
    init_subset      = kwargs.get("init_subset_size", 5000)
    probe_epochs     = kwargs.get("probe_epochs", 30)
    probe_lr         = kwargs.get("probe_lr", 1e-3)
    chunk_size       = kwargs.get("chunk_size", 2000)

    from .uncertainty_herding import uncertainty_herding_sampling

    num_samples = image_embeddings.shape[0]
    b = min(max_budget, num_samples)
    strategies = ["coreset", "typiclust", "margin", "badge"]

    pool_indices = list(range(num_samples))

    for r in range(R):
        if len(pool_indices) <= b:
            break

        if r == 0:
            sample_size = min(init_subset, len(pool_indices))
        else:
            sample_size = min(max(b + 1, int(alpha * len(pool_indices))), len(pool_indices))

        next_pool: set = set()
        for s_name in strategies:
            for _ in range(J):
                sub_local  = np.random.choice(len(pool_indices), sample_size, replace=False)
                sub_global = [pool_indices[i] for i in sub_local]

                local_sel = get_sampler(
                    s_name,
                    image_embeddings=image_embeddings[sub_global],
                    oracle_labels=oracle_labels[sub_global],
                    num_classes=num_classes,
                    max_budget=min(b, sample_size),
                    device=device,
                    chunk_size=chunk_size,
                    probe_epochs=probe_epochs,
                    probe_lr=probe_lr,
                )
                for li in local_sel:
                    next_pool.add(sub_global[li])

        if len(next_pool) < b:
            extras = [i for i in pool_indices if i not in next_pool]
            need = b - len(next_pool)
            next_pool.update(np.random.choice(extras, min(need, len(extras)), replace=False).tolist())

        pool_indices = list(next_pool)

    # Stage 2 — UncertaintyHerding over the refined pool.
    local_order = uncertainty_herding_sampling(
        image_embeddings=image_embeddings[pool_indices],
        oracle_labels=oracle_labels[pool_indices],
        num_classes=num_classes,
        max_budget=b,
        device=device,
        probe_epochs=probe_epochs,
        probe_lr=probe_lr,
        chunk_size=chunk_size,
    )
    selected_indices = [pool_indices[li] for li in local_order]

    if len(selected_indices) < max_budget:
        used = set(selected_indices)
        remaining = [i for i in range(num_samples) if i not in used]
        need = max_budget - len(selected_indices)
        selected_indices.extend(
            np.random.choice(remaining, min(need, len(remaining)), replace=False).tolist()
        )

    return selected_indices
