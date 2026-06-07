from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from . import register_sampler


@register_sampler("badge")
def badge_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 100)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]

    from trainer import train_linear

    num_samples = image_embeddings.shape[0]

    all_indices = list(range(num_samples))

    if len(selected_indices := kwargs.get("existing_labeled_indices", [])) == 0:
        chosen = np.random.choice(num_samples, max_budget, replace=False).tolist()
        return chosen

    labeled_features = image_embeddings[selected_indices]
    labeled_labels = oracle_labels[selected_indices]

    probe = train_linear(
        labeled_features, labeled_labels, num_classes,
        probe_epochs, probe_lr, device
    )

    unlabeled_indices = [i for i in all_indices if i not in set(selected_indices)]
    unlabeled_probs = probe.predict_proba(image_embeddings[unlabeled_indices], device)
    del probe

    unlabeled_embs = torch.tensor(
        image_embeddings[unlabeled_indices], device=device, dtype=torch.float32
    )
    unlabeled_probs_t = torch.tensor(unlabeled_probs, device=device, dtype=torch.float32)

    preds = torch.argmax(unlabeled_probs_t, dim=1)
    residuals = -unlabeled_probs_t
    residuals = residuals + F.one_hot(preds, num_classes=num_classes).to(residuals.dtype)

    emb_norms_sq = torch.sum(unlabeled_embs ** 2, dim=1)
    res_norms_sq = torch.sum(residuals ** 2, dim=1)

    chosen_local = []
    D2 = torch.full((len(unlabeled_indices),), float("inf"), device=device)

    for _ in range(min(max_budget, len(unlabeled_indices))):
        if len(chosen_local) == 0:
            grad_magnitudes = emb_norms_sq * res_norms_sq
            ind = torch.argmax(grad_magnitudes).item()
        else:
            sum_D2 = torch.sum(D2)
            if sum_D2 <= 1e-9:
                valid = [i for i in range(len(D2)) if i not in chosen_local]
                ind = np.random.choice(valid) if valid else 0
            else:
                prob_dist = D2 / sum_D2  
                ind = torch.multinomial(prob_dist, 1).item()

        chosen_local.append(ind)

        c_emb = unlabeled_embs[ind]
        c_res = residuals[ind]
        c_emb_norm_sq = emb_norms_sq[ind]
        c_res_norm_sq = res_norms_sq[ind]

        dot_res = torch.matmul(residuals, c_res)
        dot_emb = torch.matmul(unlabeled_embs, c_emb)
        dist_sq = torch.clamp(
            res_norms_sq * emb_norms_sq + c_res_norm_sq * c_emb_norm_sq - 2 * dot_res * dot_emb,
            min=0.0
        )

        D2 = torch.minimum(D2, dist_sq)
        D2[chosen_local] = 0.0 

    chosen = [unlabeled_indices[i] for i in chosen_local]
    del unlabeled_embs, unlabeled_probs_t, residuals, D2

    return chosen