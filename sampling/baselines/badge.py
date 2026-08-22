from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from .. import register_sampler


@register_sampler("badge")
def badge_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 50)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]

    from training.probe import train_linear

    num_samples = image_embeddings.shape[0]
    step_budget = max(1, int(0.2 * max_budget))

    selected_indices: List[int] = []
    unlabeled_indices = list(range(num_samples))

    while len(selected_indices) < max_budget:
        current_need = min(step_budget, max_budget - len(selected_indices))

        if len(selected_indices) == 0:
            chosen_local = np.random.choice(len(unlabeled_indices), current_need, replace=False)
            chosen = [unlabeled_indices[i] for i in chosen_local]
        else:
            probe = train_linear(
                image_embeddings[selected_indices],
                oracle_labels[selected_indices],
                num_classes, probe_epochs, probe_lr, device,
            )
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

            chosen_local: List[int] = []
            D2 = torch.full((len(unlabeled_indices),), float("inf"), device=device)

            for _ in range(current_need):
                if len(chosen_local) == 0:
                    ind = torch.argmax(emb_norms_sq * res_norms_sq).item()
                else:
                    sum_D2 = torch.sum(D2)
                    if sum_D2 <= 1e-9:
                        valid = [i for i in range(len(unlabeled_indices)) if i not in chosen_local]
                        ind = int(np.random.choice(valid)) if valid else 0
                    else:
                        ind = torch.multinomial(D2 / sum_D2, 1).item()

                chosen_local.append(ind)

                c_res = residuals[ind]
                c_emb = unlabeled_embs[ind]
                dot_res = torch.matmul(residuals, c_res)
                dot_emb = torch.matmul(unlabeled_embs, c_emb)
                dist_sq = torch.clamp(
                    res_norms_sq * emb_norms_sq
                    + res_norms_sq[ind] * emb_norms_sq[ind]
                    - 2 * dot_res * dot_emb,
                    min=0.0,
                )
                D2 = torch.minimum(D2, dist_sq)
                D2[chosen_local] = 0.0

            chosen = [unlabeled_indices[i] for i in chosen_local]
            del unlabeled_embs, unlabeled_probs_t, residuals, D2

        selected_indices.extend(chosen)
        chosen_set = set(chosen)
        unlabeled_indices = [i for i in unlabeled_indices if i not in chosen_set]

    return selected_indices
