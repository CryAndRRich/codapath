"""BADGE (Ash et al., ICLR 2020).

Verified against `repos/badge/query_strategies/badge_sampling.py::init_centers`.

BADGE runs k-means++ seeding in the space of loss-gradient embeddings
`g_x = (onehot(argmax p) - p) (x) h(x)`. Those vectors are never formed: for a
Kronecker product the squared distance factorises as

    ||g_a - g_b||^2 = ||r_a||^2||h_a||^2 + ||r_b||^2||h_b||^2
                      - 2 (r_a . r_b)(h_a . h_b)

which is what the code below computes. First center is the largest-norm point
(deterministic, as upstream), and each later center is drawn with probability
proportional to the squared distance to the nearest chosen center. Upstream
rejects repeats by resampling; zeroing their probability is equivalent.
"""

import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from utils.progress import Stopwatch
from ..registry import register_sampler


@register_sampler("badge")
def badge_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    oracle_labels = kwargs["oracle_labels"]
    max_budget = kwargs["max_budget"]
    num_classes = kwargs["num_classes"]
    probe_epochs = kwargs.get("probe_epochs", 50)
    probe_lr = kwargs.get("probe_lr", 1e-3)
    device = kwargs["device"]
    trace = kwargs.get("trace")

    from training.probe import train_probe

    num_samples = image_embeddings.shape[0]
    step_budget = max(1, int(0.2 * max_budget))

    selected_indices: List[int] = []
    unlabeled_indices = list(range(num_samples))
    watch = Stopwatch(max_budget, "BADGE")
    round_index = 0

    while len(selected_indices) < max_budget:
        current_need = min(step_budget, max_budget - len(selected_indices))
        round_started = time.time()
        if trace is not None:
            trace.start_round(round_index)
        step_records: List[tuple] = []
        round_scores = None

        if len(selected_indices) == 0:
            chosen_local = np.random.choice(len(unlabeled_indices), current_need, replace=False)
            chosen = [unlabeled_indices[i] for i in chosen_local]
        else:
            probe = train_probe(
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

            # BADGE's gradient-embedding magnitude ||g_x||^2 = ||r||^2 ||h||^2.
            # This is the method's uncertainty term: a confident point has a
            # near-zero residual and so a near-zero gradient.
            grad_norms_sq = emb_norms_sq * res_norms_sq

            for _ in range(current_need):
                if len(chosen_local) == 0:
                    ind = torch.argmax(grad_norms_sq).item()
                else:
                    sum_D2 = torch.sum(D2)
                    if sum_D2 <= 1e-9:
                        valid = [i for i in range(len(unlabeled_indices)) if i not in chosen_local]
                        ind = int(np.random.choice(valid)) if valid else 0
                    else:
                        ind = torch.multinomial(D2 / sum_D2, 1).item()

                if trace is not None:
                    # The first centre is a deterministic argmax of the
                    # gradient norm; the rest are SAMPLED with probability
                    # proportional to D2, so D2 -- not a maximised score -- is
                    # what actually drove the pick. Recorded as `score`, with
                    # the two BADGE factors kept separately alongside it.
                    step_records.append((
                        unlabeled_indices[ind],
                        float(grad_norms_sq[ind].item()) if len(chosen_local) == 0
                        else float(D2[ind].item()),
                        float(grad_norms_sq[ind].item()),
                        None if len(chosen_local) == 0 else float(D2[ind].item()),
                    ))

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
            round_scores = grad_norms_sq.detach().cpu().numpy()
            del unlabeled_embs, unlabeled_probs_t, residuals, D2, grad_norms_sq

        if trace is not None:
            if step_records:
                for index, score, grad_norm, d2 in step_records:
                    extra = {"uncertainty": grad_norm}
                    if d2 is not None:
                        extra["coverage"] = d2
                    trace.add_step(int(index), score=score, **extra)
            else:
                # Round 1 draws at random: no model, so no score exists.
                for index in chosen:
                    trace.add_step(int(index))
            trace.add_round(
                num_selected=len(chosen),
                seconds=time.time() - round_started,
                scores=round_scores,
                weights=round_scores,
            )

        selected_indices.extend(chosen)
        chosen_set = set(chosen)
        unlabeled_indices = [i for i in unlabeled_indices if i not in chosen_set]
        watch.advance(len(chosen))
        watch.report()
        round_index += 1

    return selected_indices
