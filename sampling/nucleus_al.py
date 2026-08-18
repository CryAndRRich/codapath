"""Nucleus-aware iterative AL with full-image DINOv2 coverage.

Coverage is deliberately fixed to the original full-patch DINOv2 space. The
only experimental axis is how a frozen CellViT-derived view changes the
per-target uncertainty weights used by weighted facility-location greedy.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from trainer import LinearProbe, train_linear
from nucleus.uncertainty import (
    js_disagreement_from_logits,
    margin_uncertainty_from_logits,
    row_layer_norm,
)
from . import register_sampler
from .scalpel import (
    _adaptive_sigma,
    _greedy_coverage_batch,
    _rank_normalize,
)


VALID_SOURCES = {"crop_dino", "cellvit_embedding"}
VALID_UNCERTAINTIES = {
    "cell_margin",
    "disagreement",
    "fusion_concat",
    "fusion_add",
}


def _can_train(labels: np.ndarray) -> bool:
    return len(labels) >= 2 and len(np.unique(labels)) >= 2


def _fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    selected: List[int],
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    valid: Optional[np.ndarray] = None,
) -> Optional[LinearProbe]:
    indices = np.asarray(selected, dtype=np.int64)
    if valid is not None:
        indices = indices[valid[indices]]
    y = labels[indices]
    if not _can_train(y):
        return None
    return train_linear(
        features[indices], y, num_classes, epochs, lr, device,
        weight_decay=weight_decay,
    )


def _fusion_features(
    image_logits: np.ndarray,
    cell_logits: np.ndarray,
    mode: str,
    normalize_logits: bool = False,
) -> np.ndarray:
    image_norm = row_layer_norm(image_logits) if normalize_logits else image_logits
    cell_norm = row_layer_norm(cell_logits) if normalize_logits else cell_logits
    if mode == "fusion_concat":
        return np.concatenate([image_norm, cell_norm], axis=1)
    if mode == "fusion_add":
        return image_norm + cell_norm
    raise ValueError(f"Unsupported fusion mode: {mode}")


@register_sampler("nucleus_al")
def nucleus_al_sampling(**kwargs) -> List[int]:
    dino_np = np.asarray(kwargs["image_embeddings"], dtype=np.float32)
    nucleus_np = np.asarray(kwargs["nucleus_embeddings"], dtype=np.float32)
    reliability = np.asarray(kwargs["nucleus_reliability"], dtype=np.float32)
    labels = np.asarray(kwargs["oracle_labels"])
    num_classes = int(kwargs["num_classes"])
    max_budget = int(kwargs["max_budget"])
    device = kwargs["device"]

    cell_source = kwargs.get("cell_source", "cellvit_embedding")
    uncertainty_mode = kwargs.get("uncertainty_mode", "disagreement")
    num_rounds = int(kwargs.get("num_rounds", 5))
    chunk_size = int(kwargs.get("chunk_size", 2000))
    n_sigma = int(kwargs.get("n_sigma", 2000))
    probe_epochs = int(kwargs.get("probe_epochs", 50))
    probe_lr = float(kwargs.get("probe_lr", 1e-3))
    probe_weight_decay = float(kwargs.get("probe_weight_decay", 1e-4))
    fusion_min_labels_per_class = float(
        kwargs.get("fusion_min_labels_per_class", 2.0)
    )
    fusion_normalize_logits = bool(kwargs.get("fusion_normalize_logits", False))
    diag = bool(kwargs.get("diag", True))

    if cell_source not in VALID_SOURCES:
        raise ValueError(f"Unknown cell_source={cell_source!r}")
    if uncertainty_mode not in VALID_UNCERTAINTIES:
        raise ValueError(f"Unknown uncertainty_mode={uncertainty_mode!r}")
    if cell_source == "crop_dino" and uncertainty_mode != "cell_margin":
        raise ValueError("crop_dino currently supports uncertainty_mode=cell_margin only")
    if len(dino_np) != len(nucleus_np) or len(dino_np) != len(reliability):
        raise ValueError("DINO, nucleus, and reliability arrays must align by patch")
    if dino_np.ndim != 2 or nucleus_np.ndim != 2:
        raise ValueError("image_embeddings and nucleus_embeddings must be 2-D")

    num_samples = len(dino_np)
    budget = min(max_budget, num_samples)
    rounds = max(1, min(num_rounds, budget))
    base, remainder = divmod(budget, rounds)
    round_sizes = [base + (1 if r < remainder else 0) for r in range(rounds)]

    reliability = np.clip(reliability, 0.0, 1.0)
    valid_nucleus = reliability > 0.0
    coverage_features = F.normalize(
        torch.as_tensor(dino_np, device=device, dtype=torch.float32),
        p=2,
        dim=1,
    )
    sigma = _adaptive_sigma(coverage_features, n_ref=n_sigma)
    covered = torch.zeros(num_samples, device=device, dtype=torch.float32)

    selected: List[int] = []
    selected_set = set()
    fusion_min_labels = max(
        int(np.ceil(fusion_min_labels_per_class * num_classes)), 20
    )

    for round_idx in tqdm(range(rounds), desc="NUCLEUS-AL Rounds"):
        n_select = round_sizes[round_idx]
        fallback_reason = None
        if round_idx == 0:
            uncertainty = None
            fallback_reason = "cold coverage"
        else:
            image_probe = _fit_probe(
                dino_np, labels, selected, num_classes, probe_epochs,
                probe_lr, probe_weight_decay, device,
            )
            if image_probe is None:
                uncertainty = None
                fallback_reason = "<2 observed classes"
            else:
                image_logits = image_probe.predict_logits(dino_np, device)
                image_uncertainty = margin_uncertainty_from_logits(image_logits)
                del image_probe

                cell_probe = _fit_probe(
                    nucleus_np, labels, selected, num_classes, probe_epochs,
                    probe_lr, probe_weight_decay, device, valid=valid_nucleus,
                )
                if cell_probe is None:
                    uncertainty = image_uncertainty
                    fallback_reason = "cell probe unavailable"
                else:
                    cell_logits = cell_probe.predict_logits(nucleus_np, device)
                    del cell_probe
                    rho = reliability

                    if uncertainty_mode == "cell_margin":
                        cell_uncertainty = margin_uncertainty_from_logits(cell_logits)
                        uncertainty = (
                            rho * cell_uncertainty
                            + (1.0 - rho) * image_uncertainty
                        )
                    elif uncertainty_mode == "disagreement":
                        disagreement = js_disagreement_from_logits(
                            image_logits, cell_logits
                        )
                        uncertainty = (
                            rho * disagreement
                            + (1.0 - rho) * image_uncertainty
                        )
                    else:
                        fusion_x = _fusion_features(
                            image_logits, cell_logits, uncertainty_mode,
                            normalize_logits=fusion_normalize_logits,
                        )
                        valid_selected = np.asarray(selected, dtype=np.int64)
                        valid_selected = valid_selected[
                            valid_nucleus[valid_selected]
                        ]
                        fusion_labels = labels[valid_selected]
                        if (
                            len(valid_selected) < fusion_min_labels
                            or not _can_train(fusion_labels)
                        ):
                            uncertainty = image_uncertainty
                            fallback_reason = (
                                "fusion warm-up "
                                f"{len(valid_selected)}/{fusion_min_labels} valid"
                            )
                        else:
                            fusion_probe = train_linear(
                                fusion_x[valid_selected], fusion_labels,
                                num_classes, probe_epochs, probe_lr, device,
                                weight_decay=probe_weight_decay,
                            )
                            fusion_logits = fusion_probe.predict_logits(fusion_x, device)
                            fusion_uncertainty = margin_uncertainty_from_logits(
                                fusion_logits
                            )
                            uncertainty = (
                                rho * fusion_uncertainty
                                + (1.0 - rho) * image_uncertainty
                            )
                            del fusion_probe, fusion_logits
                    del cell_logits
                del image_logits
                clear_memory()

        if uncertainty is None:
            weights = torch.ones(
                num_samples, device=device, dtype=torch.float32
            )
        else:
            normalized = _rank_normalize(uncertainty)
            weights = torch.as_tensor(
                normalized, device=device, dtype=torch.float32
            )
            if float(weights.max()) <= 0.0:
                weights.fill_(1.0)

        if diag:
            mean_unc = float(np.mean(uncertainty)) if uncertainty is not None else 1.0
            valid_rate = float(valid_nucleus.mean())
            suffix = f" | fallback={fallback_reason}" if fallback_reason else ""
            print(
                f"[NUC-DIAG b={budget} r={round_idx}] "
                f"source={cell_source} uncertainty={uncertainty_mode} "
                f"valid_cell={valid_rate:.3f} mean_unc={mean_unc:.3f}{suffix}"
            )

        picks = _greedy_coverage_batch(
            coverage_features, weights, covered, sigma, n_select,
            selected_set, chunk_size,
        )
        selected.extend(picks)
        del weights
        clear_memory()
        if len(picks) < n_select:
            break

    del coverage_features, covered
    clear_memory()
    return selected
