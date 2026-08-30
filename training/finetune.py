"""Final-training pass: LoRA + auxiliary loss + augmentation on raw pixels.

**This runs AFTER a budget's points are already selected -- never inside the
active-learning loop.** Selection always uses the frozen embedding cache
(fast, and the whole point of comparing samplers under one frozen protocol);
this module is what `run_al_main.ipynb` calls once per budget, after
`main.run`'s sweep has produced `selected_indices`, to optionally fine-tune
the backbone on just those points and re-evaluate.

**Every budget gets its own full-curve point**, not just the largest one --
the confirmed choice for `TRAIN_MODE` axes at every budget in the sweep
(`PLAN_IMPLEMENT.md` §6.4's open question, resolved: full curve, ~8x the cost
of frozen-only, because the research question is whether LoRA helps more at
low or high budgets, which a single max-budget point cannot answer).

**The fast path is unchanged.** `use_lora=False` and `augment="none"` train a
plain linear probe on the embedding cache -- `training/probe.py::train_probe`,
no pixel ever loaded -- exactly the frozen-backbone control this project has
always run. Only `use_lora=True` or `augment != "none"` need pixels, because
only they can act on something the frozen embedding cache does not carry.

Results are written into the SAME `results["linear"]` shape `main.py`
already produces -- no second key. Two configurations of this pass
(`USE_LORA=False` vs `True`, say) are two notebook runs with two distinct
zip names (§2.0), so nothing here needs to disambiguate them a second time
inside one file.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.augment import build_augment_transform
from data.loaders import RawRGBDataset, default_transform
from training.losses import center_loss, supcon_loss, triplet_loss
from training.probe import LinearProbe, class_weights

__all__ = ["needs_pixels", "finetune_and_evaluate"]

AUX_LOSS_FNS = {"center": center_loss, "supcon": supcon_loss, "triplet": triplet_loss}


def needs_pixels(use_lora: bool, augment: str) -> bool:
    """True iff this configuration needs raw pixels at all.

    The one predicate `run_al_main.ipynb` and this module both call, so
    "when do we load images" is decided in exactly one place -- duplicating
    this condition inline at two call sites is exactly the kind of drift
    that produces a config where one site thinks pixels are needed and the
    other does not.
    """
    return bool(use_lora) or augment != "none"


class _SelectedPixelDataset(Dataset):
    """`RawRGBDataset` restricted to `selected_indices`, with a transform
    (encoder-specific resize/normalize, optionally composed with augmentation)
    applied at read time.

    Not a `torch.utils.data.Subset` wrapping `RawRGBDataset`: `Subset` would
    apply no transform on its own, and `RawRGBDataset.__getitem__` returns a
    raw PIL image specifically so a caller can choose what transform to
    apply -- this class is that choice, made once, for the final-training
    pass.
    """

    def __init__(self, raw: RawRGBDataset, indices, transform) -> None:
        self.raw = raw
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        image, label, _sample_id = self.raw[self.indices[i]]
        return self.transform(image), label


def _resolve_transform(image_encoder: str, augment: str, conch_preprocess=None):
    if image_encoder == "conch":
        if conch_preprocess is None:
            raise ValueError(
                "image_encoder='conch' needs conch_preprocess (the transform "
                "load_conch's factory returned) -- it carries CONCH's own "
                "448x448 resize and OpenAI CLIP normalization, which must not "
                "be hand-rolled (features/vlm.py module docstring)."
            )
        base = conch_preprocess
    else:
        base = default_transform()

    augment_transform = build_augment_transform(augment)
    if augment == "none":
        return base

    import torchvision.transforms as transforms

    # Augment BEFORE the resize/normalize: flip/rotate operate on the raw
    # PIL image, and running them after ToTensor()+Normalize would rotate
    # already-normalized pixel statistics, which is harmless for a 90-degree
    # rotation but wrong in general and inconsistent with every other
    # augmentation pipeline in this codebase (none exist yet, so there is
    # nothing to match -- this is the precedent).
    return transforms.Compose([augment_transform, base])


def finetune_and_evaluate(
    train_dataset,
    selected_indices,
    labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int,
    device: torch.device,
    probe_epochs: int,
    probe_lr: float,
    image_encoder: str = "dinov2",
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    aux_loss: str = "none",
    aux_weight: float = 0.5,
    augment: str = "none",
    encoder_model: Optional[nn.Module] = None,
    conch_preprocess=None,
    batch_size: int = 32,
    weight_decay: float = 0.0,
) -> Tuple[LinearProbe, Dict[str, float]]:
    """Run the final-training pass for one budget and evaluate it.

    Returns `(probe, metrics)` -- `metrics` has the same keys `main.py`'s
    `results[budget]` dict already uses (`acc`, `precision`, `recall`, `f1`),
    so the caller can drop it straight into that dict without renaming
    anything.

    `test_features` stays the FROZEN embedding-cache features, never
    re-extracted through a fine-tuned encoder: this pass adapts the encoder
    only on the training pixels, and the whole point of a shared test set is
    that scoring against it means the same thing across every run. A LoRA
    run's test-time features would need the ADAPTED encoder to be
    meaningful, which is a larger change (a per-run test cache) explicitly
    out of scope here -- `evaluate_al_sampler.ipynb`'s existing encoder-aware
    scoring does not (yet) know how to load an adapted-encoder cache, and
    building that is future work, not silently approximated by scoring a
    fine-tuned model against frozen features.
    """
    if not needs_pixels(use_lora, augment):
        raise ValueError(
            "finetune_and_evaluate is for the pixel-reading path only "
            "(use_lora=True or augment != 'none'); the frozen-embedding "
            "path should call training.probe.train_probe directly instead"
        )
    if aux_loss not in ("none", *AUX_LOSS_FNS):
        raise ValueError(f"aux_loss must be one of {('none', *AUX_LOSS_FNS)}, got {aux_loss!r}")
    if use_lora and encoder_model is None:
        raise ValueError("use_lora=True needs encoder_model (a LoRA-wrapped encoder)")
    if encoder_model is None:
        # The augment-only path still has to turn augmented PIXELS into
        # features, which needs an encoder. An earlier version fell through to
        # `features = images` here, feeding a raw (B, 3, H, W) tensor straight
        # into the probe -- a shape error at best, and at worst (if the widths
        # ever lined up) a silently meaningless run. main.py always supplies
        # an encoder when needs_pixels() is True, so this only catches a
        # direct caller.
        raise ValueError(
            "encoder_model is required whenever needs_pixels() is True: the "
            f"augmented pixels have to be encoded (augment={augment!r}). Pass "
            "the same frozen encoder main.py loads once per run."
        )

    selected_indices = list(selected_indices)
    labeled_labels = np.asarray(labels)[selected_indices]

    transform = _resolve_transform(image_encoder, augment, conch_preprocess)
    dataset = _SelectedPixelDataset(RawRGBDataset(train_dataset), selected_indices, transform)
    loader = DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=True,
    )

    if use_lora:
        # LoRA path: encoder produces features end-to-end, with gradient.
        feat_dim = _encoder_output_dim(encoder_model, image_encoder)
        probe = LinearProbe(feat_dim, num_classes).to(device)
        encoder_model = encoder_model.to(device)
        encoder_model.train()
        from training.lora import lora_parameters

        trainable = lora_parameters(encoder_model) + list(probe.parameters())
    else:
        # Augment-only path: encoder is frozen and used only under no_grad to
        # produce features; only the probe (and, if used, the aux loss acting
        # on those features) gets gradients.
        feat_dim = _encoder_output_dim(encoder_model, image_encoder)
        probe = LinearProbe(feat_dim, num_classes).to(device)
        encoder_model = encoder_model.to(device)
        encoder_model.eval()
        trainable = list(probe.parameters())

    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights(labeled_labels, num_classes), device=device)
    )
    optimizer = torch.optim.Adam(trainable, lr=probe_lr, weight_decay=weight_decay)
    aux_fn = AUX_LOSS_FNS.get(aux_loss)

    for _ in range(probe_epochs):
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True).long()

            if use_lora:
                features = _encode(encoder_model, images, image_encoder)
            else:
                with torch.no_grad():
                    features = _encode(encoder_model, images, image_encoder)

            logits = probe(features)
            loss = criterion(logits, batch_labels)
            if aux_fn is not None:
                loss = loss + aux_weight * aux_fn(features, logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    predictions = np.argmax(probe.predict_proba(test_features, device), axis=1)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = {
        "acc": float(accuracy_score(test_labels, predictions)),
        "precision": float(precision_score(test_labels, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(test_labels, predictions, average="macro", zero_division=0)),
        "f1": float(f1_score(test_labels, predictions, average="macro", zero_division=0)),
    }
    return probe, metrics


def _encoder_output_dim(encoder_model: Optional[nn.Module], image_encoder: str) -> int:
    if encoder_model is None:
        raise ValueError("encoder_model is required to determine the feature width")
    if image_encoder == "dinov2":
        return encoder_model.config.hidden_size

    # CONCH RAW_SPACE width = whatever `VisualModel.forward_no_head` returns,
    # which is NOT the projected `embed_dim` and NOT the trunk width in
    # general -- it depends on which branch that method takes
    # (conch/open_clip_custom/vision_tower.py):
    #
    #   use_attentional_pool_contrast=True  (CONCH ViT-B-16's own config):
    #       pooled = ln_contrast(attn_pool_contrast(trunk(x))[:, 0])
    #       -> width = embed_dim_contrast = 512, readable off ln_contrast.
    #       The `@ proj_contrast` matmul that `forward` applies afterwards is
    #       skipped here; it happens to be square (512x512), which is why
    #       RAW_SPACE and PROJ_SPACE are both 512 and why a wrong width here
    #       would NOT have been caught by a shape mismatch downstream.
    #   otherwise:
    #       pooled, _ = self._global_pool(trunk(x))  -> width = trunk width.
    #
    # `VisualModel` defines neither `output_dim` nor `embed_dim`, so reading
    # those (an earlier version of this function did) always raised.
    visual = getattr(encoder_model, "visual", encoder_model)
    if getattr(visual, "use_attentional_pool_contrast", False):
        ln_contrast = getattr(visual, "ln_contrast", None)
        normalized_shape = getattr(ln_contrast, "normalized_shape", None)
        if not normalized_shape:
            raise AttributeError(
                "CONCH vision tower reports use_attentional_pool_contrast=True "
                "but has no ln_contrast.normalized_shape to read the RAW_SPACE "
                "width from -- conch/open_clip_custom/vision_tower.py layout "
                "may have changed."
            )
        return int(normalized_shape[0])

    trunk = getattr(visual, "trunk", None)
    trunk_width = getattr(trunk, "num_features", None) if trunk is not None else None
    if trunk_width is None:
        raise AttributeError(
            "Could not determine CONCH's RAW_SPACE feature width: no "
            "attentional contrast pooler and no visual.trunk.num_features."
        )
    return int(trunk_width)


def _encode(encoder_model: nn.Module, images: torch.Tensor, image_encoder: str) -> torch.Tensor:
    if image_encoder == "dinov2":
        return encoder_model(pixel_values=images).last_hidden_state[:, 0, :]
    # CONCH: RAW_SPACE (proj_contrast=False, normalize=False) -- the probe
    # space, never PROJ_SPACE (features/vlm.py module docstring).
    return encoder_model.encode_image(images, normalize=False, proj_contrast=False)
