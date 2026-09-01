"""Final-training pass: LoRA + auxiliary loss + augmentation on raw pixels.

**`finetune_and_evaluate` runs AFTER a budget's points are already selected**,
once per budget, on `main.run`'s `selected_indices` -- it is what optionally
fine-tunes the backbone on just those points and re-evaluates.

**Every axis here is final-training only -- none of them changes what gets
selected.** `AUGMENT` used to be the exception: `make_augmented_feature_provider`
(below) was handed to `sampling/scalpel` so the per-round uncertainty probe
trained on augmented pixels. That was removed after measuring it on histoset
seed 42: it changed the SELECTED SET, with only ~55% of points overlapping
the un-augmented run at every budget. The cause is structural, not a bug --
the provider can only augment the probe's TRAINING rows (augmenting ~90k pool
rows per round costs ~27x), and the CellViT cell probe cannot be augmented at
all because its embeddings come from a cache with no pixels behind them, so
`disagreement = JS(visual, cell)` compared a noised probe against a clean one
and read the noise as disagreement (JS 1.65x the frozen run's on average, max
3.28x). `make_augmented_feature_provider` is still used, but only by the
final-training pass below.

**Two invariants the LoRA path depends on, both of which were violated in
the first version and produced near-chance accuracy on a real run:**

- *Score the space you trained in.* A LoRA run's probe reads features from
  the ADAPTED encoder, so it must be evaluated on test features from that
  same encoder (`encode_dataset`, hence the required `test_dataset`). The
  frozen embedding cache describes a different encoder; because both are
  768-d, using it raises nothing and simply reports noise.
- *Every budget starts from a clean adapter.* `main.run` loads this encoder
  once and reuses it across the whole sweep, and the training loop below
  mutates it in place, so `main.run` calls
  `training.lora.reset_lora_parameters` before each budget. Without that,
  budget k+1 inherits budget k's adapter, trained on a different labeled
  set, and the sweep stops measuring "N labels -> this accuracy".

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
from data.loaders import RawRGBDataset, default_num_workers, default_transform
from training.losses import center_loss, supcon_loss, triplet_loss
from training.probe import LinearProbe, _EarlyStopper, class_weights

__all__ = [
    "needs_pixels",
    "finetune_and_evaluate",
    "make_augmented_feature_provider",
    "encode_dataset",
]

AUX_LOSS_FNS = {"center": center_loss, "supcon": supcon_loss, "triplet": triplet_loss}

# Smallest trailing batch worth forming. The pairwise auxiliary losses need
# one same-class pair, and the chance a batch has none grows sharply as the
# batch shrinks relative to the class count: measured over 14 classes,
# all-singleton batches occur 62.7% of the time at size 4, 8.6% at size 8,
# and ~0% from size 12 up. Scaled by `num_classes` rather than fixed, because
# the same absolute size is safe for 2 classes and unsafe for 14.
def _min_tail_batch(num_classes: int) -> int:
    return max(8, min(2 * num_classes, 32))


def needs_pixels(use_lora: bool, augment: str, aux_loss: str = "none") -> bool:
    """True iff this configuration needs the pixel-reading final-training pass.

    The one predicate `run_al_main.ipynb` and this module both call, so
    "when do we load images" is decided in exactly one place -- duplicating
    this condition inline at two call sites is exactly the kind of drift
    that produces a config where one site thinks pixels are needed and the
    other does not.

    **`aux_loss` alone does NOT make this true, and that is deliberate.**
    An auxiliary loss acts on the encoder's features; with `use_lora=False`
    the encoder is frozen and those features carry no gradient, so the term
    is a constant that changes nothing (see `finetune_and_evaluate`'s guard,
    which refuses that combination outright rather than running a no-op).
    `aux_loss` is therefore only meaningful together with `use_lora`, which
    already makes this predicate true on its own.

    `aux_loss` is still in the signature so callers pass the whole config and
    this function -- not each call site -- decides. An earlier version
    omitted the parameter entirely, which meant `run_al_main.ipynb` and
    `main.py` could not even express the question.
    """
    if aux_loss != "none" and not use_lora:
        # Not silently False: this configuration is refused downstream, and
        # returning False here would route it to the frozen probe path where
        # it would run to completion as a mislabeled baseline.
        raise ValueError(
            f"aux_loss={aux_loss!r} with use_lora=False is not a valid "
            "configuration: the auxiliary losses act on encoder features, "
            "which carry no gradient when the encoder is frozen. Enable "
            "LoRA, or set aux_loss='none'."
        )
    return bool(use_lora) or augment != "none"


def _loader_workers(train_dataset) -> int:
    """DataLoader workers for a pixel-reading loop over `train_dataset`.

    Defers to `data.loaders.default_num_workers`, which already encodes this
    project's policy (2 for a per-file ImageFolder whose JPEG decode would
    otherwise block the GPU on the main thread, 0 for an already-decoded and
    usually memory-mapped .npz). The dataset object here does not carry its
    own path, so the kind is inferred from the attribute an NPZDataset has
    and an ImageFolder does not.
    """
    base = getattr(train_dataset, "dataset", train_dataset)
    is_npz = hasattr(base, "npz_path") or hasattr(base, "mmap")
    return default_num_workers("x.npz" if is_npz else "x/dir")


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


class _AllPixelDataset(Dataset):
    """Every row of `raw`, under a fixed (never augmented) transform.

    The test-set counterpart of `_SelectedPixelDataset`. Evaluation must see
    the encoder's plain preprocessing -- augmentation is a TRAINING-time
    perturbation, and applying it at test time would measure the encoder on
    inputs no protocol in this project scores on.
    """

    def __init__(self, raw: RawRGBDataset, transform) -> None:
        self.raw = raw
        self.transform = transform

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, i: int):
        image, label, _sample_id = self.raw[i]
        return self.transform(image), label


@torch.no_grad()
def encode_dataset(
    dataset,
    encoder_model: nn.Module,
    device: torch.device,
    image_encoder: str = "dinov2",
    conch_preprocess=None,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode EVERY row of `dataset` with `encoder_model`, in dataset order.

    Used to re-derive test features through a LoRA-adapted encoder. The
    transform is the encoder's plain preprocessing -- `augment="none"` is
    passed explicitly rather than defaulted, because augmenting a test set is
    never correct here.

    Returns `(len(dataset), feat_dim)` float32, aligned row-for-row with the
    dataset's own order, which is the order `test_labels` is in.
    """
    transform = _resolve_transform(image_encoder, "none", conch_preprocess)
    pixels = _AllPixelDataset(RawRGBDataset(dataset), transform)
    workers = _loader_workers(dataset)
    loader = DataLoader(
        pixels, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=workers > 0,
    )
    was_training = encoder_model.training
    encoder_model.eval()
    encoder_model = encoder_model.to(device)
    outputs = []
    for images, _labels in loader:
        images = images.to(device, non_blocking=True)
        outputs.append(_encode(encoder_model, images, image_encoder).float().cpu().numpy())
    if was_training:
        encoder_model.train()
    if not outputs:
        raise ValueError("encode_dataset got an empty dataset")
    return np.concatenate(outputs, axis=0).astype(np.float32)


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
    lora_lr: Optional[float] = None,
    aux_loss: str = "none",
    aux_weight: float = 0.5,
    augment: str = "none",
    encoder_model: Optional[nn.Module] = None,
    conch_preprocess=None,
    batch_size: int = 32,
    weight_decay: float = 0.0,
    test_dataset=None,
) -> Tuple[LinearProbe, Dict[str, float]]:
    """Run the final-training pass for one budget and evaluate it.

    Returns `(probe, metrics)` -- `metrics` has the same keys `main.py`'s
    `results[budget]` dict already uses (`acc`, `precision`, `recall`, `f1`),
    so the caller can drop it straight into that dict without renaming
    anything.

    **Which test features get scored depends on whether the encoder moved.**

    `use_lora=True` adapts the encoder, so the probe is trained on features
    from the ADAPTED encoder. Scoring that probe against the frozen
    embedding cache would compare two different feature spaces -- the probe
    would be reading coordinates that no longer mean what it learned. So
    when `use_lora=True`, `test_dataset` is REQUIRED and the test features
    are re-encoded through the adapted encoder here.

    Measured cost of getting this wrong (histoset, 14 balanced classes,
    r=8): accuracy fell to 0.10-0.21 against a 0.071 random floor, with
    predictions collapsing onto a few classes (3885 of 5600 test rows into
    one class at budget 200) -- while a LoRA perturbation far milder than
    real training already moves DINOv2 CLS features 27% in L2 (cosine 0.96).
    Nothing about the shapes disagrees, which is why this survived: both
    spaces are 768-d.

    `use_lora=False` (the augment-only path) leaves the encoder frozen, so
    its features ARE the cache's features and `test_features` is used
    directly -- `test_dataset` is not needed and re-encoding would only
    spend a forward pass to reproduce the cache.
    """
    if not needs_pixels(use_lora, augment, aux_loss):
        raise ValueError(
            "finetune_and_evaluate is for the final-training path only "
            "(use_lora=True, augment != 'none', or aux_loss != 'none'); the "
            "plain frozen-embedding path should call "
            "training.probe.train_probe directly instead"
        )
    if aux_loss not in ("none", *AUX_LOSS_FNS):
        raise ValueError(f"aux_loss must be one of {('none', *AUX_LOSS_FNS)}, got {aux_loss!r}")
    if aux_loss != "none" and not use_lora:
        # Every loss in training/losses.py reads only `features`, and on the
        # frozen path those are produced under torch.no_grad() -- so the aux
        # term has no grad_fn and adds a CONSTANT to the loss. It cannot move
        # the probe (the probe's own gradient comes from cross-entropy alone)
        # and it cannot move the encoder (frozen). Verified directly: calling
        # .backward() on center_loss over no_grad features raises "does not
        # require grad".
        #
        # Running it anyway would produce a result file stamped `auxcenter`
        # that is numerically identical to the no-aux baseline -- the same
        # class of silent no-op as the minmax-on-a-constant-vector lesson in
        # CLAUDE.md. Refuse instead.
        raise ValueError(
            f"aux_loss={aux_loss!r} requires use_lora=True. The auxiliary "
            "losses act on the ENCODER's features, but with use_lora=False "
            "the encoder is frozen and its features are computed under "
            "no_grad, so the term is a constant that changes nothing -- the "
            "run would be identical to aux_loss='none' while being labeled "
            "otherwise. Enable LoRA, or set aux_loss='none'."
        )
    if use_lora and encoder_model is None:
        raise ValueError("use_lora=True needs encoder_model (a LoRA-wrapped encoder)")
    if use_lora and test_dataset is None:
        # Hard error, never a silent fall back to `test_features`: that fall
        # back is exactly the bug this parameter exists to make impossible,
        # and it is invisible downstream because both spaces share a width.
        raise ValueError(
            "use_lora=True needs test_dataset: the probe is trained on "
            "features from the LoRA-ADAPTED encoder, so it must be scored on "
            "test features from that same encoder. Scoring it against the "
            "frozen embedding cache compares two different feature spaces "
            "and silently produces near-chance accuracy."
        )
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
    # ImageFolder datasets (histoset, skintissue) decode a JPEG per sample; at
    # num_workers=0 that happens on the main thread and the GPU waits on the
    # CPU rather than the reverse. `default_num_workers` already encodes this
    # project's policy -- 2 for a per-file dataset, 0 for an already-decoded
    # (and usually mapped) .npz, where a worker would only add pickling.
    loader_workers = _loader_workers(train_dataset)
    # A tiny TRAILING batch is the problem here, not the batch size itself.
    # `supcon`/`triplet` need at least one same-class pair somewhere in the
    # batch, and a 4-sample remainder over 14 classes is all-singletons ~61%
    # of the time (measured) -- which at 100 epochs is a near-certain crash
    # partway through a budget. `drop_last` is not the fix: at budget 25 it
    # would throw away labeled points this project spent its entire budget
    # acquiring. Instead, when the remainder would be too small to contain a
    # pair, widen the batch so the samples are redistributed and every batch
    # stays viable. `len(dataset)` is at most the largest budget (200), so a
    # widened batch is still small.
    batch_size = min(batch_size, len(dataset))
    min_tail = _min_tail_batch(num_classes)
    remainder = len(dataset) % batch_size
    if 0 < remainder < min_tail and len(dataset) > batch_size:
        # Grow the batch until the split is even, or until one batch holds
        # everything (the full-batch case, which has no remainder at all).
        while batch_size < len(dataset):
            batch_size += 1
            remainder = len(dataset) % batch_size
            if remainder == 0 or remainder >= min_tail:
                break
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=loader_workers, pin_memory=loader_workers > 0,
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
    # The probe and the adapter get SEPARATE learning rates.
    #
    # They were on one `lr=probe_lr` (1e-3), which is right for a linear probe
    # on frozen features and far too large for a rank-8 adapter inside a
    # 12-layer ViT: measured on real DINOv2, lr=1e-3 moves the CLS feature
    # 116.9% in L2 (cosine 0.318 with the frozen output) against 82.9%
    # (cosine 0.652) at 1e-4. On a real histoset run that showed up as the
    # encoder's features losing class separation entirely -- 2849 of 5600
    # test rows predicted as one class at budget 200, with the probe's own
    # weight norm SMALLER than the frozen baseline's (2.44 vs 3.93) and its
    # per-class rows more uniform (max/min 1.13 vs 1.29). That is a collapsed
    # feature space, not an over-fit probe, and it cost the most at low
    # budgets (-2.8 sigma at 50, -3.8 sigma at 75).
    #
    # `lora_lr=None` means "same as probe_lr", which reproduces the old
    # single-rate behaviour exactly for anyone comparing against it.
    if use_lora and lora_lr is not None and lora_lr != probe_lr:
        from training.lora import lora_parameters as _lora_parameters

        adapter_params = _lora_parameters(encoder_model)
        adapter_ids = {id(p) for p in adapter_params}
        probe_params = [p for p in trainable if id(p) not in adapter_ids]
        optimizer = torch.optim.Adam(
            [
                {"params": probe_params, "lr": probe_lr},
                {"params": adapter_params, "lr": lora_lr},
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(trainable, lr=probe_lr, weight_decay=weight_decay)
    aux_fn = AUX_LOSS_FNS.get(aux_loss)

    # Same budget and same stopping rule as every other training loop in this
    # project (training/probe.py::_EarlyStopper): up to `probe_epochs`, cut
    # short once the epoch's mean training loss has plateaued for 20 epochs.
    # It matters most here -- this is the only loop that pays a full encoder
    # forward (and, under LoRA, a backward) per image per epoch.
    stopper = _EarlyStopper()
    aux_skipped = 0
    aux_batches = 0
    for epoch in range(probe_epochs):
        running, batches = 0.0, 0
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
                try:
                    loss = loss + aux_weight * aux_fn(features, logits, batch_labels)
                except ValueError:
                    # The pairwise losses raise when NO anchor in the batch has
                    # a same-class positive. Batch widening above makes that
                    # rare, but it cannot be ruled out (a labeled set can be
                    # spread across more classes than any batch can pair up),
                    # and aborting a budget hours into a sweep over one
                    # unlucky shuffle is the wrong trade. Skip the aux term
                    # for THIS batch only; cross-entropy still trains on it.
                    #
                    # Counted, not swallowed: a run where this fires on most
                    # batches is not really an auxiliary-loss run, and the
                    # printed summary below is what says so.
                    aux_skipped += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            batches += 1
            if aux_fn is not None:
                aux_batches += 1
        if batches and stopper.step(running / batches, epoch):
            break

    if aux_skipped:
        # Loud, because a high ratio means the run is closer to aux_loss
        # "none" than to what its filename claims.
        share = aux_skipped / max(aux_batches, 1)
        print(
            f"[final-train] aux_loss={aux_loss!r} skipped on {aux_skipped}/"
            f"{aux_batches} batches ({share:.1%}) -- no same-class pair in "
            "those batches"
            + ("  *** the auxiliary loss barely acted on this run ***"
               if share > 0.5 else "")
        )

    probe.eval()
    if use_lora:
        # The encoder moved during training, so the cache no longer describes
        # it. Re-encode the test set through the adapted encoder: this is the
        # only feature space the probe above was ever trained to read.
        eval_features = encode_dataset(
            test_dataset, encoder_model, device,
            image_encoder=image_encoder, conch_preprocess=conch_preprocess,
            batch_size=max(batch_size, 32),
        )
        if eval_features.shape[0] != len(test_labels):
            raise ValueError(
                f"re-encoded test features have {eval_features.shape[0]} rows "
                f"but test_labels has {len(test_labels)} -- test_dataset is "
                "not the dataset test_labels came from"
            )
    else:
        eval_features = test_features
    predictions = np.argmax(probe.predict_proba(eval_features, device), axis=1)
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


def make_augmented_feature_provider(
    train_dataset,
    encoder_model: nn.Module,
    device: torch.device,
    image_encoder: str = "dinov2",
    augment: str = "flip_rotate",
    conch_preprocess=None,
    batch_size: int = 64,
):
    """Return `provider(indices) -> np.ndarray` of freshly-augmented features.

    **No longer used for SELECTION.** It once fed `sampling/scalpel` an opaque
    callable so the per-round uncertainty probe trained on augmented pixels;
    that path is gone (see the module docstring for the measurement that
    removed it). `sampling/scalpel/uncertainty.py` still accepts an
    `augmented_feature_provider` and its tests still cover that branch, but
    `main.run` never passes one -- augmentation is a final-training axis.

    Why this is cheap enough to run inside the AL loop: the probe trains on
    the LABELED set only (<= the largest budget, 200 here), never on the
    ~90k-row pool. A whole 8-budget x 5-round sweep is ~2.7k image forwards
    per epoch, against 3.6M for re-extracting the pool every round -- ~27x
    cheaper, which is what makes "augment during selection" viable at all.

    **The asymmetry that made this unusable for selection**, kept here because
    it explains why the selection path was removed rather than repaired. The
    probe TRAINS on augmented pixels but scores the pool from the FROZEN
    cache, since augmenting 90k rows per round is the 27x cost this avoids.
    For a final probe that is just the ordinary
    train-with-augmentation/infer-without arrangement. Inside the AL loop it
    was not, because two things could not follow:

    * `sampling/scalpel`'s CELL probe cannot be augmented at all -- CellViT
      embeddings come from a cache with no pixels behind them, so
      `uncertainty_mode="disagreement"` compared an augmented visual probe
      against an un-augmented cell probe and read the difference as
      disagreement.
    * The temperature calibration preceding that comparison was fitted on the
      augmented features, so it calibrated the augmented probe, not a frozen
      one.

    Measured consequence on histoset seed 42: JS 1.65x the frozen run's on
    average (max 3.28x), ~55% selection overlap, and worse class balance at
    the high budgets.
    """
    if augment == "none":
        raise ValueError(
            "make_augmented_feature_provider is for augment != 'none'; the "
            "un-augmented path must read the frozen cache directly instead of "
            "re-encoding identical pixels every round"
        )
    transform = _resolve_transform(image_encoder, augment, conch_preprocess)
    raw = RawRGBDataset(train_dataset)
    encoder_model = encoder_model.to(device)
    encoder_model.eval()
    # Resolved once, outside `provider`: the answer cannot change between
    # rounds, and `default_num_workers` is a policy lookup, not a measurement.
    loader_workers = _loader_workers(train_dataset)

    def provider(indices) -> np.ndarray:
        indices = list(indices)
        if not indices:
            return np.empty((0, _encoder_output_dim(encoder_model, image_encoder)), dtype=np.float32)
        dataset = _SelectedPixelDataset(raw, indices, transform)
        loader = DataLoader(
            dataset, batch_size=min(batch_size, len(dataset)), shuffle=False,
            num_workers=loader_workers, pin_memory=loader_workers > 0,
        )
        chunks = []
        with torch.no_grad():
            for images, _labels in loader:
                images = images.to(device, non_blocking=True)
                chunks.append(_encode(encoder_model, images, image_encoder).cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32)

    return provider
