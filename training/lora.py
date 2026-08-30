"""LoRA (Low-Rank Adaptation) for the two ViT architectures this project uses.

`peft` is NOT installed in this environment (checked alongside `timm` and
`open_clip` -- none of the three are present, torch 2.2.2 only). This module
implements LoRA by hand instead of depending on it.

**Three attention module shapes, three wrap strategies.** The two encoders
`run_al_main.ipynb` supports do not share an attention module shape, and
CONCH does not even use one shape internally:

* **DINOv2** (`transformers.Dinov2Model`) exposes `query`/`key`/`value` as
  three separate `nn.Linear` layers (`Dinov2SelfAttention`) -- `LinearLoRA`
  below wraps any one of them directly, by attribute path.
* **CONCH's VISION tower** -- the one this project fine-tunes -- is
  **`timm`'s** `VisionTransformer`, NOT any class in `conch/`. CONCH builds
  it in `coca_model.py::_build_vision_tower`
  (`from timm.models.vision_transformer import VisionTransformer`, then
  `VisualModel(trunk=trunk, ...)`), so the module path is
  `model.visual.trunk.blocks[i].attn`, a `timm.layers.attention.Attention`
  whose Q/K/V is ONE fused `qkv` `nn.Linear` of shape `(3*dim, dim)`.
  `FusedQKVLoRA` wraps that single `nn.Linear` and adds the low-rank delta
  only into its Q and V output row-blocks, leaving timm's own attention
  `forward` completely untouched -- deliberately, because that forward
  varies across timm versions (`fused_attn`, `q_norm`/`k_norm`, a post-attn
  `norm`), and re-implementing it here would silently drift from whatever
  timm the Kaggle image actually installs.
* **CONCH's TEXT/multimodal towers**
  (`conch.open_clip_custom.transformer.ResidualAttentionBlock`) use
  `nn.MultiheadAttention`, whose Q/K/V is a fused `in_proj_weight`
  *parameter* with no `nn.Linear` to wrap at all.
  `MultiheadAttentionLoRA` handles that shape by re-deriving Q/K/V from the
  fused weight and recomputing attention via `scaled_dot_product_attention`.
  The final-training pass does not currently adapt the text tower (it is
  frozen), so this class is unused by `apply_lora_to_conch` -- it is kept
  because it is verified correct and is what a text-tower adaptation would
  need, not because the vision path uses it.

**Which module is which was a real bug.** An earlier version of this file
assumed CONCH's vision tower was `ResidualAttentionBlock`/`nn.MultiheadAttention`
and made `apply_lora_to_conch` walk `model.visual.transformer.resblocks` --
a path that does not exist (`VisualModel` has `.trunk`, and `resblocks`
belongs to the TEXT-side `Transformer` class). It was never caught because
the only test exercising it built a fake module matching that same wrong
assumption. Verify a wrap target against the real class, not against a fake
built from the same belief that produced the wrap.

**Only Q and V get a LoRA delta, never K** -- the original LoRA paper's
finding (adapting K contributes little) and this project's own budget: two
low-rank deltas per attention block is already the ablation this step exists
to run, not four.

`r=0` must be bit-for-bit identical to the frozen module it wraps -- this is
the control every LoRA run is compared against, and `test_lora_zero_equals_frozen`
pins it. Both classes special-case `r=0` to skip the delta computation
entirely rather than computing `alpha/r * (x @ A.T @ B.T)` with an empty `A`,
`B` (which would be zero anyway, but every LoRA delta parameter is still
constructed here, since a run that starts at `r=0` and is later swept to
`r>0` should not have needed different code to build the model).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LinearLoRA",
    "FusedQKVLoRA",
    "MultiheadAttentionLoRA",
    "apply_lora_to_dinov2",
    "apply_lora_to_conch",
    "lora_parameters",
]


class LinearLoRA(nn.Module):
    """Wraps one `nn.Linear`, adding a frozen base + a trainable low-rank delta.

    `y = base(x) + (alpha/r) * x @ A.T @ B.T`, `A` initialized with the
    standard LoRA Kaiming scheme, `B` initialized to ZERO so the delta starts
    at exactly zero -- a freshly-wrapped module computes the same output as
    the module it wraps, before any training step happens (not just at
    `r=0`, at any `r`, on step 0).

    `base` is frozen (`requires_grad_(False)`) here rather than left to the
    caller: a LoRA run must never silently also fine-tune the full backbone
    weight, which would defeat the entire point of using LoRA at this budget
    scale (a few hundred labeled points).
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        if r < 0:
            raise ValueError(f"r must be >= 0, got {r}")
        self.base = base
        self.base.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        in_features = base.in_features
        out_features = base.out_features
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            # Registered as empty buffers, not omitted, so `r` can be swept
            # across runs of the same notebook without the module's parameter
            # set changing shape depending on the value.
            self.register_buffer("lora_A", torch.empty(0, in_features))
            self.register_buffer("lora_B", torch.empty(out_features, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r == 0:
            return out
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out + (self.alpha / self.r) * delta


class FusedQKVLoRA(nn.Module):
    """Wraps a FUSED `qkv` `nn.Linear` -- `timm`'s attention shape, which is
    what CONCH's vision tower is built from.

    `timm.layers.attention.Attention` computes Q, K and V in ONE projection
    of shape `(3*dim, dim)` and splits the result afterwards, so the three
    projections are three ROW-BLOCKS of a single weight rather than three
    modules. This class keeps that single `nn.Linear` frozen and adds a
    separate low-rank delta into the Q block (rows `[0:dim]`) and the V block
    (rows `[2*dim:3*dim]`), leaving the K block (rows `[dim:2*dim]`) exactly
    as the frozen base produced it -- the same Q-and-V-only choice
    `MultiheadAttentionLoRA` and `apply_lora_to_dinov2` make.

    Wrapping the `nn.Linear` rather than replacing the whole `Attention`
    module is deliberate: timm's attention `forward` differs across versions
    (`fused_attn` toggling `scaled_dot_product_attention` vs an explicit
    softmax, optional `q_norm`/`k_norm`, a post-attention `norm` in newer
    releases). Because this wrap is a drop-in for the projection alone,
    timm's own forward runs unchanged whatever version Kaggle installs, and
    a timm upgrade cannot silently change the math under this adapter.

    Like `LinearLoRA`, `B` is initialized to ZERO, so a freshly-wrapped
    module is output-identical to the frozen one at any `r` before the first
    optimizer step, and `r=0` skips the delta entirely.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        if r < 0:
            raise ValueError(f"r must be >= 0, got {r}")
        out_features = base.out_features
        if out_features % 3 != 0:
            raise ValueError(
                f"FusedQKVLoRA expects a fused qkv projection whose out_features "
                f"is divisible by 3 (Q|K|V stacked); got {out_features}. This is "
                "not a timm-style fused attention projection."
            )
        self.base = base
        self.base.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.dim = out_features // 3
        in_features = base.in_features
        if r > 0:
            self.lora_A_q = nn.Parameter(torch.empty(r, in_features))
            self.lora_B_q = nn.Parameter(torch.zeros(self.dim, r))
            self.lora_A_v = nn.Parameter(torch.empty(r, in_features))
            self.lora_B_v = nn.Parameter(torch.zeros(self.dim, r))
            nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        else:
            # Same reasoning as LinearLoRA: registered, not omitted, so the
            # module's parameter set does not change shape with `r`.
            self.register_buffer("lora_A_q", torch.empty(0, in_features))
            self.register_buffer("lora_B_q", torch.empty(self.dim, 0))
            self.register_buffer("lora_A_v", torch.empty(0, in_features))
            self.register_buffer("lora_B_v", torch.empty(self.dim, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r == 0:
            return out
        scaling = self.alpha / self.r
        delta_q = F.linear(F.linear(x, self.lora_A_q), self.lora_B_q) * scaling
        delta_v = F.linear(F.linear(x, self.lora_A_v), self.lora_B_v) * scaling
        # Build the full (…, 3*dim) delta and add it out-of-place. An in-place
        # `out[..., :dim] += delta_q` on the base's output would be an autograd
        # hazard (that tensor is needed to backprop through `base`), and the
        # zero K block keeps the K projection bit-identical to the frozen base.
        zeros_k = torch.zeros_like(delta_q)
        return out + torch.cat([delta_q, zeros_k, delta_v], dim=-1)


class MultiheadAttentionLoRA(nn.Module):
    """Replaces an `nn.MultiheadAttention` with a LoRA-adapted equivalent.

    Reads Q/K/V weights out of the original module's fused `in_proj_weight`
    (shape `(3*dim, dim)`, the `torch.chunk(..., 3)` split PyTorch's own
    `nn.MultiheadAttention.forward` uses internally) rather than
    re-initializing them, so a freshly-wrapped module starts with the
    ORIGINAL pretrained weights, not random ones. `out_proj` is a real
    `nn.Linear` subclass already (`NonDynamicallyQuantizableLinear`), so it
    is wrapped with `LinearLoRA` directly rather than re-implemented here.

    Only supports the call shape CONCH's `ResidualAttentionBlock.attention`
    actually uses: `forward(q_x, k_x, v_x, need_weights=False, attn_mask=...)`,
    self-attention only (`q_x is k_x is v_x` in every call CONCH makes) --
    this is not a general `nn.MultiheadAttention` drop-in, and raises if
    asked for anything else (cross-attention, `need_weights=True`).
    """

    def __init__(self, original: nn.MultiheadAttention, r: int, alpha: float) -> None:
        super().__init__()
        if original._qkv_same_embed_dim is False:
            raise ValueError(
                "MultiheadAttentionLoRA only supports self-attention with a "
                "fused in_proj_weight (qkv_same_embed_dim=True)"
            )
        if r < 0:
            raise ValueError(f"r must be >= 0, got {r}")

        self.embed_dim = original.embed_dim
        self.num_heads = original.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout_p = float(original.dropout)

        # Q/K/V split out of the ORIGINAL fused weight -- these are plain
        # (non-nn.Linear) parameters, matching how nn.MultiheadAttention
        # itself stores them, and are frozen: LoRA adapts through the deltas
        # below, never through these.
        q_w, k_w, v_w = original.in_proj_weight.detach().chunk(3, dim=0)
        self.q_weight = nn.Parameter(q_w.clone(), requires_grad=False)
        self.k_weight = nn.Parameter(k_w.clone(), requires_grad=False)
        self.v_weight = nn.Parameter(v_w.clone(), requires_grad=False)
        if original.in_proj_bias is not None:
            q_b, k_b, v_b = original.in_proj_bias.detach().chunk(3, dim=0)
            self.q_bias = nn.Parameter(q_b.clone(), requires_grad=False)
            self.k_bias = nn.Parameter(k_b.clone(), requires_grad=False)
            self.v_bias = nn.Parameter(v_b.clone(), requires_grad=False)
        else:
            self.q_bias = self.k_bias = self.v_bias = None

        self.out_proj = LinearLoRA(original.out_proj, r=r, alpha=alpha)

        self.r = r
        self.alpha = alpha
        if r > 0:
            # Q and V only -- never K (see module docstring).
            self.lora_A_q = nn.Parameter(torch.empty(r, self.embed_dim))
            self.lora_B_q = nn.Parameter(torch.zeros(self.embed_dim, r))
            self.lora_A_v = nn.Parameter(torch.empty(r, self.embed_dim))
            self.lora_B_v = nn.Parameter(torch.zeros(self.embed_dim, r))
            nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        else:
            self.register_buffer("lora_A_q", torch.empty(0, self.embed_dim))
            self.register_buffer("lora_B_q", torch.empty(self.embed_dim, 0))
            self.register_buffer("lora_A_v", torch.empty(0, self.embed_dim))
            self.register_buffer("lora_B_v", torch.empty(self.embed_dim, 0))

    def _lora_delta(self, x: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor) -> torch.Tensor:
        if self.r == 0:
            return torch.zeros_like(x[..., :0]).new_zeros(x.shape[:-1] + (lora_B.shape[0],))
        return (self.alpha / self.r) * F.linear(F.linear(x, lora_A), lora_B)

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: torch.Tensor,
        v_x: torch.Tensor,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if need_weights:
            raise NotImplementedError(
                "MultiheadAttentionLoRA does not return attention weights "
                "(need_weights=True) -- CONCH's own attention() call never "
                "requests them (need_weights=False), and nothing here reads them."
            )
        if not (q_x is k_x is v_x):
            raise NotImplementedError(
                "MultiheadAttentionLoRA only supports self-attention "
                "(q_x is k_x is v_x) -- CONCH's vision tower never calls "
                "cross-attention through this module."
            )

        q = F.linear(q_x, self.q_weight, self.q_bias)
        k = F.linear(k_x, self.k_weight, self.k_bias)
        v = F.linear(v_x, self.v_weight, self.v_bias)
        if self.r > 0:
            q = q + self._lora_delta(q_x, self.lora_A_q, self.lora_B_q)
            v = v + self._lora_delta(v_x, self.lora_A_v, self.lora_B_v)

        # (seq, batch, embed) -> (batch, heads, seq, head_dim), the layout
        # scaled_dot_product_attention expects.
        seq_len, batch, _ = q.shape
        k_len = k.shape[0]

        def split_heads(t, length):
            return t.reshape(length, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        q = split_heads(q, seq_len)
        k = split_heads(k, k_len)
        v = split_heads(v, k_len)

        attn_mask_arg = attn_mask
        if attn_mask_arg is not None and attn_mask_arg.dim() == 2:
            attn_mask_arg = attn_mask_arg.unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask_arg,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.permute(2, 0, 1, 3).reshape(seq_len, batch, self.embed_dim)
        out = self.out_proj(out)
        return out, None


def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


def _set_submodule(root: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def apply_lora_to_dinov2(model: nn.Module, r: int, alpha: float) -> nn.Module:
    """Wrap every attention block's `query` and `value` projections
    (`Dinov2SelfAttention.query`/`.value`) with `LinearLoRA`, in place.

    Every parameter in `model` is frozen FIRST, unconditionally -- a
    checkpoint loaded via `from_pretrained` has every parameter trainable by
    default, and `LinearLoRA.__init__` only ever freezes the ONE `nn.Linear`
    it is given. Without freezing the whole model up front, everything this
    function does not explicitly wrap (patch embeddings, layer norms, MLPs,
    `key` projections, `attention.output.dense`) stays trainable and a
    "LoRA-only" run would silently fine-tune the entire backbone --
    `key` is intentionally left un-adapted (LoRA delta, not gradient): it
    still must not receive a gradient either.

    Returns `model` (mutated in place) for chaining convenience; the return
    value is the same object passed in.
    """
    model.requires_grad_(False)
    encoder = model.encoder if hasattr(model, "encoder") else model
    layers = encoder.layer
    for layer in layers:
        attention = layer.attention.attention  # Dinov2SelfAttention
        attention.query = LinearLoRA(attention.query, r=r, alpha=alpha)
        attention.value = LinearLoRA(attention.value, r=r, alpha=alpha)
    return model


def apply_lora_to_conch(model: nn.Module, r: int, alpha: float) -> nn.Module:
    """Wrap every vision-tower attention block's `nn.MultiheadAttention`
    with `MultiheadAttentionLoRA`, in place.

    Freezes the ENTIRE model first -- `model` here is the full CoCa
    checkpoint (image tower + text tower + the captioning head, if present),
    but `run_al_main.ipynb`'s final-training pass only ever needs gradients
    into the vision tower's LoRA deltas: the text prototypes it might compare
    against are computed once, frozen, by `extract_vlm_features.ipynb`. Not
    freezing the text tower here would let a "LoRA-only" run silently
    fine-tune it too.

    Walks `model.visual.trunk.blocks[i].attn.qkv` -- CONCH's vision tower is
    `timm`'s `VisionTransformer` (built in `coca_model.py::_build_vision_tower`
    and stored as `VisualModel.trunk`), whose attention keeps Q/K/V in one
    fused `qkv` `nn.Linear`. Each is replaced with `FusedQKVLoRA`, which adds
    a delta to the Q and V row-blocks only.

    NOT `model.visual.transformer.resblocks`: `VisualModel` has no
    `.transformer` attribute at all, and `resblocks` belongs to the TEXT-side
    `Transformer` class in `conch/open_clip_custom/transformer.py`. An earlier
    version of this function used that path and would have raised
    `AttributeError` on the first real CONCH run.

    Raises `AttributeError` naming the path it could not find rather than
    silently wrapping zero blocks -- a layout change in CONCH or timm must
    not look like a successful, no-op LoRA run.
    """
    model.requires_grad_(False)
    visual = getattr(model, "visual", model)
    trunk = getattr(visual, "trunk", None)
    if trunk is None:
        raise AttributeError(
            "apply_lora_to_conch: expected model.visual.trunk (the timm "
            "VisionTransformer CONCH's _build_vision_tower puts there) -- not "
            "found. CONCH's internal module layout may have changed."
        )
    blocks = getattr(trunk, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise AttributeError(
            "apply_lora_to_conch: model.visual.trunk.blocks is missing or "
            "empty -- nothing to wrap"
        )
    wrapped = 0
    for block in blocks:
        attn = getattr(block, "attn", None)
        qkv = getattr(attn, "qkv", None) if attn is not None else None
        if not isinstance(qkv, nn.Linear):
            raise AttributeError(
                "apply_lora_to_conch: expected model.visual.trunk.blocks[i]"
                ".attn.qkv to be a fused nn.Linear (timm's attention shape), "
                f"got {type(qkv).__name__}. timm may have changed its "
                "VisionTransformer attention layout."
            )
        attn.qkv = FusedQKVLoRA(qkv, r=r, alpha=alpha)
        wrapped += 1
    print(f"[lora] CONCH vision trunk: wrapped {wrapped} fused qkv projections")
    return model


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Every trainable LoRA delta parameter in `model` -- `lora_A`/`lora_B`
    (`LinearLoRA`) and `lora_A_q`/`lora_B_q`/`lora_A_v`/`lora_B_v`
    (`FusedQKVLoRA` and `MultiheadAttentionLoRA`, which use the same names) --
    for building an optimizer that updates ONLY the adapters, never the frozen
    base weights `requires_grad_(False)` already excludes from autograd but
    which a naive `model.parameters()` would still hand to the optimizer as
    zero-gradient dead weight.

    Matching on parameter NAME rather than on `requires_grad` alone is what
    keeps an `r=0` run honest: at `r=0` the deltas are buffers, not
    parameters, so this returns [] and the optimizer gets only the probe --
    exactly the frozen control.
    """
    names = ("lora_A", "lora_B", "lora_A_q", "lora_B_q", "lora_A_v", "lora_B_v")
    params = []
    for name, param in model.named_parameters():
        if any(name.endswith(f".{n}") or name == n for n in names) and param.requires_grad:
            params.append(param)
    return params
