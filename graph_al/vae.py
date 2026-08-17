"""Unsupervised MLP-VAE for fusing/compressing DINOv2 + CellViT embeddings into
a graph-ready latent space (Hướng 3, xem EXPERIMENT.md).

Trained ONCE on the full pool before AL starts — fully unsupervised, no
oracle labels needed (see EXPERIMENT.md "Hướng 3 > Bối cảnh"). Two deliberate
deviations from the official SARGraphAL CVAE
(`repos/MSTAR-Active-Learning/Python/models.py::CVAE`, `trainVAE.py`):

1. Reconstruction loss is MSE, not the official BCE. BCE only makes sense
   there because SAR magnitude/phase images are rescaled to [0,1] pixel
   intensities before training; DINO/CellViT embeddings are unbounded real
   vectors, so BCE would be an invalid likelihood model (undefined outside
   [0,1]).
2. `encode()`/`latent()` return `mu` (deterministic, no sampling noise). The
   official `CVAE.encode()` actually returns the pre-latent flattened conv
   feature (7744-dim), NOT the true 32-dim reparameterized latent — a quirk
   of adapting a CNN classifier's `encode()` convention onto a VAE, not a
   deliberate design choice. Our encoder is a plain MLP with no conv
   structure to preserve, so there is no equivalent "pre-bottleneck feature"
   worth keeping; `mu` is the literal code the ELBO shapes into a smooth
   prior and is the natural choice for graph construction.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _mlp_stack(dims: Sequence[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class MLPVAE(nn.Module):
    """Single-input MLP-VAE.

    Covers Option 1A (one instance per modality, independent latents) and
    Option 1C (one instance on the concatenated `[dino, cellvit]` vector) of
    Hướng 3 — those two options differ only in what the caller feeds as
    `input_dim`/data, not in the model itself.
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (512, 256), latent_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = _mlp_stack([input_dim, *hidden_dims])
        last_hidden = hidden_dims[-1]
        self.fc_mu = nn.Linear(last_hidden, latent_dim)
        self.fc_logvar = nn.Linear(last_hidden, latent_dim)
        self.decoder = nn.Sequential(
            _mlp_stack([latent_dim, *reversed(hidden_dims)]),
            nn.Linear(hidden_dims[0], input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.inference_mode()
    def latent(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        mu, _ = self.encode(x)
        self.train(was_training)
        return mu


class DualBranchVAE(nn.Module):
    """Shared-core VAE with 2 separate input/output heads (Option 1B).

    One head per modality (visual/cell), projected to a common width before
    entering a SHARED encoder trunk + shared latent (`fc_mu`/`fc_logvar`) +
    shared decoder trunk, so a visual-only patch and a cell-covered patch
    land in the SAME metric space without their raw dims needing to match.
    An align loss (`||mu_visual - mu_cell||²`, cell branch only for patches
    with `reliability>0` — see `nucleus.ragged.pool_ragged_features`) pulls
    both branches' latents toward each other for the same patch; anneal its
    weight in from 0 (`align_weight` in `dual_branch_loss`/`train_dual_branch_vae`)
    to avoid posterior collapse from day one — flagged as a real risk in
    EXPERIMENT.md Option 1B (2 branches with almost no shared structure being
    forced together too hard, too early).
    """

    def __init__(
        self,
        visual_dim: int,
        cell_dim: int,
        proj_dim: int = 256,
        hidden_dims: Sequence[int] = (256,),
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_head = nn.ModuleDict({
            "visual": nn.Linear(visual_dim, proj_dim),
            "cell": nn.Linear(cell_dim, proj_dim),
        })
        self.core_encoder = _mlp_stack([proj_dim, *hidden_dims])
        last_hidden = hidden_dims[-1]
        self.fc_mu = nn.Linear(last_hidden, latent_dim)
        self.fc_logvar = nn.Linear(last_hidden, latent_dim)
        self.core_decoder = nn.Sequential(
            _mlp_stack([latent_dim, *reversed(hidden_dims)]),
            nn.Linear(hidden_dims[0], proj_dim),
            nn.ReLU(inplace=True),
        )
        self.output_head = nn.ModuleDict({
            "visual": nn.Linear(proj_dim, visual_dim),
            "cell": nn.Linear(proj_dim, cell_dim),
        })

    def encode_branch(self, x: torch.Tensor, branch: str) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.core_encoder(self.input_head[branch](x))
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode_branch(self, z: torch.Tensor, branch: str) -> torch.Tensor:
        return self.output_head[branch](self.core_decoder(z))

    def forward_branch(self, x: torch.Tensor, branch: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_branch(x, branch)
        z = self.reparameterize(mu, logvar)
        return self.decode_branch(z, branch), mu, logvar

    @torch.inference_mode()
    def encode_for_graph(
        self,
        visual: torch.Tensor,
        cell: torch.Tensor,
        reliability: torch.Tensor,
        combine: str = "mean",
    ) -> torch.Tensor:
        """Per-patch representative latent for graph construction.

        `combine="mean"` (default): average of `mu_visual`/`mu_cell` for
        patches with `reliability>0`, `mu_visual` alone otherwise (natural
        fallback — no ad-hoc zero-fill needed, the visual branch is always
        available). `combine="visual_only"`: always `mu_visual`, ignoring the
        cell branch entirely (ablation — isolates whether the cell branch
        contributes anything beyond its effect on training).
        """
        if combine not in ("mean", "visual_only"):
            raise ValueError(f"Unknown combine={combine!r}, expected 'mean' or 'visual_only'")

        was_training = self.training
        self.eval()
        mu_visual, _ = self.encode_branch(visual, "visual")
        z = mu_visual.clone()
        if combine == "mean":
            valid = reliability > 0
            if valid.any():
                mu_cell, _ = self.encode_branch(cell[valid], "cell")
                z[valid] = 0.5 * (mu_visual[valid] + mu_cell)
        self.train(was_training)
        return z


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard ELBO, sum-reduced (matches the official code's `reduction='sum'`
    convention so `beta` scales consistently with typical beta-VAE usage)."""
    recon_loss = F.mse_loss(recon, target, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.detach(), kl.detach()


def train_vae(
    model: MLPVAE,
    data: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    beta: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> List[float]:
    """Trains `model` in place on `data` (N, input_dim). Returns per-epoch mean loss."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(data), batch_size=min(batch_size, len(data)), shuffle=True)

    history: List[float] = []
    model.train()
    for _ in range(epochs):
        running = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(xb)
            loss, _, _ = vae_loss(recon, xb, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            running += loss.item()
        history.append(running / len(data))
    return history


def dual_branch_loss(
    model: DualBranchVAE,
    visual: torch.Tensor,
    cell: torch.Tensor,
    reliability: torch.Tensor,
    beta: float = 1.0,
    align_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_v, mu_v, logvar_v = model.forward_branch(visual, "visual")
    loss_v, _, _ = vae_loss(recon_v, visual, mu_v, logvar_v, beta=beta)

    valid = reliability > 0
    loss_c = torch.zeros((), device=visual.device)
    align = torch.zeros((), device=visual.device)
    if valid.any():
        recon_c, mu_c, logvar_c = model.forward_branch(cell[valid], "cell")
        loss_c, _, _ = vae_loss(recon_c, cell[valid], mu_c, logvar_c, beta=beta)
        align = F.mse_loss(mu_v[valid], mu_c, reduction="sum")

    total = loss_v + loss_c + align_weight * align
    return total, loss_v.detach(), loss_c.detach(), align.detach()


def train_dual_branch_vae(
    model: DualBranchVAE,
    visual: torch.Tensor,
    cell: torch.Tensor,
    reliability: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    beta: float = 1.0,
    align_weight_max: float = 1.0,
    align_anneal_epochs: int = 20,
    device: torch.device = torch.device("cpu"),
) -> List[float]:
    """Trains `model` in place. `align_weight` ramps linearly from 0 to
    `align_weight_max` over the first `align_anneal_epochs` epochs (posterior
    collapse guard, see class docstring), then holds at `align_weight_max`.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(visual, cell, reliability),
        batch_size=min(batch_size, len(visual)),
        shuffle=True,
    )

    history: List[float] = []
    model.train()
    for epoch in range(epochs):
        align_weight = align_weight_max * min(1.0, (epoch + 1) / max(1, align_anneal_epochs))
        running = 0.0
        for vb, cb, rb in loader:
            vb, cb, rb = vb.to(device), cb.to(device), rb.to(device)
            optimizer.zero_grad()
            loss, _, _, _ = dual_branch_loss(model, vb, cb, rb, beta=beta, align_weight=align_weight)
            loss.backward()
            optimizer.step()
            running += loss.item()
        history.append(running / len(visual))
    return history
