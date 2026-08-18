import torch

from graph_al.vae import (
    DualBranchVAE,
    MLPVAE,
    train_dual_branch_vae,
    train_vae,
    vae_loss,
)


def _toy_data(n=64, dim=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g)


def test_mlpvae_forward_shapes():
    model = MLPVAE(input_dim=10, hidden_dims=(16, 8), latent_dim=4)
    x = _toy_data()
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape
    assert mu.shape == (64, 4)
    assert logvar.shape == (64, 4)


def test_mlpvae_latent_is_deterministic():
    model = MLPVAE(input_dim=10, hidden_dims=(16, 8), latent_dim=4)
    x = _toy_data()
    z1 = model.latent(x)
    z2 = model.latent(x)
    torch.testing.assert_close(z1, z2)


def test_mlpvae_training_reduces_loss():
    model = MLPVAE(input_dim=10, hidden_dims=(16, 8), latent_dim=4)
    x = _toy_data(n=128)
    history = train_vae(model, x, epochs=20, batch_size=32, lr=1e-2)
    assert history[-1] < history[0]


def test_vae_loss_zero_kl_when_mu_zero_logvar_zero():
    recon = torch.zeros(4, 3)
    target = torch.zeros(4, 3)
    mu = torch.zeros(4, 2)
    logvar = torch.zeros(4, 2)
    total, recon_loss, kl = vae_loss(recon, target, mu, logvar)
    assert recon_loss.item() == 0.0
    assert kl.item() == 0.0
    assert total.item() == 0.0


def test_dual_branch_vae_encode_for_graph_fallback_and_mean():
    model = DualBranchVAE(visual_dim=6, cell_dim=4, proj_dim=8, hidden_dims=(8,), latent_dim=3)
    visual = _toy_data(n=5, dim=6)
    cell = _toy_data(n=5, dim=4)
    reliability = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

    z_mean = model.encode_for_graph(visual, cell, reliability, combine="mean")
    z_visual_only = model.encode_for_graph(visual, cell, reliability, combine="visual_only")

    assert z_mean.shape == (5, 3)
    mu_visual, _ = model.encode_branch(visual, "visual")
    torch.testing.assert_close(z_visual_only, mu_visual)
    # patches with reliability==0 must fall back to mu_visual alone
    torch.testing.assert_close(z_mean[1], mu_visual[1])
    torch.testing.assert_close(z_mean[3], mu_visual[3])
    # patches with reliability>0 must differ from visual-only (cell branch mixed in)
    assert not torch.allclose(z_mean[0], mu_visual[0])


def test_dual_branch_vae_training_reduces_loss():
    model = DualBranchVAE(visual_dim=6, cell_dim=4, proj_dim=8, hidden_dims=(8,), latent_dim=3)
    visual = _toy_data(n=64, dim=6)
    cell = _toy_data(n=64, dim=4)
    reliability = (torch.arange(64) % 3 != 0).float()  # ~2/3 reliable

    history = train_dual_branch_vae(
        model, visual, cell, reliability,
        epochs=15, batch_size=16, lr=1e-2, align_weight_max=0.5, align_anneal_epochs=5,
    )
    assert history[-1] < history[0]
