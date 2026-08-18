import torch

from graph_al.pca import pca_reduce


def test_pca_reduce_shape():
    torch.manual_seed(0)
    x = torch.randn(500, 768)
    z = pca_reduce(x, q=32)
    assert z.shape == (500, 32)


def test_pca_reduce_is_deterministic():
    torch.manual_seed(0)
    x = torch.randn(200, 100)
    z1 = pca_reduce(x, q=16)
    z2 = pca_reduce(x, q=16)
    assert torch.allclose(z1, z2)


def test_pca_reduce_pads_when_q_exceeds_intrinsic_rank():
    torch.manual_seed(0)
    x = torch.randn(20, 10)  # rank <= min(n-1, d) = 10
    z = pca_reduce(x, q=32)
    assert z.shape == (20, 32)
    nonzero_cols = (z.abs().sum(dim=0) > 1e-8).sum().item()
    assert nonzero_cols <= 10


def test_pca_reduce_orders_components_by_decreasing_variance():
    torch.manual_seed(0)
    # 3 informative directions with clearly decreasing spread, rest is noise
    n = 300
    a = torch.randn(n, 1) * 5.0
    b = torch.randn(n, 1) * 2.0
    c = torch.randn(n, 1) * 0.5
    noise = torch.randn(n, 20) * 0.01
    x = torch.cat([a, b, c, noise], dim=1)
    z = pca_reduce(x, q=5)
    var = z.var(dim=0)
    assert torch.all(var[:-1] >= var[1:] - 1e-6)
