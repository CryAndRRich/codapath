"""`graph_sargraph._pca` — covariance-route PCA must match the SVD route.

`_pca` deliberately avoids `torch.linalg.svd` on the full (N, D) matrix (it
would allocate `U` at ~600 MB for N=100k) and accumulates the D x D Gram matrix
in chunks instead. These tests pin the two properties that buys us: the same
subspace as SVD, and independence from `chunk_size`.
"""
import numpy as np
import torch

from sampling.graph_sargraph import _pca


def _spectrum_data(n=2000, d=120, seed=0):
    """Data with a decaying spectrum, like a real embedding."""
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=(n, d)) * np.exp(-np.arange(d) / 25.0)) @ rng.normal(size=(d, d))
    return torch.as_tensor(x, dtype=torch.float32)


def _svd_projection(x, latent_dim):
    xc = (x - x.mean(dim=0, keepdim=True)).to(torch.float64)
    _, _, vh = torch.linalg.svd(xc, full_matrices=False)
    return xc @ vh[:latent_dim].T


def _match_signs(a, b):
    """Eigenvectors are defined up to sign; align before comparing."""
    sign = torch.sign((a * b).sum(dim=0))
    sign[sign == 0] = 1.0
    return b * sign


def test_pca_shape():
    x = _spectrum_data()
    assert _pca(x, latent_dim=32, chunk_size=256).shape == (x.shape[0], 32)


def test_pca_matches_svd_projection():
    x = _spectrum_data()
    got = _pca(x, latent_dim=32, chunk_size=256).to(torch.float64)
    expected = _svd_projection(x, 32)
    aligned = _match_signs(expected, got)
    assert torch.allclose(aligned, expected, atol=1e-4, rtol=1e-4)


def test_pca_is_chunk_size_invariant():
    x = _spectrum_data()
    small = _pca(x, latent_dim=16, chunk_size=64)
    large = _pca(x, latent_dim=16, chunk_size=8192)
    assert torch.allclose(_match_signs(small, large), small, atol=1e-5)


def test_full_rank_projection_preserves_pairwise_distances():
    """latent_dim == D is a rotation: every pairwise distance must survive it,
    which is the strongest statement that no variance was dropped or rescaled."""
    x = _spectrum_data(n=300, d=40)
    z = _pca(x, latent_dim=40, chunk_size=64)
    assert torch.allclose(torch.cdist(z, z), torch.cdist(x, x), atol=1e-2)
