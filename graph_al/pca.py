"""PCA-based dimensionality reduction — deterministic, non-trainable
alternative to `graph_al.vae.MLPVAE` for building the visual/cell latent used
in `graph_deuce`'s kNN graph construction (Hướng 3, `embedding_reduction="pca"`).

Added 2026-08-18 while diagnosing catastrophically low accuracy across every
`graph_deuce` acquisition_variant. Isolates whether VAE POSTERIOR COLLAPSE
(see `graph_al/vae.py` docstring, CLAUDE.md 2026-08-18 entry) is the root
cause: PCA cannot collapse — no training loop, no KL term, a fixed
closed-form linear projection of the actual data — so if switching to PCA
also fixes accuracy, VAE collapse was indeed the culprit; if not, the
problem lies elsewhere (acquisition mechanism / graph sparsity / scale
mismatch relative to SARGraphAL's original small-scale validation).
"""

import torch


def pca_reduce(x: torch.Tensor, q: int) -> torch.Tensor:
    """Projects `x` (N, D) onto its top-`q` principal components.

    Deliberately implemented via eigendecomposition of the (D, D) covariance
    matrix (`torch.linalg.eigh`) rather than `torch.pca_lowrank` — the latter
    uses a RANDOMIZED low-rank SVD internally (a random Gaussian test matrix
    advances the global torch RNG state each call and gives numerically
    different results run-to-run even with niter>1), which would silently
    break this project's `set_seed`-based reproducibility convention. `eigh`
    is exact and fully deterministic for the same input, and cheap here since
    D (768 for visual, typically a few hundred for cell embeddings) is far
    smaller than N — the only O(N·D²) cost is the single `x.T @ x` matmul.

    If the data's intrinsic rank (`min(N-1, D)`) is smaller than `q` (only
    possible on tiny/toy pools — real datasets here have N in the tens of
    thousands and D>=768, far above any realistic `q`), the extra columns are
    zero-padded rather than raising: those dimensions then contribute 0 to
    every pairwise distance downstream, which is harmless (equivalent to
    simply not having them) and keeps a fixed output width for the kNN/graph
    code that follows.
    """
    n, d = x.shape
    eff_q = max(1, min(q, n - 1, d))

    mean = x.mean(dim=0, keepdim=True)
    xc = x - mean
    cov = (xc.T @ xc) / max(1, n - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending order, real symmetric input
    top_idx = torch.argsort(eigvals, descending=True)[:eff_q]
    top_vecs = eigvecs[:, top_idx]

    z = xc @ top_vecs
    if eff_q < q:
        pad = torch.zeros(n, q - eff_q, device=x.device, dtype=z.dtype)
        z = torch.cat([z, pad], dim=1)
    return z
