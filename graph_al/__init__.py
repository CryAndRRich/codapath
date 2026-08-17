"""Graph-based Active Learning infra (Hướng 3, xem EXPERIMENT.md): VAE fusion
of DINOv2 + CellViT embeddings, kNN graphs, DEUCE dual-neighbor-graph merge,
Laplace learning, Personalized PageRank — reimplemented from
`repos/MSTAR-Active-Learning/` (SARGraphAL) and `pdfs/DEUCE_2502.00305.pdf`
without adding either paper's own software dependency (`graphlearning`).
"""

from .deuce_merge import merge_dual_neighbor_graphs
from .graph import (
    fuzzy_symmetrize,
    graphnorm_weights,
    knn_graph,
    knn_graph_partial,
    knn_graph_umap,
)
from .laplace import laplace_learning, laplace_margin, personalized_pagerank
from .sparse_coverage import greedy_coverage_sparse
from .vae import (
    DualBranchVAE,
    MLPVAE,
    dual_branch_loss,
    train_dual_branch_vae,
    train_vae,
    vae_loss,
)

__all__ = [
    "DualBranchVAE",
    "MLPVAE",
    "dual_branch_loss",
    "fuzzy_symmetrize",
    "graphnorm_weights",
    "greedy_coverage_sparse",
    "knn_graph",
    "knn_graph_partial",
    "knn_graph_umap",
    "laplace_learning",
    "laplace_margin",
    "merge_dual_neighbor_graphs",
    "personalized_pagerank",
    "train_dual_branch_vae",
    "train_vae",
    "vae_loss",
]
