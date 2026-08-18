"""Graph-based Active Learning infra (Hướng 3, xem EXPERIMENT.md): VAE fusion
of DINOv2 + CellViT embeddings, kNN graphs, DEUCE dual-neighbor-graph merge,
Laplace learning, Personalized PageRank — reimplemented from
`repos/MSTAR-Active-Learning/` (SARGraphAL) and `pdfs/DEUCE_2502.00305.pdf`
without adding either paper's own software dependency (`graphlearning`).
"""

from .deuce_acquisition import (
    deuce_native_round,
    fps_on_graph,
    hdbscan_propagate_uncertainty,
    similarity_to_distance,
)
from .deuce_merge import merge_dual_neighbor_graphs
from .graph import (
    fuzzy_symmetrize,
    graphnorm_weights,
    knn_graph,
    knn_graph_partial,
    knn_graph_umap,
)
from .laplace import laplace_learning, laplace_margin, personalized_pagerank
from .pca import pca_reduce
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
    "deuce_native_round",
    "dual_branch_loss",
    "fps_on_graph",
    "fuzzy_symmetrize",
    "graphnorm_weights",
    "greedy_coverage_sparse",
    "hdbscan_propagate_uncertainty",
    "knn_graph",
    "knn_graph_partial",
    "knn_graph_umap",
    "laplace_learning",
    "laplace_margin",
    "merge_dual_neighbor_graphs",
    "pca_reduce",
    "personalized_pagerank",
    "similarity_to_distance",
    "train_dual_branch_vae",
    "train_vae",
    "vae_loss",
]
