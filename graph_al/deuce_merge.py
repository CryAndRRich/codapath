"""DEUCE's Dual-Neighbor Graph (DNG) merge — Sec 3.2.3 "Merging",
`pdfs/DEUCE_2502.00305.pdf` (arXiv:2502.00305), verified against the PDF
directly (not a summary). Combines two already normalized+symmetrized kNN
graphs (see `graph_al.graph.knn_graph_umap`) into one, boosting edges that
appear in BOTH input graphs ("dual-neighbor edges").
"""

import scipy.sparse as sps


def merge_dual_neighbor_graphs(
    W1: sps.spmatrix,
    W2: sps.spmatrix,
    gamma: float = 1.0,
) -> sps.csr_matrix:
    """DEUCE's merge formula:

        w_dual(i,j) = w1(i,j)*w2(i,j) + gamma   if edge (i,j) is nonzero in BOTH W1 and W2
        w_dual(i,j) = w1(i,j)                   if nonzero in W1 only
        w_dual(i,j) = w2(i,j)                   if nonzero in W2 only

    `gamma=1.0` is DEUCE's own default and is a THRESHOLD, not a tuned blend
    weight: `W1, W2` (from `knn_graph_umap`'s fuzzy-union symmetrization) are
    both `∈ (0,1]`, so `gamma=1.0` guarantees every dual-neighbor edge
    (`<= 1*1 + 1 = 2`) outweighs every single-neighbor edge (`<= 1`) — see
    EXPERIMENT.md Hướng 3 mục 0/7.6 for the full derivation and the paper's
    own hyperparameter choice (`k=500, kr=3, gamma=1.0` for their text pools;
    `gamma` does not need to be re-tuned for a different domain since its
    role is a fixed threshold, not a learned/tuned coefficient).
    """
    W1 = W1.tocsr()
    W2 = W2.tocsr()

    dual = W1.multiply(W2).tocsr()
    dual.eliminate_zeros()
    dual.data = dual.data + gamma

    single1 = (W1 - W1.multiply(W2 > 0)).tocsr()
    single2 = (W2 - W2.multiply(W1 > 0)).tocsr()

    W_dual = (dual + single1 + single2).tocsr()
    W_dual.eliminate_zeros()
    return W_dual
