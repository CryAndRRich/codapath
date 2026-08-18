"""DEUCE's own Acquisition Module (Section 3.2.4, Algorithm 1,
`pdfs/DEUCE_2502.00305.pdf`) — HDBSCAN* clustering on the Dual-Neighbor
Graph, within-cluster uncertainty propagation, and Farthest Point Sampling
(FPS). This is the mechanism the paper ACTUALLY uses to turn a merged graph
+ per-point uncertainty into a selected set — `graph_deuce`'s other 3
acquisition_variants (`laplace_margin`, `uherding_swap_*`,
`laplace_plus_ppr`) all skip this and feed the merged graph into an
unrelated mechanism instead (SARGraphAL's Laplace learning, or
`uncertainty_herding`'s dense-kernel greedy). Implemented here as
`acquisition_variant="deuce_native"` (`sampling/graph_deuce.py`) to test
whether the paper's OWN acquisition — not just its graph-construction
recipe — is what makes this kind of graph useful. See CLAUDE.md 2026-08-18
entry for the investigation that motivated this.

Deviation from the paper, necessary and documented: DEUCE's own `u_i` is
computed ONCE, entirely label-free (Section 3.2.2, entropy of a PLM's
OVA-calibrated zero-shot prediction) — the whole algorithm is a ONE-SHOT
cold-start seed-set selector, no AL rounds at all. Both uncertainty sources
this project reuses for `deuce_native` (`laplace_margin`, LinearProbe+ECE)
REQUIRE labeled points to compute at all, so `deuce_native` must run as an
ITERATIVE AL loop (same 5-round convention as the other 3 variants) rather
than the paper's one-shot design — round 1 (no labels yet) falls back to
`graph_al.sparse_coverage.greedy_coverage_sparse` with uniform uncertainty
(the same cold-start path `laplace_margin`/`laplace_plus_ppr` use), and only
rounds 2+ run the mechanism in this file with the real, labeled-dependent
`u_i`.

Empirically verified 2026-08-18 (not assumed from docs, which are
ambiguous): `sklearn.cluster.HDBSCAN(metric="precomputed")` DOES accept a
sparse `scipy.sparse` distance matrix and correctly treats missing
(unstored) entries as "no edge" / unreachable — NEVER as zero distance — but
raises a hard `ValueError` if the graph has more than one connected
component. `hdbscan_propagate_uncertainty` below restricts clustering to the
graph's largest connected component and treats every other node as
unclustered (noise, probability 0) instead of letting a rare disconnected
pool crash the whole run. Timing at N=100,000 with k=20 (this project's
default): a single `sklearn.cluster.HDBSCAN` sparse-precomputed fit takes
~2.4s, a single `scipy.sparse.csgraph.dijkstra` call ~0.2-0.35s — both
verified directly in-sandbox before writing this file, not assumed.
"""

from typing import List, Optional, Set

import numpy as np
import scipy.sparse as sps
from sklearn.cluster import HDBSCAN


def similarity_to_distance(W: sps.spmatrix) -> sps.csr_matrix:
    """Monotonic similarity->distance transform for a sparse graph whose
    stored values are all > 0 (as `graph_al.deuce_merge.merge_dual_neighbor_graphs`
    guarantees for every edge it emits): `d = 1/w`.

    The DEUCE paper never specifies how its fuzzy-membership edge weights
    should be converted into a "distance" for HDBSCAN*/FPS (both of which
    only need edges to be internally ORDER-consistent — nearest-neighbor and
    farthest-point are both rank-based, not scale-based) — `1/w` is simply
    the simplest transform that is always positive and strictly decreasing
    in `w`. This is not claimed to be "the" DEUCE choice, only *a*
    principled, monotonic one, documented explicitly to avoid the kind of
    silent oversimplification found (and corrected) elsewhere in this
    project's history.
    """
    D = W.tocsr(copy=True)
    D.data = 1.0 / D.data
    return D


def hdbscan_propagate_uncertainty(
    W_dual: sps.spmatrix,
    u: np.ndarray,
    min_cluster_size: int,
) -> np.ndarray:
    """Algorithm 1 lines 10-14 (HDBSCAN* on the DNG) + 15-18 (within-cluster
    uncertainty propagation):

        ũ_i = u_i + Σ_{x_j in same cluster, {x_i,x_j} an edge of W_dual}
                        w_dual(x_i,x_j) · p_j · u_j

    `p_j` is HDBSCAN*'s own soft cluster-membership probability for point j.
    Points HDBSCAN* labels as noise, or that fall outside the graph's
    largest connected component (see module docstring), keep ũ_i = u_i
    unchanged — matching Algorithm 1's implicit "if x_i belongs to no
    cluster, don't propagate" branch (lines 16-18 only assign ũ_i when
    `∃c_l : x_i ∈ c_l`).

    Parameters
    ----------
    W_dual : (N, N) sparse SYMMETRIC similarity graph, all stored values > 0
    u : (N,) raw per-point uncertainty (label-dependent — see module docstring)
    min_cluster_size : HDBSCAN*'s `k_r` (paper default 3, tuned for a very
        different text-pool scale — not directly transferable, re-tune here)

    Returns
    -------
    u_tilde : (N,) float64 propagated uncertainty
    """
    N = W_dual.shape[0]
    W_csr = W_dual.tocsr()

    n_comp, comp_ids = sps.csgraph.connected_components(W_csr, directed=False)
    if n_comp > 1:
        sizes = np.bincount(comp_ids)
        largest = int(np.argmax(sizes))
        in_scope = np.where(comp_ids == largest)[0]
        print(
            f"[deuce_native] W_dual has {n_comp} connected components — "
            f"HDBSCAN* requires a single connected graph, so clustering only "
            f"the largest ({len(in_scope)}/{N} nodes). The rest are treated "
            f"as unclustered (no propagation), matching Algorithm 1's own "
            f"behavior for non-clustered points."
        )
    else:
        in_scope = np.arange(N)

    dist_sub = similarity_to_distance(W_csr[in_scope][:, in_scope])
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed", copy=True)
    sub_labels = clusterer.fit_predict(dist_sub)
    sub_probs = clusterer.probabilities_

    labels_full = -np.ones(N, dtype=np.int64)
    probs_full = np.zeros(N, dtype=np.float64)
    labels_full[in_scope] = sub_labels
    probs_full[in_scope] = sub_probs

    W_coo = W_csr.tocoo()
    rows, cols, vals = W_coo.row, W_coo.col, W_coo.data
    same_cluster = (
        (labels_full[rows] == labels_full[cols])
        & (labels_full[rows] != -1)
        & (rows != cols)
    )

    q = probs_full * u.astype(np.float64)
    contrib = np.zeros(N, dtype=np.float64)
    np.add.at(contrib, rows[same_cluster], vals[same_cluster] * q[cols[same_cluster]])

    return u.astype(np.float64) + contrib


def fps_on_graph(
    dist_graph: sps.csr_matrix,
    start_idx: int,
    n_select: int,
    excluded: Optional[Set[int]] = None,
) -> List[int]:
    """Farthest Point Sampling (Eldar et al., 1994) on a graph, using
    shortest-path distance (`scipy.sparse.csgraph.dijkstra`) instead of
    Euclidean distance — the graph analogue of this project's established
    "running coverage, updated per pick" pattern (e.g.
    `graph_al.sparse_coverage.greedy_coverage_sparse`'s `K_n`), just tracking
    the MINIMUM shortest-path distance to the picked set instead of a
    kernel-similarity MAXIMUM.

    `start_idx` is included as the first pick (Algorithm 1 starts FPS FROM
    each top-degree node, it doesn't pick something else first). `excluded`
    (already-labeled indices) are pinned to `-inf` so they can never be
    (re)selected, including as `start_idx` itself — validate that
    `start_idx not in excluded` before calling.
    """
    N = dist_graph.shape[0]
    running_min = np.full(N, np.inf)
    if excluded:
        running_min[list(excluded)] = -np.inf

    picks = [start_idx]
    running_min[start_idx] = -np.inf
    d = sps.csgraph.dijkstra(dist_graph, directed=False, indices=start_idx)
    running_min = np.where(running_min == -np.inf, running_min, np.minimum(running_min, d))

    for _ in range(n_select - 1):
        nxt = int(np.argmax(running_min))
        if running_min[nxt] == -np.inf:
            break  # every remaining candidate is already excluded/picked
        picks.append(nxt)
        running_min[nxt] = -np.inf
        d = sps.csgraph.dijkstra(dist_graph, directed=False, indices=nxt)
        running_min = np.where(running_min == -np.inf, running_min, np.minimum(running_min, d))

    return picks


def deuce_native_round(
    W_dual: sps.spmatrix,
    u: np.ndarray,
    min_cluster_size: int,
    fps_starts: int,
    n_select: int,
    selected_set: Set[int],
) -> List[int]:
    """One round of DEUCE's own acquisition (Algorithm 1 lines 10-20),
    restricted to the currently UNLABELED candidates (`selected_set`
    excluded) — see module docstring for why this must be called per-round
    rather than once, unlike the paper's one-shot use.

    Runs FPS from each of the `fps_starts` highest-WEIGHTED-degree unlabeled
    nodes (paper: "documents xi with top-k degrees", reusing the same `k` as
    the kNN graph itself — this project exposes it separately as
    `fps_starts` so it can be tuned independently of graph density), then
    returns whichever candidate set has the highest total propagated
    uncertainty (Σ ũ_j over its members), per the paper's own selection rule.

    Does NOT mutate `selected_set` — the caller is responsible for adding
    the returned picks, matching every other acquisition path in
    `sampling/graph_deuce.py`.
    """
    N = W_dual.shape[0]
    u_tilde = hdbscan_propagate_uncertainty(W_dual, u, min_cluster_size)

    W_csr = W_dual.tocsr()
    degree = np.asarray(W_csr.sum(axis=1)).ravel().astype(np.float64)
    if selected_set:
        degree[list(selected_set)] = -np.inf

    n_available = N - len(selected_set)
    n_starts = min(fps_starts, n_available)
    if n_starts <= 0 or n_select <= 0:
        return []
    start_candidates = np.argsort(-degree)[:n_starts]

    dist_graph = similarity_to_distance(W_csr)

    best_score = -np.inf
    best_picks: List[int] = []
    for start in start_candidates:
        picks = fps_on_graph(dist_graph, int(start), n_select, excluded=selected_set)
        score = float(u_tilde[picks].sum()) if picks else -np.inf
        if score > best_score:
            best_score = score
            best_picks = picks

    return best_picks
