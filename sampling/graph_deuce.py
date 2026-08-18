"""graph_deuce sampler — Hướng 3, Tổ hợp B (`EXPERIMENT.md`, "Hướng 3 (ĐANG
LÀM)"): 2 MLPVAE độc lập (visual/cell, Option 1A) -> 2 kNN graph (DEUCE
GRAPHNORM, Option 2E) -> DEUCE dual-neighbor-graph merge -> 1 trong 4 biến
thể acquisition, chọn qua `acquisition_variant`:

  - "laplace_margin"            : Laplace learning + margin thuần (V1)
  - "uherding_swap_uncertainty" : coverage gốc uncertainty_herding.py (kernel
                                   Gaussian trên DINO thô) + uncertainty đổi
                                   sang laplace_margin
  - "uherding_swap_coverage"    : uncertainty gốc uncertainty_herding.py
                                   (LinearProbe+ECE-calibration trên DINO) +
                                   coverage đổi sang W_dual (kernel thưa)
  - "laplace_plus_ppr"          : laplace_margin (uncertainty) + Personalized
                                   PageRank (coverage), trộn weighted-sum +
                                   rank-normalize

Mỗi biến thể dùng CHUNG 100% phần build VAE+graph+merge (mục 7.1-7.6 của
EXPERIMENT.md) — chỉ khác công thức tính điểm/vòng lặp chọn mỗi vòng (mục
7.7/7.8/7.10). `uherding_swap_uncertainty`/`uherding_swap_coverage` cố ý
KHÔNG dùng chung vòng lặp chọn-batch với `laplace_margin`/`laplace_plus_ppr`
— chúng mirror ĐÚNG cơ chế per-point-trong-vòng của `uncertainty_herding.py`
(chọn 1 điểm, đánh giá lại toàn bộ ứng viên, lặp lại `n_select` lần/vòng),
trong khi 2 biến thể kia dùng 1 cơ chế approximate-batch tự thiết kế (rẻ hơn,
không từ paper nào) — xem EXPERIMENT.md mục 7.8/7.10 để biết lý do đánh đổi.
"""

from typing import Dict, List, Set

import numpy as np
import scipy.sparse as sps
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from trainer import train_linear
from graph_al.deuce_merge import merge_dual_neighbor_graphs
from graph_al.graph import knn_graph_partial, knn_graph_umap
from graph_al.laplace import laplace_learning, laplace_margin, personalized_pagerank
from graph_al.sparse_coverage import greedy_coverage_sparse
from graph_al.vae import MLPVAE, train_vae
from .uncertainty_herding import _calibrate_temperature
from . import register_sampler

VALID_VARIANTS = {
    "laplace_margin",
    "uherding_swap_uncertainty",
    "uherding_swap_coverage",
    "laplace_plus_ppr",
}


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Rank-normalize to [0,1] — avoids the minmax-on-near-constant-distribution
    degeneracy this project has hit repeatedly (SCALPEL v9, nucleus_coverage)."""
    order = np.argsort(np.argsort(x))
    n = len(x)
    return (order / max(1, n - 1)).astype(np.float32)


def _has_two_classes(oracle_labels: np.ndarray, selected_indices: List[int]) -> bool:
    return len(set(oracle_labels[si] for si in selected_indices)) >= 2


# ---------------------------------------------------------------------------
# Diagnostic thêm 2026-08-18: nghi ngờ VAE posterior collapse sau khi thấy
# accuracy quá tệ trên MỌI acquisition_variant (kể cả uherding_swap_coverage,
# vốn không hề dùng Laplace learning — chỉ dùng W_dual làm coverage kernel —
# nên nếu nó cũng tệ thì rất có khả năng chính W_dual/latent VAE mới là vấn
# đề, không phải phần acquisition). Đọc PDF gốc DEUCE xác nhận công thức gộp
# graph (merge_dual_neighbor_graphs, graphnorm_weights, fuzzy_symmetrize) là
# ĐÚNG — nên nghi ngờ chuyển sang input của graph (latent VAE) có thể đã
# collapse (encoder bỏ qua input, KL->0, latent gần như hằng số mọi mẫu).
# Diagnostic CHỈ đọc/in, không tự sửa gì — mục đích để xác nhận trước khi
# quyết định hướng sửa (per_point/KL-annealing/redesign acquisition/tăng k).
# ---------------------------------------------------------------------------

def _nn_distance_sample(z: torch.Tensor, n_ref: int, chunk_size: int) -> torch.Tensor:
    """Khoảng cách Euclid tới láng giềng gần nhất, tính cho `n_ref` điểm mẫu
    ngẫu nhiên so với TOÀN BỘ `z` (chunked để tránh OOM — z ở đây là latent
    VAE thô, KHÔNG L2-normalize, nên không dùng công thức sqrt(2-2cos) như
    chỗ khác trong project)."""
    N = z.shape[0]
    n_ref = min(n_ref, N)
    ref_idx = np.random.choice(N, n_ref, replace=False)
    ref = z[ref_idx]
    ref_sq = (ref ** 2).sum(dim=1)
    best = torch.full((n_ref,), float("inf"), device=z.device)

    for cs in range(0, N, chunk_size):
        ce = min(cs + chunk_size, N)
        chunk = z[cs:ce]
        chunk_sq = (chunk ** 2).sum(dim=1)
        dist_sq = torch.clamp(ref_sq[:, None] + chunk_sq[None, :] - 2.0 * (ref @ chunk.T), min=0.0)
        for i, gi in enumerate(ref_idx):
            if cs <= gi < ce:
                dist_sq[i, gi - cs] = float("inf")
        best = torch.minimum(best, dist_sq.min(dim=1).values)

    return torch.sqrt(best)


def _report_latent_diagnostics(z: torch.Tensor, name: str, chunk_size: int) -> None:
    """In 2 tín hiệu posterior-collapse cho latent `z` (N, latent_dim) của
    1 VAE đã train xong:
    1. Phương sai từng chiều latent qua toàn bộ N mẫu + số "active dims"
       (var>0.01) — chỉ báo collapse chuẩn trong tài liệu VAE: nếu encoder
       bỏ qua input (decoder chỉ học ra trung bình tập dữ liệu, KL->0), gần
       như mọi chiều sẽ có phương sai gần 0.
    2. Khoảng cách tới láng giềng gần nhất (mẫu ngẫu nhiên) — trả lời trực
       tiếp "graph có cấu trúc thật để dựng kNN hay không", độc lập với #1.
    Chỉ đọc/in, không tự sửa gì."""
    with torch.no_grad():
        var_per_dim = z.var(dim=0, unbiased=False)
        mean_var = var_per_dim.mean().item()
        total_dims = z.shape[1]
        active_dims = int((var_per_dim > 0.01).sum().item())

        nn_dist = _nn_distance_sample(z, n_ref=500, chunk_size=chunk_size)
        near_dup_frac = (nn_dist < 1e-3).float().mean().item()

    print(
        f"[graph_deuce DIAG] {name}: latent_dim={total_dims} "
        f"active_dims(var>0.01)={active_dims} mean_var={mean_var:.5f} | "
        f"NN-dist min={nn_dist.min().item():.5f} mean={nn_dist.mean().item():.5f} "
        f"max={nn_dist.max().item():.5f} near-duplicate_frac(<1e-3)={near_dup_frac:.3f}"
    )
    if active_dims <= max(1, total_dims // 4):
        print(
            f"[graph_deuce WARNING] {name}: only {active_dims}/{total_dims} latent dims "
            f"are active (var>0.01) — looks like POSTERIOR COLLAPSE (decoder likely "
            f"ignoring the latent). The resulting graph may carry near-zero real structure."
        )


# ---------------------------------------------------------------------------
# Mục 7.1-7.6: VAE độc lập + graph construction + DEUCE merge (build 1 LẦN,
# trước vòng lặp, độc lập nhãn — dùng chung cho cả 4 biến thể)
# ---------------------------------------------------------------------------

def _build_dual_graph(
    dino_np: np.ndarray,
    cell_np: np.ndarray,
    reliability: np.ndarray,
    device: torch.device,
    k: int,
    chunk_size: int,
    vae_epochs: int,
    vae_lr: float,
    vae_visual_hidden,
    vae_cell_hidden,
    vae_latent_dim: int,
    gamma: float,
    vae_batch_size: int = 512,
) -> sps.csr_matrix:
    N = dino_np.shape[0]
    x_visual = F.normalize(
        torch.as_tensor(dino_np, device=device, dtype=torch.float32), p=2, dim=1
    )

    reliable_mask = reliability > 0
    reliable_idx = np.where(reliable_mask)[0]
    if len(reliable_idx) == 0:
        raise ValueError(
            "graph_deuce needs at least some patches with reliability>0 to "
            "train VAE_cell and build the cell graph — got 0 reliable patches"
        )
    x_cell_reliable = F.normalize(
        torch.as_tensor(cell_np[reliable_mask], device=device, dtype=torch.float32), p=2, dim=1
    )

    vae_visual = MLPVAE(
        input_dim=x_visual.shape[1], hidden_dims=vae_visual_hidden, latent_dim=vae_latent_dim
    ).to(device)
    train_vae(
        vae_visual, x_visual, epochs=vae_epochs, lr=vae_lr, batch_size=vae_batch_size,
        device=device, desc="graph_deuce VAE_visual",
    )
    z_visual = vae_visual.latent(x_visual)
    del vae_visual, x_visual
    clear_memory()
    _report_latent_diagnostics(z_visual, "VAE_visual latent", chunk_size)

    vae_cell = MLPVAE(
        input_dim=x_cell_reliable.shape[1], hidden_dims=vae_cell_hidden, latent_dim=vae_latent_dim
    ).to(device)
    train_vae(
        vae_cell, x_cell_reliable, epochs=vae_epochs, lr=vae_lr, batch_size=vae_batch_size,
        device=device, desc="graph_deuce VAE_cell",
    )
    z_cell_reliable = vae_cell.latent(x_cell_reliable)
    del vae_cell, x_cell_reliable
    clear_memory()
    _report_latent_diagnostics(z_cell_reliable, "VAE_cell latent", chunk_size)

    W_visual = knn_graph_umap(z_visual, k=k, chunk_size=chunk_size)
    W_cell = knn_graph_partial(z_cell_reliable, reliable_idx, N, k=k, chunk_size=chunk_size)
    W_dual = merge_dual_neighbor_graphs(W_visual, W_cell, gamma=gamma)

    del z_visual, z_cell_reliable
    clear_memory()
    return W_dual


# Process-lifetime cache: VAE training + graph construction is fully
# unsupervised (mục 7.1 EXPERIMENT.md) — it does not depend on the AL budget
# or on `acquisition_variant`. `run.py`'s per-budget loop calls
# `graph_deuce_sampling` once per entry in `cumulative_budget` (e.g. 8x for
# [25..200]) and the SAME `train_features`/`nucleus_embeddings` numpy arrays
# are passed every time (never reassigned/copied across that loop) — so
# without this cache, both VAEs get retrained and both graphs rebuilt from
# scratch 8 times per run, and again for every acquisition_variant tried in
# the same session, for zero benefit. Keyed on `id()` of the input arrays
# (valid within one process's lifetime, per the argument above) plus every
# hyperparameter that changes the resulting graph.
_GRAPH_CACHE: Dict[tuple, sps.csr_matrix] = {}


def _build_dual_graph_cached(
    dino_np: np.ndarray,
    cell_np: np.ndarray,
    reliability: np.ndarray,
    device: torch.device,
    k: int,
    chunk_size: int,
    vae_epochs: int,
    vae_lr: float,
    vae_visual_hidden,
    vae_cell_hidden,
    vae_latent_dim: int,
    gamma: float,
    vae_batch_size: int,
) -> sps.csr_matrix:
    key = (
        id(dino_np), id(cell_np), id(reliability), k, vae_epochs, vae_lr,
        tuple(vae_visual_hidden), tuple(vae_cell_hidden), vae_latent_dim, gamma,
        vae_batch_size,
    )
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        print(
            "[graph_deuce] Reusing cached VAE+graph build from earlier this "
            "session (same pool + hyperparams) — skipping VAE retrain/graph rebuild."
        )
        return cached

    print(
        "[graph_deuce] No cached graph yet for this pool/hyperparams — training "
        "VAE_visual + VAE_cell and building W_dual now. This happens ONCE per "
        "session and is reused for every budget and every acquisition_variant "
        "after this."
    )
    W_dual = _build_dual_graph(
        dino_np, cell_np, reliability, device, k, chunk_size,
        vae_epochs, vae_lr, vae_visual_hidden, vae_cell_hidden, vae_latent_dim, gamma,
        vae_batch_size,
    )
    _GRAPH_CACHE[key] = W_dual
    return W_dual


# ---------------------------------------------------------------------------
# Mục 7.8: chọn-batch approximate cho các biến thể có Score = scalar/điểm
# (laplace_margin, laplace_plus_ppr) — KHÔNG dùng cho 2 biến thể uherding_*
# (chúng có vòng lặp per-point riêng, xem bên dưới).
# ---------------------------------------------------------------------------

def _select_batch_with_discount(
    scores_full: torch.Tensor,
    unlabeled_indices: List[int],
    current_need: int,
    W_dual: sps.spmatrix,
    device: torch.device,
) -> List[int]:
    """Chọn `current_need` điểm theo `scores_full`, giảm điểm láng giềng-graph
    sau mỗi lần chọn TRONG CÙNG lệnh gọi này (không re-solve `scores_full`
    giữa các điểm) — tránh chọn trùng cụm trong 1 vòng, không đổi ý nghĩa
    "Uncertainty = margin/Score thuần" (mục 7.8/mục 6#5 EXPERIMENT.md)."""
    N = scores_full.shape[0]
    unlabeled_arr = np.asarray(unlabeled_indices, dtype=np.int64)
    n_unlabeled = len(unlabeled_arr)

    global_to_local = -np.ones(N, dtype=np.int64)
    global_to_local[unlabeled_arr] = np.arange(n_unlabeled)

    local_scores = scores_full[torch.as_tensor(unlabeled_arr, device=device)].clone()
    discount = torch.ones(n_unlabeled, device=device, dtype=torch.float32)
    remaining_mask = torch.ones(n_unlabeled, dtype=torch.bool, device=device)

    W_csr = W_dual.tocsr()
    neg_inf = torch.tensor(-float("inf"), device=device)
    picks: List[int] = []

    for _ in range(current_need):
        effective = torch.where(remaining_mask, local_scores * discount, neg_inf)
        best_local = int(torch.argmax(effective).item())
        best_idx = int(unlabeled_arr[best_local])
        picks.append(best_idx)
        remaining_mask[best_local] = False

        row_start, row_end = W_csr.indptr[best_idx], W_csr.indptr[best_idx + 1]
        row_cols = W_csr.indices[row_start:row_end]
        row_vals = W_csr.data[row_start:row_end]
        if len(row_cols) > 0:
            local_positions = global_to_local[row_cols]
            valid = local_positions >= 0
            if valid.any():
                lp = torch.as_tensor(local_positions[valid], device=device, dtype=torch.long)
                dv = torch.as_tensor(row_vals[valid], device=device, dtype=torch.float32)
                discount[lp] = discount[lp] * (1.0 - dv)

    return picks


def _round_scores_laplace_margin(
    W_dual: sps.spmatrix,
    selected_indices: List[int],
    oracle_labels: np.ndarray,
    num_classes: int,
    N: int,
    device: torch.device,
) -> torch.Tensor:
    if not _has_two_classes(oracle_labels, selected_indices):
        return torch.ones(N, device=device, dtype=torch.float32)
    u = laplace_learning(
        W_dual, np.asarray(selected_indices), oracle_labels[selected_indices], num_classes
    )
    margin = laplace_margin(u)
    return torch.as_tensor(margin, device=device, dtype=torch.float32)


def _round_scores_laplace_plus_ppr(
    W_dual: sps.spmatrix,
    selected_indices: List[int],
    oracle_labels: np.ndarray,
    num_classes: int,
    N: int,
    device: torch.device,
    damping: float,
    alpha: float,
) -> torch.Tensor:
    if not _has_two_classes(oracle_labels, selected_indices):
        return torch.ones(N, device=device, dtype=torch.float32)
    u = laplace_learning(
        W_dual, np.asarray(selected_indices), oracle_labels[selected_indices], num_classes
    )
    uncertainty = laplace_margin(u)
    pi = personalized_pagerank(W_dual, np.asarray(selected_indices), damping=damping)
    coverage = -pi  # LOW pi (far from labeled set) -> HIGH coverage priority
    score = (1.0 - alpha) * _rank_normalize(uncertainty) + alpha * _rank_normalize(coverage)
    return torch.as_tensor(score, device=device, dtype=torch.float32)


def _unlabeled_array(N: int, selected_set: Set[int]) -> np.ndarray:
    return np.setdiff1d(
        np.arange(N), np.fromiter(selected_set, dtype=np.int64, count=len(selected_set)),
        assume_unique=True,
    )


# ---------------------------------------------------------------------------
# per_point=True: giải lại Laplace learning sau MỖI điểm chọn — đúng vòng lặp
# tuần tự thật của bài gốc SARGraphAL (`active_learning.py::active_learning_loop`,
# xem docstring `graph_al/laplace.py`), thay vì 1 lần/vòng (mặc định,
# `_select_batch_with_discount`). Chỉ áp dụng cho laplace_margin/laplace_plus_ppr
# — 2 biến thể uherding_swap_* KHÔNG dùng cơ chế này (uncertainty của chúng cố
# ý giữ nguyên round-based như chính uncertainty_herding.py gốc, xem 7.10.1/7.10.2).
# Chi phí: ~budget lần giải CG thay vì ~num_rounds lần — mỗi lần đã song song
# hoá theo lớp (graph_al/laplace.py) nhưng số lần gọi tăng đáng kể.
# ---------------------------------------------------------------------------

def _select_points_per_point_laplace_margin(
    W_dual: sps.spmatrix,
    selected_indices: List[int],
    selected_set: Set[int],
    oracle_labels: np.ndarray,
    num_classes: int,
    N: int,
    device: torch.device,
    n_select: int,
) -> List[int]:
    picks: List[int] = []
    for _ in tqdm(range(n_select), desc="graph_deuce laplace_margin (per_point)"):
        scores = _round_scores_laplace_margin(
            W_dual, selected_indices + picks, oracle_labels, num_classes, N, device,
        )
        unlabeled = _unlabeled_array(N, selected_set)
        unlabeled_t = torch.as_tensor(unlabeled, device=device, dtype=torch.long)
        best_idx = int(unlabeled[int(torch.argmax(scores[unlabeled_t]).item())])
        picks.append(best_idx)
        selected_set.add(best_idx)
    return picks


def _select_points_per_point_laplace_plus_ppr(
    W_dual: sps.spmatrix,
    selected_indices: List[int],
    selected_set: Set[int],
    oracle_labels: np.ndarray,
    num_classes: int,
    N: int,
    device: torch.device,
    n_select: int,
    ppr_damping: float,
    ppr_alpha: float,
) -> List[int]:
    picks: List[int] = []
    for _ in tqdm(range(n_select), desc="graph_deuce laplace_plus_ppr (per_point)"):
        scores = _round_scores_laplace_plus_ppr(
            W_dual, selected_indices + picks, oracle_labels, num_classes, N, device,
            ppr_damping, ppr_alpha,
        )
        unlabeled = _unlabeled_array(N, selected_set)
        unlabeled_t = torch.as_tensor(unlabeled, device=device, dtype=torch.long)
        best_idx = int(unlabeled[int(torch.argmax(scores[unlabeled_t]).item())])
        picks.append(best_idx)
        selected_set.add(best_idx)
    return picks


# ---------------------------------------------------------------------------
# Mục 7.10.1: uherding_swap_uncertainty — coverage kernel gốc
# `uncertainty_herding.py` (DENSE, trên DINO, KHÔNG dùng VAE/graph),
# uncertainty đổi sang laplace_margin. Mirror ĐÚNG vòng lặp per-point gốc
# (lines 131-221 của uncertainty_herding.py) — không dùng discount-batch.
# ---------------------------------------------------------------------------

def _bootstrap_dense_sigma(dino_features_norm: torch.Tensor, device: torch.device) -> float:
    N = dino_features_norm.shape[0]
    n_ref = min(1000, N)
    ref_idx = np.random.choice(N, n_ref, replace=False)
    ref = dino_features_norm[ref_idx]
    sim_ref = torch.matmul(ref, dino_features_norm.T)
    for i, gi in enumerate(ref_idx):
        sim_ref[i, gi] = -2.0
    nn_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sim_ref.max(dim=1).values, min=0.0))
    sigma = max(nn_dist.mean().item(), 1e-3)
    del ref, sim_ref, nn_dist
    clear_memory()
    return sigma


def _run_uherding_swap_uncertainty_round(
    dino_features_norm: torch.Tensor,
    W_dual: sps.spmatrix,
    oracle_labels: np.ndarray,
    num_classes: int,
    selected_indices: List[int],
    selected_set: Set[int],
    state: Dict,
    n_select: int,
    chunk_size: int,
    device: torch.device,
) -> List[int]:
    N = dino_features_norm.shape[0]

    if len(selected_indices) >= 2:
        sel_feats = dino_features_norm[selected_indices]
        sel_sim = torch.matmul(sel_feats, sel_feats.T)
        sel_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sel_sim, min=0.0))
        sel_dist.fill_diagonal_(float("inf"))
        valid_dist = sel_dist[sel_dist > 1e-6]
        if valid_dist.numel() > 0:
            state["sigma"] = max(valid_dist.min().item(), 1e-3)
        del sel_feats, sel_sim, sel_dist
        clear_memory()

        state["k_running"].zero_()
        for si in selected_indices:
            si_feat = dino_features_norm[si].unsqueeze(0)
            for cs in range(0, N, chunk_size):
                ce = min(cs + chunk_size, N)
                chunk = dino_features_norm[cs:ce]
                sim_c = torch.matmul(chunk, si_feat.T).squeeze(1)
                dist_sq_c = torch.clamp(2.0 - 2.0 * sim_c, min=0.0)
                k_c = torch.exp(-dist_sq_c / (state["sigma"] ** 2))
                state["k_running"][cs:ce] = torch.maximum(state["k_running"][cs:ce], k_c)
                del chunk, sim_c, dist_sq_c, k_c
        clear_memory()

        if _has_two_classes(oracle_labels, selected_indices):
            u = laplace_learning(
                W_dual, np.asarray(selected_indices), oracle_labels[selected_indices], num_classes
            )
            margin = laplace_margin(u)
            U = torch.as_tensor(margin, device=device, dtype=torch.float32)
        else:
            U = torch.ones(N, device=device, dtype=torch.float32)
    else:
        U = torch.ones(N, device=device, dtype=torch.float32)

    sigma = state["sigma"]
    k_running = state["k_running"]
    picks: List[int] = []

    for _ in tqdm(range(n_select), desc="graph_deuce uherding_swap_uncertainty"):
        best_idx = -1
        best_score = -float("inf")
        best_k_col = None

        for cs in range(0, N, chunk_size):
            ce = min(cs + chunk_size, N)
            cand = dino_features_norm[cs:ce]
            sim = torch.matmul(dino_features_norm, cand.T)
            dist_sq = torch.clamp(2.0 - 2.0 * sim, min=0.0)
            k_vals = torch.exp(-dist_sq / (sigma ** 2))
            gain = torch.clamp(k_vals - k_running.unsqueeze(1), min=0.0)
            scores = (U.unsqueeze(1) * gain).sum(dim=0)

            for si in selected_set:
                if cs <= si < ce:
                    scores[si - cs] = -float("inf")

            local_best = torch.argmax(scores).item()
            if scores[local_best].item() > best_score:
                best_score = scores[local_best].item()
                best_idx = cs + local_best
                best_k_col = k_vals[:, local_best].clone()

            del cand, sim, dist_sq, k_vals, gain, scores
            clear_memory()

        if best_idx >= 0 and best_idx not in selected_set:
            picks.append(best_idx)
            selected_set.add(best_idx)
            k_running.copy_(torch.maximum(k_running, best_k_col))
            del best_k_col
            clear_memory()
        else:
            break

    return picks


# ---------------------------------------------------------------------------
# Mục 7.10.2: uherding_swap_coverage — uncertainty gốc `uncertainty_herding.py`
# (LinearProbe+ECE trên DINO), coverage đổi sang W_dual (kernel thưa, qua
# `greedy_coverage_sparse` — cũng per-point, không phải discount-batch).
# ---------------------------------------------------------------------------

def _round_uncertainty_uherding_original(
    dino_np: np.ndarray,
    oracle_labels: np.ndarray,
    selected_indices: List[int],
    num_classes: int,
    probe_epochs: int,
    probe_lr: float,
    device: torch.device,
) -> torch.Tensor:
    N = dino_np.shape[0]
    if len(selected_indices) < 2:
        return torch.ones(N, device=device, dtype=torch.float32)

    tau = _calibrate_temperature(
        dino_np, oracle_labels, selected_indices, num_classes, probe_epochs, probe_lr, device,
    )
    probe = train_linear(
        dino_np[selected_indices], oracle_labels[selected_indices],
        num_classes, probe_epochs, probe_lr, device,
    )
    logits = probe.predict_logits(dino_np, device)
    probs = F.softmax(torch.as_tensor(logits / tau, dtype=torch.float32), dim=1).numpy()
    s_probs = np.sort(probs, axis=1)
    margin = s_probs[:, -1] - s_probs[:, -2]
    del probe
    clear_memory()
    return torch.as_tensor(1.0 - margin, device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Sampler chính
# ---------------------------------------------------------------------------

@register_sampler("graph_deuce")
def graph_deuce_sampling(**kwargs) -> List[int]:
    dino_np = np.asarray(kwargs["image_embeddings"], dtype=np.float32)
    cell_np = np.asarray(kwargs["nucleus_embeddings"], dtype=np.float32)
    reliability = np.asarray(kwargs["nucleus_reliability"], dtype=np.float32)
    oracle_labels = np.asarray(kwargs["oracle_labels"])
    num_classes = int(kwargs["num_classes"])
    max_budget = int(kwargs["max_budget"])
    device = kwargs["device"]

    acquisition_variant = kwargs.get("acquisition_variant", "laplace_margin")
    if acquisition_variant not in VALID_VARIANTS:
        raise ValueError(
            f"Unknown acquisition_variant={acquisition_variant!r}, "
            f"expected one of {sorted(VALID_VARIANTS)}"
        )

    k = int(kwargs.get("k", 20))
    chunk_size = int(kwargs.get("chunk_size", 2000))
    vae_epochs = int(kwargs.get("vae_epochs", 100))
    vae_lr = float(kwargs.get("vae_lr", 1e-3))
    vae_visual_hidden = tuple(kwargs.get("vae_visual_hidden", (256, 128)))
    vae_cell_hidden = tuple(kwargs.get("vae_cell_hidden", (128, 64)))
    vae_latent_dim = int(kwargs.get("vae_latent_dim", 32))
    gamma = float(kwargs.get("gamma", 1.0))
    probe_epochs = int(kwargs.get("probe_epochs", 50))
    probe_lr = float(kwargs.get("probe_lr", 1e-3))
    ppr_damping = float(kwargs.get("ppr_damping", 0.85))
    ppr_alpha = float(kwargs.get("ppr_alpha", 0.5))
    vae_batch_size = int(kwargs.get("vae_batch_size", 512))
    per_point = bool(kwargs.get("per_point", False))

    N = dino_np.shape[0]
    B = min(max_budget, N)
    step_budget = max(1, int(0.2 * B))

    W_dual = _build_dual_graph_cached(
        dino_np, cell_np, reliability, device, k, chunk_size,
        vae_epochs, vae_lr, vae_visual_hidden, vae_cell_hidden, vae_latent_dim, gamma,
        vae_batch_size,
    )

    dino_features_norm = F.normalize(
        torch.as_tensor(dino_np, device=device, dtype=torch.float32), p=2, dim=1
    )

    selected_indices: List[int] = []
    selected_set: Set[int] = set()

    uherding_state = None
    if acquisition_variant == "uherding_swap_uncertainty":
        uherding_state = {
            "sigma": _bootstrap_dense_sigma(dino_features_norm, device),
            "k_running": torch.zeros(N, device=device, dtype=torch.float32),
        }

    while len(selected_indices) < B:
        current_need = min(step_budget, B - len(selected_indices))

        if acquisition_variant in ("laplace_margin", "laplace_plus_ppr") and not _has_two_classes(
            oracle_labels, selected_indices
        ):
            # Cold start: Laplace learning needs >=2 labeled classes, so there is
            # no real uncertainty signal yet. Previously this branch fed a
            # uniform `scores=ones(N)` into `_select_batch_with_discount`, whose
            # very first pick is `argmax` over an all-tied array — torch breaks
            # ties by LOWEST INDEX, so the "coverage" round-1 pick was really
            # just pool-index-0 (or whichever index survives ties), unrelated to
            # any coverage/diversity property, unlike the coverage-only-round-1
            # convention every other sampler in this project follows (see
            # CLAUDE.md 2×2 sampler table). Use the SAME genuine facility-location
            # greedy the uherding_swap_coverage variant uses for its own round 1
            # (U=ones over the real graph structure) instead — a real, data-
            # dependent representative pick. Found + fixed 2026-08-17 while
            # investigating low budget=25 accuracy.
            ones = torch.ones(N, device=device, dtype=torch.float32)
            chosen = greedy_coverage_sparse(W_dual, ones, current_need, selected_set, device)

        elif acquisition_variant == "laplace_margin" and per_point:
            chosen = _select_points_per_point_laplace_margin(
                W_dual, selected_indices, selected_set, oracle_labels, num_classes, N, device, current_need,
            )

        elif acquisition_variant == "laplace_margin":
            unlabeled_indices = _unlabeled_array(N, selected_set).tolist()
            scores = _round_scores_laplace_margin(
                W_dual, selected_indices, oracle_labels, num_classes, N, device,
            )
            chosen = _select_batch_with_discount(scores, unlabeled_indices, current_need, W_dual, device)

        elif acquisition_variant == "laplace_plus_ppr" and per_point:
            chosen = _select_points_per_point_laplace_plus_ppr(
                W_dual, selected_indices, selected_set, oracle_labels, num_classes, N, device,
                current_need, ppr_damping, ppr_alpha,
            )

        elif acquisition_variant == "laplace_plus_ppr":
            unlabeled_indices = _unlabeled_array(N, selected_set).tolist()
            scores = _round_scores_laplace_plus_ppr(
                W_dual, selected_indices, oracle_labels, num_classes, N, device,
                ppr_damping, ppr_alpha,
            )
            chosen = _select_batch_with_discount(scores, unlabeled_indices, current_need, W_dual, device)

        elif acquisition_variant == "uherding_swap_uncertainty":
            chosen = _run_uherding_swap_uncertainty_round(
                dino_features_norm, W_dual, oracle_labels, num_classes,
                selected_indices, selected_set, uherding_state,
                current_need, chunk_size, device,
            )

        else:  # "uherding_swap_coverage"
            U = _round_uncertainty_uherding_original(
                dino_np, oracle_labels, selected_indices, num_classes,
                probe_epochs, probe_lr, device,
            )
            chosen = greedy_coverage_sparse(W_dual, U, current_need, selected_set, device)

        if not chosen:
            break
        selected_indices.extend(chosen)
        selected_set.update(chosen)

    del W_dual, dino_features_norm
    clear_memory()
    return selected_indices
