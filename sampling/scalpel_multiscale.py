"""SCALPEL-Multiscale — multi-zoom morphology uncertainty (pathology-specific).

Replaces v8/v9's stain-shortcut branch with a different pathology-relevant
signal: multi-scale morphology. Every image is viewed at 3 zoom levels —
original, upscaled x2 then center-cropped back to the input size (zoomed into
roughly the center half of the field of view), upscaled x4 then center-cropped
(zoomed further in) — each re-encoded by the same frozen DINOv2 backbone. The
agreement/disagreement (or a fused representation) across these 3 views drives
uncertainty; Coverage always comes from the original-scale DINOv2 features
only, using the same UHerding-style weighted-herding greedy already used by
`scalpel.py` (`Sigma_n W_n * max(K(n,i) - K_n, 0)`), unchanged.

Fixed contract for this experiment (locked, do not reintroduce v8/v9's
explore->reconcile vacuity/`t` schedule here):
  - Round 1: pure Coverage (W = ones).
  - Rounds 2..T: `W = minmax(Uncertainty)` only, no vacuity term, no schedule.
    Because the coverage kernel is already bounded in [0,1] (Gaussian RBF),
    the `Sigma_n W_n * gain(n,i)` sum computed by `_greedy_coverage_batch` IS
    `Score(i) = norm(Uncertainty) x norm(Coverage)`.

Two uncertainty variants, selected via `fusion_mode`:
  - "disagreement" (Ver 1): one linear probe per scale, trained on that
    scale's own DINOv2 features. Uncertainty = BALD disagreement across the
    3 posteriors: `H[mean_v p_v] - mean_v H[p_v]`.
  - "concat" (Ver 2): concatenate (or sum, via `feature_fusion`) the 3 scales'
    raw DINOv2 features into one vector, train a single linear probe on that,
    and use plain margin uncertainty on its output.

Multi-scale features are used ONLY for sampling; the final train/eval linear
probe (used to compare every sampler fairly) still trains on the original-scale
DINOv2 features exactly as it does for every other sampler.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from set_up import clear_memory
from . import register_sampler
from .scalpel import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _minmax,
    _margin_uncertainty,
    _norm_rows,
    _entropy,
    _adaptive_sigma,
    _greedy_coverage_batch,
)


# ---------------------------------------------------------------------------
# Multi-scale DINOv2 feature extraction (train pool only — used for sampling)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_scaled_dino_features(loader, device: torch.device, vit_name: str,
                                  scale_factors) -> Dict[int, np.ndarray]:
    """One DINOv2 pass per batch, producing features for each requested scale.

    Per batch: recover approx RGB (inverse ImageNet normalize, same pattern as
    scalpel.py's extract_stain_features) -> upscale by `scale_factor` (bilinear)
    -> center-crop back to the batch's own H,W -> re-normalize (ImageNet) ->
    encode with a single shared DINOv2Extractor instance. One model load, one
    loader traversal produces every requested scale together.
    """
    from model import DINOv2Extractor

    mean_t = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

    extractor = DINOv2Extractor(model_name=vit_name).to(device)
    extractor.eval()

    feats: Dict[int, List[np.ndarray]] = {s: [] for s in scale_factors}

    for imgs, _ in tqdm(loader, desc="Multiscale DINO extraction", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        rgb = (imgs * std_t + mean_t).clamp(0.0, 1.0)
        H, W = rgb.shape[-2], rgb.shape[-1]

        for s in scale_factors:
            up = F.interpolate(rgb, scale_factor=s, mode="bilinear", align_corners=False)
            top = (up.shape[-2] - H) // 2
            left = (up.shape[-1] - W) // 2
            cropped = up[:, :, top:top + H, left:left + W]
            normed = (cropped - mean_t) / std_t
            out = extractor(normed)
            feats[s].append(out.cpu().numpy().astype(np.float32))
            del up, cropped, normed, out

        del imgs, rgb

    del extractor
    clear_memory()
    return {s: np.concatenate(arrs, axis=0) for s, arrs in feats.items()}


def _multiscale_cache_paths(cache_dir: str, dataset_key: str, seed: int,
                             vit_name: str, scale_factors) -> Dict[int, str]:
    safe_vit = vit_name.replace("/", "_")
    base = f"{dataset_key}_seed{seed}_{safe_vit}"
    return {s: f"{cache_dir}/{base}_scale{s}_train.npy" for s in scale_factors}


def get_or_extract_multiscale_features(train_loader, dataset_key: str, seed: int,
                                        vit_name: str, scale_factors, device: torch.device,
                                        cache_dir: str = "features") -> Dict[int, np.ndarray]:
    """Train-only cache for multi-scale DINOv2 features — mirrors
    `model.py::get_or_extract_features`'s cache-key convention (dataset+seed+
    backbone, scale-suffixed), but never needs the test split since only the
    training pool is ever sampled from.
    """
    import os

    paths = _multiscale_cache_paths(cache_dir, dataset_key, seed, vit_name, scale_factors)
    n_train = len(train_loader.dataset)

    if all(os.path.exists(p) for p in paths.values()):
        loaded = {s: np.load(p) for s, p in paths.items()}
        if all(arr.shape[0] == n_train for arr in loaded.values()):
            print(f"[multiscale-features] Loaded cache for scales {scale_factors} → {list(paths.values())}")
            return loaded
        print("[multiscale-features] Cache size mismatch — re-extracting.")

    print(f"[multiscale-features] Cache miss — extracting DINOv2 features at scales "
          f"{scale_factors} for '{dataset_key}' (seed={seed}, backbone={vit_name}).")
    feats = extract_scaled_dino_features(train_loader, device, vit_name, scale_factors)

    os.makedirs(cache_dir, exist_ok=True)
    for s, arr in feats.items():
        np.save(paths[s], arr)
    print(f"[multiscale-features] Saved cache → {list(paths.values())}")
    return feats


# ---------------------------------------------------------------------------
# Main sampling function — iterative, round-based, multi-scale uncertainty
# ---------------------------------------------------------------------------

def _bald_disagreement(probs_per_scale: List[np.ndarray]) -> np.ndarray:
    """BALD: H[mean_v p_v] - mean_v H[p_v]. High = predictions disagree across
    scales (confident individually, but flip with zoom level)."""
    mean_p = np.mean(probs_per_scale, axis=0)
    h_mean = _entropy(_norm_rows(mean_p))
    mean_h = np.mean([_entropy(_norm_rows(p)) for p in probs_per_scale], axis=0)
    return (h_mean - mean_h).astype(np.float32)


@register_sampler("scalpel_multiscale")
def scalpel_multiscale_sampling(**kwargs) -> List[int]:
    dino_np: np.ndarray = kwargs["image_embeddings"]          # (N,768) original scale — coverage + scale-1 view
    scale_feats: Dict[int, np.ndarray] = kwargs["multiscale_features"]  # {2: (N,768), 4: (N,768)}
    oracle_labels = np.asarray(kwargs["oracle_labels"])
    num_classes: int = kwargs["num_classes"]
    max_budget: int = kwargs["max_budget"]
    device: torch.device = kwargs["device"]
    chunk_size: int = kwargs.get("chunk_size", 2000)
    num_rounds: int = kwargs.get("num_rounds", 5)
    probe_epochs: int = kwargs.get("probe_epochs", 50)
    probe_lr: float = kwargs.get("probe_lr", 1e-3)
    fusion_mode: str = kwargs.get("fusion_mode", "disagreement")   # "disagreement" | "concat"
    feature_fusion: str = kwargs.get("feature_fusion", "concat")  # "concat" | "sum" (fusion_mode="concat" only)
    diag: bool = kwargs.get("diag", True)
    n_sigma: int = kwargs.get("n_sigma", 2000)

    from trainer import train_linear

    N = dino_np.shape[0]
    L = num_classes
    B = min(max_budget, N)
    T = max(1, min(num_rounds, B))

    base, rem = divmod(B, T)
    sizes = [base + (1 if r < rem else 0) for r in range(T)]

    # ---- DINOv2 morphology / coverage space (original scale) ----
    features = F.normalize(
        torch.tensor(dino_np, device=device, dtype=torch.float32), p=2, dim=1
    )
    feat_np = features.cpu().numpy()
    sigma = _adaptive_sigma(features, n_ref=n_sigma)
    K_n = torch.zeros(N, device=device, dtype=torch.float32)

    # ---- per-scale L2-normalized feature dict (scale 1 = original) ----
    scale_feat_np: Dict[int, np.ndarray] = {1: feat_np}
    for s, arr in scale_feats.items():
        t = F.normalize(torch.tensor(arr, device=device, dtype=torch.float32), p=2, dim=1)
        scale_feat_np[s] = t.cpu().numpy()
        del t

    selected_indices: List[int] = []
    selected_set: set = set()

    for r in tqdm(range(T), desc="SCALPEL-Multiscale Rounds"):
        n_select = sizes[r]
        if n_select <= 0:
            continue

        if r == 0 or len(selected_indices) < 2:
            W = torch.ones(N, device=device, dtype=torch.float32)        # cold: pure coverage
        else:
            sel = selected_indices
            y = oracle_labels[sel]

            if fusion_mode == "disagreement":
                probs_per_scale = []
                accs = {}
                for s, fnp in scale_feat_np.items():
                    probe = train_linear(fnp[sel], y, L, probe_epochs, probe_lr, device)
                    p = probe.predict_proba(fnp, device)
                    probs_per_scale.append(p)
                    if diag:
                        accs[s] = float((p.argmax(1) == oracle_labels).mean())
                    del probe
                clear_memory()

                uncertainty = _bald_disagreement(probs_per_scale)

                if diag:
                    acc_str = " ".join(f"scale{s}={a:.3f}" for s, a in accs.items())
                    print(f"[DIAG-MS b={B} r={r}] fusion=disagreement probe acc: {acc_str} "
                          f"| mean disagreement={uncertainty.mean():.4f}")
            else:  # "concat"
                if feature_fusion == "sum":
                    fused = np.sum(list(scale_feat_np.values()), axis=0)
                else:
                    fused = np.concatenate(list(scale_feat_np.values()), axis=1)

                probe = train_linear(fused[sel], y, L, probe_epochs, probe_lr, device)
                p = probe.predict_proba(fused, device)
                del probe
                clear_memory()

                uncertainty = _margin_uncertainty(p)

                if diag:
                    acc = float((p.argmax(1) == oracle_labels).mean())
                    print(f"[DIAG-MS b={B} r={r}] fusion=concat({feature_fusion}) probe acc={acc:.3f} "
                          f"| mean margin unc={uncertainty.mean():.4f}")

            w_np = _minmax(uncertainty)
            W = torch.tensor(w_np, device=device, dtype=torch.float32)
            if float(W.max()) <= 0.0:
                W = torch.ones(N, device=device, dtype=torch.float32)

        picks = _greedy_coverage_batch(
            features, W, K_n, sigma, n_select, selected_set, chunk_size,
        )
        selected_indices.extend(picks)
        del W
        clear_memory()
        if len(picks) < n_select:        # pool exhausted
            break

    del features, K_n
    clear_memory()
    return selected_indices
