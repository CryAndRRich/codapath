import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Dinov2Model


class DINOv2Extractor(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-base") -> None:
        super().__init__()
        self.encoder = Dinov2Model.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values=x).last_hidden_state[:, 0, :]


@torch.inference_mode()
def extract_image_features(dataloader: DataLoader,
                            extractor: nn.Module,
                            device: torch.device) -> np.ndarray:
    extractor = extractor.to(device)
    extractor.eval()

    all_features = []
    for images, _ in tqdm(dataloader, desc="Extracting image features", leave=False):
        images = images.to(device, non_blocking=True)
        features = extractor(images)
        all_features.append(features.cpu().numpy().astype(np.float32))

    return np.vstack(all_features)


def _feature_cache_paths(cache_dir: str,
                         dataset_key: str,
                         seed: int,
                         vit_name: str):
    """Cache filenames keyed by dataset + seed + backbone.

    Seed is part of the key because ImageFolder datasets are split with a
    seeded generator in get_data_loaders — a different seed reorders samples,
    so features are only reusable when the seed matches.
    """
    safe_vit = vit_name.replace("/", "_")
    base = f"{dataset_key}_seed{seed}_{safe_vit}"
    train_path = os.path.join(cache_dir, f"{base}_train.npy")
    test_path  = os.path.join(cache_dir, f"{base}_test.npy")
    return train_path, test_path


def get_or_extract_features(train_loader: DataLoader,
                            test_loader: DataLoader,
                            dataset_key: str,
                            seed: int,
                            vit_name: str,
                            device: torch.device,
                            cache_dir: str = "features"):
    """Return (train_features, test_features) for the frozen DINOv2 backbone.

    Loads them from `cache_dir` when a valid cache exists (matching dataset,
    seed, backbone AND sample count); otherwise builds the extractor, computes
    the features, caches them to disk, and frees the model. The DINOv2 model is
    only instantiated on a cache miss, so cached runs never pay download/GPU
    cost.
    """
    train_path, test_path = _feature_cache_paths(cache_dir, dataset_key, seed, vit_name)

    n_train = len(train_loader.dataset)
    n_test  = len(test_loader.dataset)

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_features = np.load(train_path)
        test_features  = np.load(test_path)
        if train_features.shape[0] == n_train and test_features.shape[0] == n_test:
            print(f"[features] Loaded cache → {train_path} "
                  f"({train_features.shape}) + {test_path} ({test_features.shape})")
            return train_features, test_features
        print(f"[features] Cache size mismatch "
              f"(train {train_features.shape[0]} vs {n_train}, "
              f"test {test_features.shape[0]} vs {n_test}) — re-extracting.")

    print(f"[features] Cache miss — extracting DINOv2 features for '{dataset_key}' "
          f"(seed={seed}, backbone={vit_name}).")
    extractor = DINOv2Extractor(model_name=vit_name).to(device)
    train_features = extract_image_features(train_loader, extractor, device)
    test_features  = extract_image_features(test_loader,  extractor, device)
    del extractor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    os.makedirs(cache_dir, exist_ok=True)
    np.save(train_path, train_features)
    np.save(test_path,  test_features)
    print(f"[features] Saved cache → {train_path} + {test_path}")

    return train_features, test_features