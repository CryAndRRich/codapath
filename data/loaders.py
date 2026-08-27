import os
from typing import List, Tuple

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


class NPZDataset(Dataset):
    """Image/label arrays from a .npz, memory-mapped when a cache root is given.

    Memory is the reason `mmap_cache_dir` exists. PathMNIST-224 holds ~15 GiB of
    uint8 pixels, and an eager read costs that much RAM *per process*. Two
    extraction workers on a Kaggle T4 x2 session (~30 GiB) then sit exactly on
    the limit and one is OOM-killed inside `np.load` before printing anything.

    Note that `np.load(npz, mmap_mode="r")` does NOT help: numpy ignores
    mmap_mode for a .npz, because members are read through zipfile. Mapping
    requires standalone .npy files, which `data.npz_mmap` exports once.
    """

    def __init__(self,
                 npz_path: str,
                 split: str = "train",
                 transform: transforms.Compose = None,
                 mmap_cache_dir: str = None) -> None:
        from .npz_mmap import open_npz_mmap

        data = open_npz_mmap(npz_path, mmap_cache_dir, verbose=False)
        self.mmap = data is not None
        if data is None:
            data = np.load(npz_path)
        self.split = split
        if split == "train":
            self.train_img = data["train_images"]
            self.val_img = data["val_images"]

            # np.asarray: labels come back as memmaps too, and they are indexed
            # constantly. They are kilobytes, so materialise them once.
            self.train_lbl = np.asarray(data["train_labels"]).squeeze()
            self.val_lbl = np.asarray(data["val_labels"]).squeeze()
            self.lbl = np.concatenate((self.train_lbl, self.val_lbl), axis=0)

            self.len_train = len(self.train_img)
            self.total_len = self.len_train + len(self.val_img)
        elif split == "test":
            self.img = data["test_images"]
            self.lbl = np.asarray(data["test_labels"]).squeeze()
            self.total_len = len(self.img)
        else:
            raise ValueError("Split must be 'train' or 'test'")

        self.transform = transform
        self.classes = [
            "adipose", "background", "debris", "lymphocytes", "mucus",
            "smooth_muscle", "normal_colon_mucosa", "cancer_associated_stroma",
            "colorectal_adenocarcinoma"
        ]

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.get_raw_image(idx)
        label = self.lbl[idx]

        if self.transform:
            img = self.transform(img)
        return img, label

    def get_raw_image(self, idx: int) -> Image.Image:
        """Return an RGB PIL image before any DINO/CellViT transform."""
        if self.split == "train":
            if idx < self.len_train:
                img = self.train_img[idx]
            else:
                val_idx = idx - self.len_train
                img = self.val_img[val_idx]
        else:
            img = self.img[idx]

        # np.asarray materialises the one mapped row. Without it PIL would hold a
        # buffer backed by the memmap for the lifetime of the image.
        return Image.fromarray(np.asarray(img)).convert("RGB")

    def sample_id(self, idx: int) -> str:
        if self.split == "train":
            prefix = "train" if idx < self.len_train else "val"
            source_idx = idx if idx < self.len_train else idx - self.len_train
        else:
            prefix = "test"
            source_idx = idx
        return f"{prefix}:{source_idx}"


class RawRGBDataset(Dataset):
    """Raw-RGB view preserving the exact order of an existing split.

    This view deliberately bypasses the 224x224 ImageNet transform used by
    DINOv2. It is the only dataset view that CellViT extraction should consume.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def _resolve(self, idx: int):
        if isinstance(self.dataset, Subset):
            return self.dataset.dataset, int(self.dataset.indices[idx])
        return self.dataset, idx

    def sample_id(self, idx: int) -> str:
        base, source_idx = self._resolve(idx)
        if isinstance(base, NPZDataset):
            return base.sample_id(source_idx)
        if isinstance(base, ImageFolder):
            path, _ = base.samples[source_idx]
            return os.path.relpath(path, base.root).replace(os.sep, "/")
        raise TypeError(f"Unsupported dataset type for raw view: {type(base)!r}")

    def __getitem__(self, idx: int):
        base, source_idx = self._resolve(idx)
        if isinstance(base, NPZDataset):
            image = base.get_raw_image(source_idx)
            label = int(base.lbl[source_idx])
        elif isinstance(base, ImageFolder):
            path, label = base.samples[source_idx]
            image = base.loader(path).convert("RGB")
        else:
            raise TypeError(f"Unsupported dataset type for raw view: {type(base)!r}")
        return image, int(label), self.sample_id(idx)


def get_sample_ids(dataset: Dataset) -> List[str]:
    raw = RawRGBDataset(dataset)
    return [raw.sample_id(i) for i in range(len(raw))]


class ActiveLearningDataset(Dataset):
    def __init__(self,
                 subset: Dataset,
                 labels: torch.Tensor) -> None:
        self.subset = subset
        self.labels = labels

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, _ = self.subset[idx]
        return img, self.labels[idx]

def default_num_workers(data_path: str) -> int:
    """DataLoader workers to use for `data_path`.

    ImageFolder decodes a JPEG per sample, on the main thread when workers are 0,
    so the GPU waits on CPU rather than the reverse — measurably so for the
    ~130k tiles of skintissue. Two workers overlap that decode with the forward
    pass. A .npz stays at 0: its pixels are already decoded, and with
    `mmap_cache_dir` they are mapped, so a worker process would only add
    per-batch pickling.

    Two, not more: a Kaggle session has 4 vCPUs, and extraction runs one process
    per GPU, so 2 GPU workers x 2 loader workers already fills the machine.
    Oversubscribing makes both GPUs slower.
    """
    return 0 if data_path.endswith(".npz") else 2


def get_data_loaders(data_path: str,
                     seed: int,
                     verbose: bool = False,
                     mmap_cache_dir: str = None,
                     num_workers: int = None) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Build the fixed-order train/test loaders for a dataset path.

    `mmap_cache_dir` applies to .npz datasets only. Pass it when several
    processes load the same dataset at once — two per-GPU extraction workers on
    one Kaggle session — so the pixels are mapped from disk instead of each
    worker holding its own multi-GiB copy. None keeps the eager behaviour, which
    is what a single-process run wants.

    `num_workers` defaults to `default_num_workers(data_path)`. Workers never
    affect *which* samples come out or in what order — both loaders are
    `shuffle=False` over a fixed split — so this is a throughput knob only.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if num_workers is None:
        num_workers = default_num_workers(data_path)

    if data_path.endswith(".npz"):
        train_dataset = NPZDataset(
            npz_path=data_path, split="train", transform=transform,
            mmap_cache_dir=mmap_cache_dir,
        )
        test_dataset = NPZDataset(
            npz_path=data_path, split="test", transform=transform,
            mmap_cache_dir=mmap_cache_dir,
        )
        class_names = train_dataset.classes
    else:
        full_dataset = ImageFolder(root=data_path, transform=transform)
        class_names = full_dataset.classes

        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        test_size = total_size - train_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=generator
        )

    if verbose:
        print(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)} "
              f"| num_workers={num_workers}")

    # persistent_workers keeps the pool alive between the two passes a loader is
    # iterated for, so the process-startup cost is paid once instead of per pass.
    # It is only valid with num_workers > 0.
    loader_kwargs = dict(batch_size=256, shuffle=False, num_workers=num_workers,
                         pin_memory=True)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    return train_loader, test_loader, class_names
