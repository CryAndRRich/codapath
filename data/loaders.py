import os
from typing import List, Tuple

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


class NPZDataset(Dataset):
    def __init__(self,
                 npz_path: str,
                 split: str = "train",
                 transform: transforms.Compose = None) -> None:
        data = np.load(npz_path)
        self.split = split
        if split == "train":
            self.train_img = data["train_images"]
            self.val_img = data["val_images"]

            self.train_lbl = data["train_labels"].squeeze()
            self.val_lbl = data["val_labels"].squeeze()
            self.lbl = np.concatenate((self.train_lbl, self.val_lbl), axis=0)

            self.len_train = len(self.train_img)
            self.total_len = self.len_train + len(self.val_img)
        elif split == "test":
            self.img = data["test_images"]
            self.lbl = data["test_labels"].squeeze()
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

        return Image.fromarray(img).convert("RGB")

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

def get_data_loaders(data_path: str,
                     seed: int,
                     verbose: bool = False) -> Tuple[DataLoader, DataLoader, List[str]]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if data_path.endswith(".npz"):
        train_dataset = NPZDataset(npz_path=data_path, split="train", transform=transform)
        test_dataset = NPZDataset(npz_path=data_path, split="test", transform=transform)
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
        print(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, test_loader, class_names
