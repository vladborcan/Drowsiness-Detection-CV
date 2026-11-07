"""Dataset, transformari si DataLoader-e pentru UTA-RLDD (cadre extrase)."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.config import CLASS_NAMES, DEFAULT_SEED, LABEL_TO_INDEX, SPLITS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(split: str, image_size: int = 128) -> T.Compose:
    """Transformari pe split.

    train : augmentari usoare, aplicate doar pe antrenare (variatii de pozitie,
            lumina, orientare). Fara flip vertical; flip orizontal e acceptabil
            pentru fata.
    val/test: doar redimensionare + normalizare (deterministic).
    """
    if split == "train":
        ops = [
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(5),
            T.ColorJitter(brightness=0.3, contrast=0.3),
        ]
    else:
        ops = [T.Resize((image_size, image_size))]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


class DrowsinessDataset(Dataset):
    """Cadre (imagini RGB) + eticheta de clasa {0: alert, 1: low_vigilance, 2: drowsy}.

    Args:
        frames: DataFrame sau calea catre splits.csv (coloane: path, label, split, ...).
        root: folderul care contine imaginile (caile din tabel sunt relative la el).
        split: "train" | "val" | "test".
        transform: transformari aplicate imaginii PIL.
    """

    def __init__(self, frames, root, split: str, transform=None):
        if split not in SPLITS:
            raise ValueError(f"split trebuie sa fie unul din {SPLITS}, nu '{split}'")
        df = pd.read_csv(frames) if isinstance(frames, (str, Path)) else frames
        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"Split-ul '{split}' nu contine niciun cadru.")
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.paths = self.df["path"].tolist()
        self.targets = [LABEL_TO_INDEX[int(l)] for l in self.df["label"]]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.root / self.paths[i]) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]

    def class_counts(self) -> dict[str, int]:
        counts = np.bincount(self.targets, minlength=len(CLASS_NAMES))
        return dict(zip(CLASS_NAMES, counts.tolist()))


def _seed_worker(worker_id: int) -> None:
    """Seed pentru numpy/random in fiecare worker (reproductibilitate)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_dataloaders(
    splits_csv: str | Path,
    data_root: str | Path,
    image_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = DEFAULT_SEED,
) -> dict[str, DataLoader]:
    """Creeaza DataLoader-ele train / val / test.

    Doar train e amestecat (shuffle), cu un generator seedat, ca ordinea sa fie
    reproductibila. Pe Windows, num_workers > 0 cere ca scriptul apelant sa aiba
    `if __name__ == "__main__":`.
    """
    df = pd.read_csv(splits_csv)
    generator = torch.Generator().manual_seed(seed)
    loaders = {}
    for split in SPLITS:
        ds = DrowsinessDataset(df, data_root, split, build_transforms(split, image_size))
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            generator=generator if split == "train" else None,
            persistent_workers=num_workers > 0,
        )
    return loaders