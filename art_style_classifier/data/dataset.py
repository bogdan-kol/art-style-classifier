"""PyTorch Dataset and Lightning DataModule for WikiArt."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from art_style_classifier.config import DataConfig


class WikiArtDataset(Dataset):
    """PyTorch Dataset for WikiArt images."""

    def __init__(
        self,
        samples_file: Path,
        raw_dir: Path,
        split: Literal["train", "val", "test"],
        image_size: tuple[int, int] = (224, 224),
        is_train: bool = False,
    ) -> None:
        """Initialize WikiArt dataset.

        Args:
            samples_file: Path to samples.json metadata file
            raw_dir: Root directory with train/val/test subdirectories
            split: Which split to use
            image_size: Target image size (height, width)
            is_train: Whether this is training data (enables augmentation)
        """
        self.raw_dir = Path(raw_dir)
        self.split = split
        self.image_size = image_size
        self.is_train = is_train

        # Load metadata
        with samples_file.open(encoding="utf-8") as f:
            all_samples = json.load(f)

        # Filter samples for this split
        self.samples = [s for s in all_samples if s["split"] == split]

        # Build label to index mapping
        unique_labels = sorted({s["label"] for s in self.samples})
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)

        # Setup transforms
        if is_train:
            # Baseline augmentation: RandomCrop + RandomFlip
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            # Validation/test: only resize and normalize
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get image and label by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image tensor, label index)
        """
        sample = self.samples[idx]
        img_path = self.raw_dir / sample["path"]

        # Load and transform image
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        # Convert label string to index
        label_str = sample["label"]
        label_idx = self.label_to_idx[label_str]

        return img_tensor, label_idx

    def get_class_distribution(self) -> dict[str, int]:
        """Get distribution of samples per class."""
        distribution: dict[str, int] = {}
        for sample in self.samples:
            label = sample["label"]
            distribution[label] = distribution.get(label, 0) + 1
        return distribution


class WikiArtDataModule(pl.LightningDataModule):
    """Lightning DataModule for WikiArt dataset."""

    def __init__(self, config: DataConfig) -> None:
        """Initialize DataModule.

        Args:
            config: Data configuration
        """
        super().__init__()
        self.config = config
        self.raw_dir = Path(config.raw_dir)
        self.samples_file = self.raw_dir / "splits" / "samples.json"

        # Normalize image_size from config into a well-typed tuple[int, int].
        # Treat the incoming value as an untyped object so mypy doesn't
        # mark the `isinstance` checks as unreachable when config
        # annotations differ at runtime (e.g., OmegaConf lists).
        from typing import Any

        image_size_raw: Any = config.image_size
        if isinstance(image_size_raw, (list, tuple)) and len(image_size_raw) == 2:
            self.image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
        else:
            self.image_size = (224, 224)

        # Add train_subset_ratio with default if not present
        self.train_subset_ratio = getattr(config, "train_subset_ratio", 1.0)

        self.train_dataset: WikiArtDataset | Subset | None = None
        self.val_dataset: WikiArtDataset | None = None
        self.test_dataset: WikiArtDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for train/val/test.

        Args:
            stage: 'fit', 'validate', 'test', or None
        """
        if stage in ("fit", None):
            train_dataset_full = WikiArtDataset(
                samples_file=self.samples_file,
                raw_dir=self.raw_dir,
                split="train",
                image_size=self.image_size,
                is_train=True,
            )

            # Optionally limit training dataset size for quick testing
            if self.train_subset_ratio < 1.0:
                total_size = len(train_dataset_full)
                subset_size = int(total_size * self.train_subset_ratio)
                indices = list(range(total_size))
                random.seed(self.config.random_seed)
                random.shuffle(indices)
                subset_indices = indices[:subset_size]
                self.train_dataset = Subset(train_dataset_full, subset_indices)
                print(
                    f"Using {subset_size}/{total_size} ({self.train_subset_ratio*100:.1f}%) of training samples"
                )
            else:
                self.train_dataset = train_dataset_full

            self.val_dataset = WikiArtDataset(
                samples_file=self.samples_file,
                raw_dir=self.raw_dir,
                split="val",
                image_size=self.image_size,
                is_train=False,
            )

        if stage in ("test", None):
            self.test_dataset = WikiArtDataset(
                samples_file=self.samples_file,
                raw_dir=self.raw_dir,
                split="test",
                image_size=self.image_size,
                is_train=False,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            self.setup("fit")

        assert self.train_dataset is not None, "Train dataset not initialized"

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            self.setup("fit")

        assert self.val_dataset is not None, "Validation dataset not initialized"

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Create test dataloader."""
        if self.test_dataset is None:
            self.setup("test")

        assert self.test_dataset is not None, "Test dataset not initialized"

        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
