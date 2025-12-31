"""PyTorch Dataset и Lightning DataModule для WikiArt."""

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
    """Dataset PyTorch для изображений WikiArt."""

    def __init__(
        self,
        samples_file: Path,
        raw_dir: Path,
        split: Literal["train", "val", "test"],
        image_size: tuple[int, int] = (224, 224),
        is_train: bool = False,
    ) -> None:
        """Инициализация датасета WikiArt.

        Args:
            samples_file: Путь к файлу метаданных samples.json
            raw_dir: Корневая директория с поддиректориями train/val/test
            split: Какой split использовать
            image_size: Целевой размер изображения (высота, ширина)
            is_train: Это ли обучающие данные (включает аугментацию)
        """
        self.raw_dir = Path(raw_dir)
        self.split = split
        self.image_size = image_size
        self.is_train = is_train

        # Загружаем метаданные
        with samples_file.open(encoding="utf-8") as f:
            all_samples = json.load(f)

        # Фильтруем samples для этого split
        self.samples = [s for s in all_samples if s["split"] == split]

        # Строим отображение метка -> индекс
        unique_labels = sorted({s["label"] for s in self.samples})
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)

        # Настраиваем трансформации
        if is_train:
            # Базовая аугментация: RandomCrop + RandomFlip
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
            # Валидация/тест: только изменение размера и нормализация
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
        """Возвращает размер датасета."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Получает изображение и метку по индексу.

        Args:
            idx: Индекс образца

        Returns:
            Кортеж (тензор изображения, индекс метки)
        """
        sample = self.samples[idx]
        img_path = self.raw_dir / sample["path"]

        # Загружаем и преобразуем изображение
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        # Преобразуем метку из строки в индекс
        label_str = sample["label"]
        label_idx = self.label_to_idx[label_str]

        return img_tensor, label_idx

    def get_class_distribution(self) -> dict[str, int]:
        """Получает распределение образцов по классам."""
        distribution: dict[str, int] = {}
        for sample in self.samples:
            label = sample["label"]
            distribution[label] = distribution.get(label, 0) + 1
        return distribution


class WikiArtDataModule(pl.LightningDataModule):
    """Lightning DataModule для датасета WikiArt."""

    def __init__(self, config: DataConfig) -> None:
        """Инициализирует DataModule.

        Args:
            config: Конфигурация данных
        """
        super().__init__()
        self.config = config
        self.raw_dir = Path(config.raw_dir)
        self.samples_file = self.raw_dir / "splits" / "samples.json"

        # Нормализуем image_size из конфига в типизированный кортеж[int, int].
        # Рассматриваем входное значение как нетипизированный объект, чтобы mypy не
        # отмечал проверки isinstance как недостижимые, когда аннотации конфига
        # отличаются во время выполнения (например, OmegaConf списки).
        from typing import Any

        image_size_raw: Any = config.image_size
        if isinstance(image_size_raw, (list, tuple)) and len(image_size_raw) == 2:
            self.image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
        else:
            self.image_size = (224, 224)

        # Добавляем train_subset_ratio с значением по умолчанию если отсутствует
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
