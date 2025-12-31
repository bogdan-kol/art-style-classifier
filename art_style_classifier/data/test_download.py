"""Test download with small subset."""

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from art_style_classifier.config import DataConfig


def download_test_subset(n_samples: int = 100) -> bool:
    """Загружает маленькую подвыборку для тестирования."""
    cfg = DataConfig()

    print(f"Downloading {n_samples} samples from WikiArt for testing...")

    # Загружаем датасет и берём подвыборку
    dataset = load_dataset("huggan/wikiart", split="train")

    # Берём только первые n_samples для теста
    dataset = dataset.select(range(min(n_samples, len(dataset))))
    num_samples = len(dataset)

    print(f"Loaded {num_samples} samples")

    # Простое разделение
    indices = list(range(num_samples))
    train_idx, temp_idx = train_test_split(
        indices, train_size=cfg.splits["train"], random_state=cfg.random_seed
    )

    val_size = cfg.splits["val"] / (cfg.splits["val"] + cfg.splits["test"])
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size, random_state=cfg.random_seed
    )

    # Сохраняем только метаданные, не сами изображения (для теста)
    raw_dir = Path(cfg.raw_dir) / "test_subset"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем информацию о датасете
    dataset_info = {
        "total_samples": num_samples,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "seed": cfg.random_seed,
        "splits": cfg.splits,
        "note": "Test subset - actual images not saved to disk",
    }

    with open(raw_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    # Сохраняем индексы
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    with open(raw_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    # Собираем информацию о стилях
    styles: dict[str, Any] = {}
    for idx in range(num_samples):
        item = dataset[idx]
        style = str(item["style"])
        styles[style] = styles.get(style, 0) + 1

    with open(raw_dir / "styles.json", "w") as f:
        json.dump(styles, f, indent=2)

    print(f"\\nTest dataset created at: {raw_dir}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Unique styles: {len(styles)}")

    return True
