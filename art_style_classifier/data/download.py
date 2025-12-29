from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from art_style_classifier.config import DataConfig


@dataclass
class SampleInfo:
    """Metadata about a single image sample."""

    path: str
    label: str
    split: Literal["train", "val", "test"]


def _save_image(img: Image.Image, out_path: Path) -> None:
    """Save PIL image to disk, creating parent directories if needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _split_indices(
    n_samples: int, seed: int, train_size: float, val_size: float, test_size: float
) -> tuple[list[int], list[int], list[int]]:
    """Create train/val/test indices with given proportions."""
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1"

    all_indices = list(range(n_samples))

    train_indices, temp_indices = train_test_split(
        all_indices, train_size=train_size, random_state=seed, shuffle=True
    )

    relative_val_size = val_size / (val_size + test_size)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=relative_val_size,
        random_state=seed,
        shuffle=True,
    )

    return train_indices, val_indices, test_indices


def download_command(
    use_dvc: bool = True, push: bool = False, max_samples: int | None = None
) -> None:
    """Download the WikiArt dataset and prepare train/val/test splits.

    Args:
        use_dvc: Reserved for future integration with DVC (not used yet).
        push: Reserved for future DVC push (not used yet).
        max_samples: Maximum number of samples to download (for testing).
    """
    cfg = DataConfig()

    raw_dir = Path(cfg.raw_dir)
    metadata_dir = raw_dir / "splits"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading WikiArt dataset via HuggingFace datasets...")
    print(f" Target directory: {raw_dir}")

    # Загружаем датасет
    dataset = load_dataset("huggan/wikiart", split="train")

    # Если указано максимальное количество samples, ограничиваем
    if max_samples and max_samples < len(dataset):
        print(f" Limiting to {max_samples} samples for testing")
        dataset = dataset.select(range(max_samples))

    num_samples = len(dataset)

    print(f" Total samples: {num_samples}")
    print(" Creating train/val/test splits with fixed seed...")

    train_idx, val_idx, test_idx = _split_indices(
        n_samples=num_samples,
        seed=cfg.random_seed,
        train_size=cfg.splits["train"],
        val_size=cfg.splits["val"],
        test_size=cfg.splits["test"],
    )

    index_to_split: dict[int, Literal["train", "val", "test"]] = {}
    for i in train_idx:
        index_to_split[i] = "train"
    for i in val_idx:
        index_to_split[i] = "val"
    for i in test_idx:
        index_to_split[i] = "test"

    samples: list[SampleInfo] = []

    print(" Saving images and building metadata...")

    # Сначала соберём уникальные стили для создания папок
    unique_styles = set()
    for idx in tqdm(range(num_samples), desc="Collecting styles"):
        item = dataset[idx]
        style = str(item["style"])  # Преобразуем в строку
        unique_styles.add(style)

    # Создаём папки для каждого стиля в каждой split директории
    for split_name in ["train", "val", "test"]:
        for style in unique_styles:
            (raw_dir / split_name / style).mkdir(parents=True, exist_ok=True)

    # Теперь сохраняем изображения
    for idx in tqdm(range(num_samples), desc="Saving images"):
        item = dataset[idx]
        img: Image.Image = item["image"]
        style = str(item["style"])  # Преобразуем в строку
        split = index_to_split[idx]

        out_path = raw_dir / split / style / f"{idx}.jpg"
        _save_image(img, out_path)

        samples.append(
            SampleInfo(
                path=str(out_path.relative_to(raw_dir)),
                label=style,
                split=split,
            )
        )

    all_metadata_path = metadata_dir / "samples.json"
    with all_metadata_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in samples], f, ensure_ascii=False, indent=2)

    splits_indices = {
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
        "seed": cfg.random_seed,
        "splits": cfg.splits,
    }
    splits_path = metadata_dir / "indices.json"
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(splits_indices, f, ensure_ascii=False, indent=2)

    print("\nDownload and preprocessing completed.")
    print(f" Raw data directory: {raw_dir}")
    print(f" Metadata (samples, indices): {metadata_dir}")

    # Выводим статистику
    style_counts: dict[str, int] = {}
    for sample in samples:
        style_counts[sample.label] = style_counts.get(sample.label, 0) + 1

    print("\nStyle distribution:")
    for style, count in sorted(style_counts.items())[
        :10
    ]:  # Показываем только первые 10
        print(f"  {style}: {count} samples")

    if len(style_counts) > 10:
        print(f"  ... and {len(style_counts) - 10} more styles")

    print(f"\nTotal styles: {len(style_counts)}")
    print(f"Total samples: {num_samples}")
