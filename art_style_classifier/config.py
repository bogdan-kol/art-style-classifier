"""Configuration management with Hydra."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf


@dataclass
class DataConfig:
    """Data configuration."""

    name: str = "wikiart"
    data_dir: str = "data"
    raw_dir: str = "data/raw/wikiart"
    processed_dir: str = "data/processed"

    image_size: tuple[int, int] = (224, 224)

    splits: dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    batch_size: int = 32
    num_workers: int = 4
    random_seed: int = 42
    pin_memory: bool = True
    persistent_workers: bool = True

    # Добавь новое поле для subset
    train_subset_ratio: float = 1.0  # По умолчанию используем все данные


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "art_style_classifier"
    architecture: str = "efficientnet_b0"
    num_classes: int = 27
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.2


@dataclass
class TrainingConfig:
    """Training configuration."""

    max_epochs: int = 5
    learning_rate: float = 0.001
    batch_size: int = 16
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10
    checkpoint_monitor: str = "val_loss"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    experiment_name: str = "art-style-classifier"
    tracker: str = "mlflow"
    tracking_uri: str = "http://localhost:5000"
    log_every_n_steps: int = 50


@dataclass
class Config:
    """Main configuration class."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
        )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "Config":
        """Create Config from Hydra DictConfig."""
        # Convert to dictionary and resolve interpolations
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        return cls.from_dict(cfg_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "logging": self.logging.__dict__,
        }

    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        OmegaConf.save(self.to_dict(), path)

    def __str__(self) -> str:
        """String representation."""
        import json

        return json.dumps(self.to_dict(), indent=2)


def load_config(
    config_path: Optional[str] = None,
    config_name: str = "main",
    overrides: Optional[list[str]] = None,
) -> Config:
    """Load configuration using Hydra.

    Args:
        config_path: Path to configs directory
        config_name: Name of the config file
        overrides: List of configuration overrides

    Returns:
        Config object
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "configs")

    # Initialize Hydra
    with hydra.initialize_config_dir(config_dir=config_path, version_base="1.3"):
        # Compose configuration
        cfg = hydra.compose(config_name=config_name, overrides=overrides or [])

        # Convert to our Config class
        return Config.from_hydra(cfg)


def test_hydra_config() -> None:
    """Test Hydra configuration loading."""
    print("Testing Hydra configuration...")

    # Test 1: Load default config
    config = load_config()
    print("\n1. Default configuration loaded:")
    print(f"   Model: {config.model.name}")
    print(f"   Architecture: {config.model.architecture}")
    print(f"   Batch size: {config.data.batch_size}")
    print(f"   Learning rate: {config.training.learning_rate}")

    # Test 2: Load with overrides
    config2 = load_config(
        overrides=[
            "data.batch_size=64",
            "training.learning_rate=0.01",
            "model.architecture=resnet50",
        ]
    )
    print("\n2. Configuration with overrides:")
    print(f"   Batch size: {config2.data.batch_size}")
    print(f"   Learning rate: {config2.training.learning_rate}")
    print(f"   Architecture: {config2.model.architecture}")

    # Test 3: Save config
    config.save(Path("test_config.yaml"))
    print("\n3. Configuration saved to test_config.yaml")

    print("\n All tests passed!")


if __name__ == "__main__":
    test_hydra_config()
