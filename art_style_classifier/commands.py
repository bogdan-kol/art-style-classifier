"""Main CLI commands for the art style classifier."""

import json
from pathlib import Path
from typing import Optional

import fire
import hydra
from omegaconf import DictConfig

from art_style_classifier.config import Config, load_config


@hydra.main(  # type: ignore[misc]
    config_path="../configs", config_name="main", version_base="1.3"
)
def train(cfg: DictConfig, max_samples: Optional[int] = None) -> None:
    """Hydra-powered training entrypoint (internal).

    Args:
        cfg: Hydra configuration
    """
    # Convert to our Config class
    config = Config.from_hydra(cfg)

    print("=" * 60)
    print("Art Style Classifier - Training")
    print("=" * 60)

    print("\nConfiguration:")
    print(json.dumps(config.to_dict(), indent=2))

    print("\nStarting training...")
    print(f" Model: {config.model.architecture}")
    print(f" Classes: {config.model.num_classes}")
    print(f" Batch size: {config.data.batch_size}")
    print(f" Epochs: {config.training.max_epochs}")
    print(f" Learning rate: {config.training.learning_rate}")

    # TODO: Implement actual training
    print("\n Training pipeline ready!")
    print("   (Actual training implementation will be added in the next step)")

    # Save config for reproducibility
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    config.save(output_dir / "config.yaml")
    print(f"\nConfiguration saved to: {output_dir / 'config.yaml'}")


def predict(
    config_path: Optional[str] = None,
    config_name: str = "main",
    overrides: Optional[list[str]] = None,
) -> None:
    """Make predictions with Hydra configuration."""
    config = load_config(config_path, config_name, overrides)

    print("Making predictions...")
    print(f"Using model: {config.model.name}")
    print(f"Configuration: {config_name}")

    # TODO: Implement prediction
    print("\n Prediction pipeline ready!")


def download_test(n_samples: int = 100) -> None:
    """Download a test subset of WikiArt dataset."""
    from art_style_classifier.data.test_download import download_test_subset

    download_test_subset(n_samples=n_samples)


def download_data(
    use_dvc: bool = True, push: bool = False, max_samples: Optional[int] = None
) -> None:
    """Download WikiArt dataset with optional DVC tracking.

    Args:
        use_dvc: Track data with DVC (default: True)
        push: Push to DVC remote after download (default: False)
        max_samples: Maximum number of samples to download (for testing)
    """
    from art_style_classifier.data.download import download_command

    download_command(use_dvc=use_dvc, push=push, max_samples=max_samples)


def show_config(
    config_path: Optional[str] = None,
    config_name: str = "main",
    overrides: Optional[list[str]] = None,
) -> None:
    """Show current configuration."""
    config = load_config(config_path, config_name, overrides)
    print(json.dumps(config.to_dict(), indent=2))


def config_test() -> None:
    """Test Hydra configuration system."""
    from art_style_classifier.config import test_hydra_config

    test_hydra_config()


def list_configs() -> None:
    """List available configuration files."""
    configs_dir = Path("configs")
    print("Available configuration files:")
    print("=" * 40)

    for yaml_file in sorted(configs_dir.rglob("*.yaml")):
        relative_path = yaml_file.relative_to(configs_dir)
        print(f"   {relative_path}")


if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "predict": predict,
            "download": download_data,  # полная загрузка
            "download_test": download_test,  # тестовая загрузка
            "config": show_config,
            "test": config_test,
            "list": list_configs,
        }
    )
