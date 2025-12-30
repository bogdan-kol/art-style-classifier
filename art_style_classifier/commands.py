"""Main CLI commands for the art style classifier."""

import json
import sys
from pathlib import Path
from typing import Optional

import fire
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from art_style_classifier.config import Config, load_config
from art_style_classifier.data.dataset import WikiArtDataModule
from art_style_classifier.models.advanced import AdvancedModel
from art_style_classifier.models.baseline import BaselineModel
from art_style_classifier.utils.git import get_git_commit_id


def _train_impl(cfg: DictConfig) -> None:
    """Internal training implementation with Hydra.

    Args:
        cfg: Hydra configuration
    """
    # Convert to our Config class
    config = Config.from_hydra(cfg)

    # Set global seeds for reproducibility
    try:
        pl.seed_everything(config.data.random_seed, workers=True)
    except TypeError:
        # Older PL versions may not accept workers kwarg
        pl.seed_everything(config.data.random_seed)

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

    # Get git commit ID for reproducibility
    git_commit_id = get_git_commit_id()
    print(f" Git commit: {git_commit_id}")

    # Setup data module
    print("\nSetting up data module...")
    data_module = WikiArtDataModule(config.data)
    data_module.setup("fit")

    # Create model
    arch = config.model.architecture
    if arch == "resnet18" or arch.startswith("baseline"):
        print("\nCreating baseline ResNet18 model...")
        model = BaselineModel(
            num_classes=config.model.num_classes,
            learning_rate=config.training.learning_rate,
            freeze_backbone=True,  # Baseline: freeze backbone
        )
    elif arch.startswith("efficientnet") or arch.startswith("efficient"):
        print("\nCreating Advanced EfficientNet model...")
        # Use AdvancedModel for EfficientNet family
        model = AdvancedModel(
            num_classes=config.model.num_classes,
            learning_rate=config.training.learning_rate,
            weight_decay=getattr(config.training, "weight_decay", 1e-2),
            dropout_rate=config.model.dropout_rate,
            pretrained=config.model.pretrained,
            unfreeze_backbone=bool(getattr(config.model, "unfreeze_backbone", False)),
            lr_scheduler=getattr(config.training, "lr_scheduler", "ReduceLROnPlateau"),
            lr_factor=getattr(config.training, "lr_factor", 0.1),
            lr_patience=getattr(config.training, "lr_patience", 3),
        )
    else:
        raise ValueError(
            f"Model architecture '{config.model.architecture}' not implemented yet."
        )

    # Setup MLflow logger
    print("\nSetting up MLflow logging...")
    print(f" Tracking URI: {config.logging.tracking_uri}")
    mlflow_logger = MLFlowLogger(
        experiment_name=config.logging.experiment_name,
        tracking_uri=config.logging.tracking_uri,
    )

    # Log hyperparameters and git commit
    hyperparams = {
        "model_architecture": config.model.architecture,
        "num_classes": config.model.num_classes,
        "batch_size": config.data.batch_size,
        "learning_rate": config.training.learning_rate,
        "weight_decay": getattr(config.training, "weight_decay", None),
        "max_epochs": config.training.max_epochs,
        "image_size": config.data.image_size,
        "random_seed": config.data.random_seed,
        "train_subset_ratio": getattr(config.data, "train_subset_ratio", 1.0),
        "git_commit_id": git_commit_id,
    }
    mlflow_logger.log_hyperparams(hyperparams)

    # Setup callbacks
    # Create output directory (Hydra would do this automatically with decorator)
    from datetime import datetime

    output_base = Path("outputs")
    output_dir = (
        output_base
        / datetime.now().strftime("%Y-%m-%d")
        / datetime.now().strftime("%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        monitor=config.training.checkpoint_monitor,
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.3f}",
    )
    early_stop_callback = EarlyStopping(
        monitor=config.training.checkpoint_monitor,
        patience=config.training.early_stopping_patience,
        mode="min",
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=config.logging.log_every_n_steps,
        gradient_clip_val=config.training.gradient_clip_val,
        accelerator="auto",
        devices="auto",
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.fit(model, data_module)

    # Save config for reproducibility
    config.save(output_dir / "config.yaml")
    print(f"\nConfiguration saved to: {output_dir / 'config.yaml'}")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")


def train(*args: str) -> None:
    """Train the model with Hydra configuration (Fire-compatible wrapper).

    Args:
        *args: Hydra overrides (e.g., model=baseline, training.max_epochs=5)
               Use dot notation for nested configs: training.max_epochs=5

    Examples:
        train model=baseline
        train model=baseline training.max_epochs=5
    """
    # Fire passes arguments after function name as *args
    # Extract Hydra overrides from command line arguments
    # Skip 'train' command name and module path
    overrides = []
    for arg in args:
        if "=" in arg:
            overrides.append(arg)

    # Also check sys.argv for any remaining overrides (in case fire didn't capture them)
    # This handles cases like: python -m art_style_classifier.commands train model=baseline
    if not overrides:
        # Try to extract from sys.argv
        for arg in sys.argv:
            if "=" in arg and arg not in ["train", "art_style_classifier.commands"]:
                overrides.append(arg)

    # Use Hydra compose API instead of decorator (for Fire compatibility)
    config_path = Path(__file__).parent.parent / "configs"

    # Initialize Hydra and compose config
    with hydra.initialize_config_dir(config_dir=str(config_path), version_base="1.3"):
        cfg = hydra.compose(config_name="main", overrides=overrides)
        _train_impl(cfg)


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
