from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Callable
import copy

# Import Models
from Models.MobileViT.MobileViT import mobilevit_xxs, mobilevit_xs, mobilevit_s

# # Import Dataloaders
# from utils.dataloaders import get_cifar100_loaders

from utils import get_cifar100_loaders

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==============================================================================
# Model & Dataset Registries
# ==============================================================================
MODEL_REGISTRY = {
    "mobilevit_xxs": mobilevit_xxs,
    "mobilevit_xs": mobilevit_xs,
    "mobilevit_s": mobilevit_s,
    # Add new models here
}

DATASET_REGISTRY = {
    "cifar100": get_cifar100_loaders,
    # Add new datasets here
}

@dataclass
class DatasetConfig:
    name: str
    data_path: Path
    batch_size: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    image_size: int = 64


@dataclass
class OptimizerConfig:
    name: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True


@dataclass
class SchedulerConfig:
    warmup_epochs: int
    start_factor: float = 1e-3


@dataclass
class TrainingConfig:
    experiment_name: str
    dataset: DatasetConfig
    epochs: int
    model_name: str
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig("sgd", 0.1, 0.9, 5e-4))
    scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig(5))
    label_smoothing: float = 0.1
    log_dir: Path = PROJECT_ROOT / "logs"
    weights_subdir: Path = Path("Models")

    def weights_dir(self) -> Path:
        return (PROJECT_ROOT / self.weights_subdir).resolve()


CONFIG_REGISTRY: Dict[str, TrainingConfig] = {
    "cifar100_mobilevit_xxs":
      TrainingConfig(
        experiment_name="cifar100_mobilevit_xxs",
        dataset=DatasetConfig(
            name="cifar100",
            data_path=PROJECT_ROOT / "data" ,
            batch_size=1,
            num_workers=2,
            pin_memory=True,
            image_size=64,
        ),
        epochs=200,
        model_name="mobilevit_xxs",
        # model_kwargs={"num_classes": 100},
        optimizer=OptimizerConfig(
            name="adamw",
            lr=0.002,
            momentum=0.9,
            weight_decay=0.05,
            nesterov=True,
        ),
        scheduler=SchedulerConfig(warmup_epochs=10, start_factor=1e-3),
        label_smoothing=0.1,
        weights_subdir=Path("Models/MobileViT/Pretrained"),
    )
}


def get_config(name: str = "cifar100_mobilevit_xxs") -> TrainingConfig:
    if name not in CONFIG_REGISTRY:
        available = ", ".join(CONFIG_REGISTRY)
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    return copy.deepcopy(CONFIG_REGISTRY[name])
