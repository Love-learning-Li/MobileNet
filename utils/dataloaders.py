from pathlib import Path
from typing import Tuple
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar100_loaders(
    batch_size,
    data_path,
    num_workers=8,
    pin_memory=True,
    image_size=32,
):
    data_path = Path(data_path)
    CIFAR100_TRAIN_MEAN = (0.507075, 0.486548, 0.440917)
    CIFAR100_TRAIN_STD  = (0.267334, 0.256438, 0.276150)

    # 训练数据增强
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
        transforms.RandomErasing(p=0.25)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=str(data_path),
        train=True,
        download=True,
        transform=transform_train,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_dataset = torchvision.datasets.CIFAR100(root=str(data_path),
                                                train=False,
                                                download=True, transform=transform_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader

DATASET_REGISTRY = {
    "cifar100": get_cifar100_loaders,
}
