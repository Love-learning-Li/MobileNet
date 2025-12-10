import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Models.MobileViT.MobileViT import mobilevit_s, mobilevit_xs, mobilevit_xxs
from configs.train_config import TrainingConfig, get_config
# from Models.MobileNet.MobileNet4CIFAR100 import MobileNetV1
# from Models.MobileNetV2.MobileNetV2 import MobileNetV2
# from Models.MobileNetV3.MobileNetV3 import MobileNetV3
# from Models.MobileNetV4.MobileNetV4 import MobileNetV4
# from Models.MobileNet.MobileNet4ImageNet100 import MobileNetV1_4ImageNet100
import time

# ----------------------------
# 0. 配置日志
# ----------------------------
def setup_logger(log_dir="logs"):
    """设置日志记录器"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"image100_training_{timestamp}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志保存到: {log_file}")
    return logger

# ----------------------------
# 1. CIAFR-100数据集加载
# ----------------------------
def get_cifar100_loaders(
    batch_size,
    data_path="G:/0_Python/Pytorch_learning/MobileNet/data/cifar-10-batches-py",
    num_workers=8,
    pin_memory=True,
):

    data_path = Path(data_path)
    CIFAR100_TRAIN_MEAN = (0.507075, 0.486548, 0.440917)
    CIFAR100_TRAIN_STD  = (0.267334, 0.256438, 0.276150)

    # 训练数据增强
    transform_train = transforms.Compose([
        # transforms.RandomResizedCrop(32, scale=(0.72, 1.0), ratio=(0.9, 1.1), padding=4),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD) 
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ])

    train_dataset = torchvision.datasets.CIFAR100(root=str(data_path),
                                                  train=True,
                                                  download=True, transform=transform_train)
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


# ----------------------------
# 2. Training & Evaluation
# ----------------------------
# ----------------------------
# 2. Training & Evaluation
# ----------------------------
def compute_topk(outputs, targets, topk=(1, 5)):
    max_k = min(max(topk), outputs.size(1))
    _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        k = min(k, outputs.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.item())
    return res


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    top1_correct = 0
    top5_correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        top1, top5 = compute_topk(outputs, targets, topk=(1, 5))
        top1_correct += top1
        top5_correct += top5
    return (
        total_loss / len(loader),
        100. * correct / total,
        100. * top1_correct / total,
        100. * top5_correct / total
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    top1_correct = 0
    top5_correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            top1, top5 = compute_topk(outputs, targets, topk=(1, 5))
            top1_correct += top1
            top5_correct += top5
    end_time = time.time()
    once_delay_time = end_time - start_time
    return (
        total_loss / len(loader),
        100. * correct / total,
        100. * top1_correct / total,
        100. * top5_correct / total, 
        once_delay_time
    )

# ----------------------------
# Builder helpers
# ----------------------------
DATASET_BUILDERS: dict[str, Callable[..., Tuple[DataLoader, DataLoader]]] = {
    "cifar100": get_cifar100_loaders,
}


MODEL_FACTORY: dict[str, Callable[..., nn.Module]] = {
    "mobilevit_xxs": mobilevit_xxs,
    "mobilevit_xs": mobilevit_xs,
    "mobilevit_s": mobilevit_s,
}


def build_dataloaders(cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    if cfg.dataset.name not in DATASET_BUILDERS:
        available = ", ".join(DATASET_BUILDERS)
        raise ValueError(f"Dataset '{cfg.dataset.name}' is not supported. Available: {available}")
    builder = DATASET_BUILDERS[cfg.dataset.name]
    return builder(
        batch_size=cfg.dataset.batch_size,
        data_path=cfg.dataset.data_path,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )


def build_model(cfg: TrainingConfig) -> nn.Module:
    if cfg.model_name not in MODEL_FACTORY:
        available = ", ".join(MODEL_FACTORY)
        raise ValueError(f"Model '{cfg.model_name}' is not supported. Available: {available}")
    factory = MODEL_FACTORY[cfg.model_name]
    try:
        return factory(**cfg.model_kwargs)
    except TypeError:
        if cfg.model_kwargs:
            raise
        return factory()


# ----------------------------
# 3. Main Training Loop
# ----------------------------
def main():
    cfg = get_config()
    logger = setup_logger(log_dir=cfg.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loaded config: {cfg.experiment_name}")

    # ============================================================= #
    # Hyperparameters
    batch_size = cfg.dataset.batch_size
    epochs = cfg.epochs
    lr = cfg.optimizer.lr
    warmup_epochs = min(cfg.scheduler.warmup_epochs, max(0, epochs - 1))
    # ============================================================= #

    # 创建保存目录（由配置提供）
    save_dir = cfg.weights_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 根据实验名生成权重文件名
    timestamp = datetime.now().strftime("%m_%d_%H%M%S")
    save_path = save_dir / f"{cfg.experiment_name}_{timestamp}.pth"

    # Data
    # 记录数据路径
    logger.info(f"使用数据路径: {cfg.dataset.data_path}")
    trainloader, testloader = build_dataloaders(cfg)

    # model = MobileNetV2(num_classes=100, width_mult=0.75).to(device)
    # model = MobileNetV3(num_classes=100).to(device)
    # model = MobileNetV4(num_classes=100).to(device).to(device)
    model = build_model(cfg).to(device)
    
    logger.info(f"模型结构:\n{model}")
    logger.info(f"超参数设置:\nBatch Size: {batch_size}, Epochs: {epochs}, Learning Rate: {lr}, Warmup Epochs: {warmup_epochs}")

    # Loss & Optimizer 
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    
    # 定义优化器 
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=cfg.optimizer.momentum, 
        weight_decay=cfg.optimizer.weight_decay,
        nesterov=cfg.optimizer.nesterov
    )

    # warmup学习率调度器
    scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=cfg.scheduler.start_factor,
                    total_iters=warmup_epochs
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, epochs - warmup_epochs)
                )
            ],
            milestones=[warmup_epochs]
        )
    
    logger.info("开始训练...")
    logger.info("=" * 80)
    
    

    # Training
    best_top1k = 0.0
    avarge_delay_time = 0.0
    fast_delay_time = 999.9
    slow_delay_time = 0.0
    once_delay_time = 0.0
    all_delay_time = 0.0
    train_acc_list = []
    test_acc_list = []
    for epoch in range(epochs):
        train_loss, train_acc, train_top1, train_top5 = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc, test_top1, test_top5, once_delay_time = evaluate(model, testloader, criterion, device)
        all_delay_time += once_delay_time
        avarge_delay_time = all_delay_time / (epoch + 1)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        logger.info(f"Epoch [{epoch + 1}/{epochs}] LR: {current_lr:.6f} | "
                   f"Train Loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Top-1: {train_top1:.2f}%, Top-5: {train_top5:.2f}% | \n"
                   f"Test Loss: {test_loss:.4f},Test acc: {test_acc:.4f}, Top-1: {test_top1:.2f}%, Top-5: {test_top5:.2f}% | \n"
                   f"Epoch [{epoch + 1}/{epochs}] , Test Once Delay: {once_delay_time:.4f}s, Avarge Delay: {avarge_delay_time:.4f}s | ")

        if test_top1 > best_top1k:
            best_top1k = test_top1
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ New best Top-1 accuracy: {best_top1k:.2f}% — model saved!")

        if once_delay_time < fast_delay_time:
            fast_delay_time = once_delay_time
        
        if once_delay_time > slow_delay_time:
            slow_delay_time = once_delay_time
    # 绘制损失曲线
    plt.plot(test_acc_list)
    plt.xlabel("epoch")
    plt.ylabel("test_accuracy")
    plt.show()

    plt.plot(train_acc_list)
    plt.xlabel("epoch")
    plt.ylabel("test_accuracy")
    plt.show()

    logger.info("=" * 80)
    logger.info(f"🎉 Training finished. Best test Top-1 accuracy: {best_top1k:.2f}%")

if __name__ == "__main__":
    main()
