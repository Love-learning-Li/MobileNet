import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader

from configs.train_config import TrainingConfig, get_config, MODEL_REGISTRY, DATASET_REGISTRY

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
def build_dataloaders(cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    if cfg.dataset.name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY)
        raise ValueError(f"Dataset '{cfg.dataset.name}' is not supported. Available: {available}")
    builder = DATASET_REGISTRY[cfg.dataset.name]
    return builder(
        batch_size=cfg.dataset.batch_size,
        data_path=cfg.dataset.data_path,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        image_size=cfg.dataset.image_size,
    )


def build_model(cfg: TrainingConfig) -> nn.Module:
    if cfg.model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Model '{cfg.model_name}' is not supported. Available: {available}")
    factory = MODEL_REGISTRY[cfg.model_name]
        return factory()
    # try:
    #     return factory(image_size=cfg.dataset.image_size, **cfg.model_kwargs)
    # except TypeError:
    #     if cfg.model_kwargs:
    #         raise
    #     return factory(image_size=cfg.dataset.image_size)


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

    model = build_model(cfg).to(device)
    
    logger.info(f"模型结构:\n{model}")
    logger.info(f"超参数设置:\nBatch Size: {batch_size}, Epochs: {epochs}, Learning Rate: {lr}, Warmup Epochs: {warmup_epochs}")

    # Loss & Optimizer 
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    
    # 定义优化器 
    if cfg.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    else:
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
    # avarge_delay_time = 0.0
    # fast_delay_time = 999.9
    # slow_delay_time = 0.0
    # once_delay_time = 0.0
    # all_delay_time = 0.0
    train_acc_list = []
    test_acc_list = []
    total_train_time = 0.0
    for epoch in range(epochs):
        epoch_start = time.time()

        train_start = time.time()
        train_loss, train_acc, train_top1, train_top5 = train_epoch(model, trainloader, criterion, optimizer, device)
        train_duration = time.time() - train_start

        eval_start = time.time()
        test_loss, test_acc, test_top1, test_top5, once_delay_time = evaluate(model, testloader, criterion, device)
        eval_duration = time.time() - eval_start

        scheduler.step()
        epoch_duration = time.time() - epoch_start
        total_train_time += epoch_duration

        current_lr = optimizer.param_groups[0]['lr']
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        logger.info(f"Epoch [{epoch + 1}/{epochs}] LR: {current_lr:.6f} | "
                   f"Train Loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Top-1: {train_top1:.2f}%, Top-5: {train_top5:.2f}% | \n"
                   f"Test Loss: {test_loss:.4f},Test acc: {test_acc:.4f}, Top-1: {test_top1:.2f}%, Top-5: {test_top5:.2f}% | \n"
                   # f"Epoch [{epoch + 1}/{epochs}] , Test Once Delay: {once_delay_time:.4f}s, Avarge Delay: {avarge_delay_time:.4f}s | "
                   f"Train: {train_duration:.2f}s | Eval: {eval_duration:.2f}s | \n"
                   f"Epoch total: {epoch_duration:.2f}s | \n"
                   f"累积训练时间: {total_train_time/60:.2f}min \n"
                   )

        if test_top1 > best_top1k:
            best_top1k = test_top1
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ New best Top-1 accuracy: {best_top1k:.2f}% — model saved!")

        # if once_delay_time < fast_delay_time:
        #     fast_delay_time = once_delay_time
        
        # if once_delay_time > slow_delay_time:
        #     slow_delay_time = once_delay_time
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
