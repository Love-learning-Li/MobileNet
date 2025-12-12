import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
# torch.cuda.amp.GradScaler(args)已经弃用, 需更换为torch.amp.GradScaler('cuda', args)
import torch.amp
from torch.amp import GradScaler  # 导入 AMP 工具

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
    # 根据compute_topk的输入决定是算top-1acc还是top-5acc
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


def train_epoch(model, loader, criterion, optimizer, device, scaler): # 新增 scaler 参数
    model.train()
    total_loss = 0
    correct = 0
    top1_correct = 0
    top5_correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        scaler = GradScaler()
        
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # 使用 scaler 进行反向传播和优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    # start_time = time.time()
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
    # end_time = time.time()
    # once_delay_time = end_time - start_time
    return (
        total_loss / len(loader),
        100. * correct / total,
        100. * top1_correct / total,
        100. * top5_correct / total, 
        # once_delay_time
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
    # return factory(image_size=cfg.dataset.image_size, **cfg.model_kwargs)


# ----------------------------
# 3. Main Training Loop
# ----------------------------
def main():
    cfg = get_config()
    logger = setup_logger(log_dir = cfg.log_dir)
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
    
    # 定义断点检查点路径 (固定名称以便查找)
    checkpoint_path = save_dir / f"{cfg.experiment_name}_checkpoint.pth"

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
    # optimizer = torch.optim.AdamW(
    #         model.parameters(),
    #         lr=lr,
    #         weight_decay=cfg.optimizer.weight_decay
    #     )
    # optimizer = torch.optim.SGD(
    #         model.parameters(), 
    #         lr=lr, 
    #         momentum=cfg.optimizer.momentum, 
    #         weight_decay=cfg.optimizer.weight_decay,
    #         nesterov=cfg.optimizer.nesterov
    #     )
    #------------------------------------------------------------
    #默认是SGD
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

    # 初始化 GradScaler
    scaler = GradScaler(device="cuda")

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
    
    # ----------------------------
    # 断点续训逻辑
    # ----------------------------
    start_epoch = 0
    best_top1k = 0.0
    train_acc_list = []
    test_acc_list = []
    
    if checkpoint_path.exists():
        logger.info(f"🔄 发现断点检查点: {checkpoint_path}，正在恢复训练...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_top1k = checkpoint['best_top1k']
            train_acc_list = checkpoint.get('train_acc_list', [])
            test_acc_list = checkpoint.get('test_acc_list', [])
            logger.info(f"✅ 成功恢复至 Epoch {start_epoch}，当前最佳 Top-1: {best_top1k:.2f}%")
        except Exception as e:
            logger.error(f"❌ 加载检查点失败: {e}，将重新开始训练")
            start_epoch = 0
    else:
        logger.info("🚀 开始新的训练...")
    
    logger.info("=" * 80)
    
    # Training
    total_train_time = 0.0

    # 开启交互模式
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        train_start = time.time()
        # 传入 scaler
        train_loss, train_acc, train_top1, train_top5 = train_epoch(model, trainloader, criterion, optimizer, device, scaler)
        train_duration = time.time() - train_start

        eval_start = time.time()
        test_loss, test_acc, test_top1, test_top5 = evaluate(model, testloader, criterion, device)
        eval_duration = time.time() - eval_start

        scheduler.step()
        epoch_duration = time.time() - epoch_start
        total_train_time += epoch_duration

        current_lr = optimizer.param_groups[0]['lr']
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        logger.info(
            f"\nEpoch [{epoch + 1}/{epochs}] | LR: {current_lr:.6f}\n"
            f"----------------------------------------------------------------\n"
            f"| Metric | {'Train':<10} | {'Test':<10} |\n"
            f"|--------|------------|------------|\n"
            f"| Loss   | {train_loss:<10.4f} | {test_loss:<10.4f} |\n"
            f"| Acc    | {train_acc:<10.2f}% | {test_acc:<10.2f}% |\n"
            f"| Top-1  | {train_top1:<10.2f}% | {test_top1:<10.2f}% |\n"
            f"| Top-5  | {train_top5:<10.2f}% | {test_top5:<10.2f}% |\n"
            f"----------------------------------------------------------------\n"
            f"Time: Train {train_duration:.1f}s | Eval {eval_duration:.1f}s | Total {epoch_duration:.1f}s | Accum {total_train_time/60:.1f}m"
        )

        if test_top1 > best_top1k:
            best_top1k = test_top1
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ New best Top-1 accuracy: {best_top1k:.2f}% — model saved!")

        # 保存断点 (每个 epoch 保存一次，确保数据安全，应对每3-4个epoch的中断)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_top1k': best_top1k,
            'train_acc_list': train_acc_list,
            'test_acc_list': test_acc_list,
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"🔖 断点保存成功: {checkpoint_path}")

        # 实时更新绘图
        ax.clear()
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.plot(train_acc_list, label="Train Acc")
        ax.plot(test_acc_list, label="Test Acc")
        ax.legend()
        plt.pause(0.1)

    # 关闭交互模式并显示最终结果
    plt.ioff()
    plt.show()

    logger.info("=" * 80)
    logger.info(f"🎉 Training finished. Best test Top-1 accuracy: {best_top1k:.2f}%")

if __name__ == "__main__":
    main()
