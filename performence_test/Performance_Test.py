import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import autoaugment
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
# from Attention4CIFAR10 import AttentionCifarClassifier_10
from Resnet4CIFAR100 import resnet18, resnet34, resnet50, resnet101
# from Attention4CIFAR_New import ViT_CIFAR
# from Attention4CIFAR100New import HybridVisionTransformer
import time

# ----------------------------
# 1. CIAFR-10数据集加载
# ----------------------------
CIFAR100_TRAIN_MEAN = (0.507075, 0.486548, 0.440917)
CIFAR100_TRAIN_STD  = (0.267334, 0.256438, 0.276150)

def get_cifar100_loaders(batch_size):

    # transform_test = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    # ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ])

    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)

    # train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
    #                                         download=True, transform=transform_train)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)

    # test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
    #                                        download=True, transform=transform_test)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)

    return test_loader


# ----------------------------
# 2. Evaluation
# ----------------------------

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    end_time = time.time()
    once_delay_time = end_time - start_time
    return 100. * correct / total, once_delay_time


# ----------------------------
# 3. Main Training Loop
# ----------------------------
def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # 临时修改模型为CPU，以验证量化前后都在CPU上模型的性能
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    epochs = 5
    lr = 0.01

    # Data
    testloader = get_cifar100_loaders(batch_size)

    # Model
    model = resnet34().to(device)
    # model = model.cuda()
    model.load_state_dict(torch.load("resnet34_cifar100_quantized_int8.pth"))

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    

    # Training
    # best_acc = 0.0
    # low_acc  = 100.0
    c
    for epoch in range(epochs):
        # train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, scheduler, device)
        test_acc, once_delay_time = evaluate(model, testloader, criterion, device)
        all_delay_time += once_delay_time
        avarge_delay_time = all_delay_time / (epoch + 1)


        # print(f"Epoch [{epoch + 1}/{epochs}] ")
        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Test Once Delay: {once_delay_time:.4f}s, Avarge Delay: {avarge_delay_time:.4f}s | ")
        # print(f"Epoch [{epoch + 1}/{epochs}] "
        #       f"Test Once Delay: {once_delay_time:.4f}s, Avarge Delay: {avarge_delay_time:.4f}s | "
        #       f"Test Acc: {test_acc:.2f}%")

        
        if once_delay_time < fast_delay_time:
            fast_delay_time = once_delay_time
            # torch.save(model.state_dict(), "resnet34_cifar100_best.pth")
            # print(f"✅ New best delay time : {fast_delay_time:.4f}%")
        
        if once_delay_time > slow_delay_time:
            slow_delay_time = once_delay_time
            # torch.save(model.state_dict(), "resnet34_cifar100_best.pth")
            # print(f"✅ New slow delay time: {slow_delay_time:.4f}%")
        
        

    print(f"🎉 Training finished.\n")
    print(f"test accuracy: {test_acc:.2f}%")
    print(f"most fast delay time : {fast_delay_time:.4f}s")
    print(f"most slow delay time: {slow_delay_time:.4f}s")
    print(f"avarge delay time: {avarge_delay_time:.4f}s")



if __name__ == "__main__":
    main()