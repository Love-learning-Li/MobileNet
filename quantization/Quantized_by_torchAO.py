# quantize_resnet34.py
import torch
from Resnet4CIFAR100 import resnet34

# Step 1: 加载 FP32 模型
model = resnet34(num_classes=100, img_size=32)
model.load_state_dict(torch.load("resnet34_cifar100_best_fp32.pth", 
                                 map_location="cpu"))
model.eval()  # 必须设为 eval 模式！

print("✅ FP32 model loaded.")

# Step 2: 应用 INT8 动态量化（推荐用于 CNN）
from torchao.quantization import quantize_
from torchao.quantization.quant_api import int8_dynamic_activation_int8_weight

quantize_(model, int8_dynamic_activation_int8_weight())

print("✅ Model quantized to INT8 (dynamic).")

# Step 3: 保存量化后的模型
torch.save(model.state_dict(), "resnet34_cifar100_int8.pth")
print("✅ Quantized model saved as 'resnet34_cifar100_int8.pth'")