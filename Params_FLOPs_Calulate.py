import torch
from thop import profile

# 假设你有一个定义好的 MobileNet 模型实例
# from torchvision.models import mobilenet_v2
from Models.MobileNetV3.MobileNetV3 import MobileNetV3
from Models.MobileNetV2.MobileNetV2 import MobileNetV2
from Models.MobileNetV2.MobileNetV2_4CIFAR100 import MobileNetV2_4CIFAR100
from Models.MobileNet.MobileNet4CIFAR100 import MobileNetV1


model = MobileNetV3()
input_tensor = torch.randn(1, 3, 224, 224)

# 计算 MACs 和 Params
macs, params = profile(model, inputs=(input_tensor, ))

print(f"MACs: {macs / 1e9:.2f} G") # 输出为 Giga MACs
print(f"Params: {params / 1e6:.2f} M") # 输出为 Mega Params

# MobileNetV3测试:
# MACs: 0.07 G
# Params: 1.80 M

# MobileNetV1测试:
# MACs: 1.71 G
# Params: 2.50 M

# MobileNetV2测试:
# 该模型针对CIFAR
# MACs: 1.30 G
# Params: 2.35 M

# MobileNetV2_4CIFAR100测试:
# 该模型是标准MobileNetV2，无额外修改
# MACs: 0.33 G
# Params: 3.50 M



