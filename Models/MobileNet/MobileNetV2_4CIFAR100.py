import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义倒置残差模块
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        # 构建倒置残差模块的网络结构
        layers = []
        if expand_ratio != 1:
            # 1x1卷积，扩展通道数
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # 3x3深度可分离卷积
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        
        # 1x1卷积，线性瓶颈
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            # 如果满足条件，使用残差连接
            return x + self.conv(x)
        else:
            return self.conv(x)

# 定义MobileNetV2网络
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        # 设置每个阶段的倒置残差模块参数
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # 初始卷积层
        input_channel = int(32 * width_mult)
        layers = [nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(input_channel),
                  nn.ReLU6(inplace=True)]

        # 构建倒置残差模块
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # 最后的卷积层
        output_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        layers.append(nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(output_channel))
        layers.append(nn.ReLU6(inplace=True))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x
