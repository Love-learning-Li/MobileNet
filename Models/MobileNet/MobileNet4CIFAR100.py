import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """修正后的深度可分离卷积"""
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                  stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # 逐点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                  stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=100, alpha=1.0):
        super().__init__()
        # 第一层不下采样，保持32×32
        self.conv1 = nn.Conv2d(3, int(32 * alpha), kernel_size=3, 
                               stride=1, padding=1, bias=False)  # stride=1
        self.bn1 = nn.BatchNorm2d(int(32 * alpha))
        self.relu = nn.ReLU(inplace=True)

        # 优化后的特征提取层，减少下采样次数
        self.features = nn.Sequential(
            DepthwiseSeparableConv(int(32 * alpha), int(64 * alpha), stride=1),   # 32×32
            DepthwiseSeparableConv(int(64 * alpha), int(128 * alpha), stride=2),  # 16×16
            DepthwiseSeparableConv(int(128 * alpha), int(128 * alpha), stride=1), # 16×16
            DepthwiseSeparableConv(int(128 * alpha), int(256 * alpha), stride=2), # 8×8
            DepthwiseSeparableConv(int(256 * alpha), int(256 * alpha), stride=1), # 8×8
            DepthwiseSeparableConv(int(256 * alpha), int(512 * alpha), stride=2), # 4×4
            # 减少512层的重复次数
            DepthwiseSeparableConv(int(512 * alpha), int(512 * alpha), stride=1), # 4×4
            DepthwiseSeparableConv(int(512 * alpha), int(512 * alpha), stride=1), # 4×4
            # 最后一层下采样到2×2
            DepthwiseSeparableConv(int(512 * alpha), int(1024 * alpha), stride=2), # 2×2
            DepthwiseSeparableConv(int(1024 * alpha), int(1024 * alpha), stride=1), # 2×2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 添加dropout防止过拟合
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(int(1024 * alpha), num_classes)
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.features(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out