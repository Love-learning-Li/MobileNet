import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
    
class hard_swish(nn.Module):
    def forward(self, x):
        x = x * F.relu6(x + 3, inplace=True) / 6
        return x

class hard_sigmoid(nn.Module):
    def forward(self, x):
        x = F.relu6(x + 3, inplace=True) / 6
        return x
    
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()

        squeeze_channels = in_channels // squeeze_factor

        self.se = nn.Sequential(
            # Squeeze - Global Pooling
            # input size: (B, C, H, W), output size:(B, C, 1, 1)
            nn.AdaptiveAvgPool2d(1),
            # Excitation - Fully Connected
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(squeeze_channels, in_channels, kernel_size=1),
            hard_sigmoid()

        )
        
    def forward(self, x):
        x = x * self.se(x)
        return x
    
class InvertedResidualBlock(nn.Module):
    def __init__(self, kernel_size, in_size, exp_size, out_size, stride, use_se, activation):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride

        # Expansion, 扩展卷积
        self.conv1 = nn.Conv2d(in_size, exp_size, kernel_size=1, bias=False)
        # 1x1卷积增加维度
        self.bn1 = nn.BatchNorm2d(exp_size)
        self.activation1 = activation()

        # Depthwise Convolution, 深度卷积
        self.conv2 = nn.Conv2d(exp_size, exp_size, kernel_size = kernel_size, stride = stride, padding = kernel_size // 2, groups = exp_size, bias=False)
        self.bn2 = nn.BatchNorm2d(exp_size)
        self.activation2 = activation()

        # Sequeeze-and-Excitation
        self.use_se = use_se
        if self.use_se:
            self.se = SqueezeExcitation(exp_size)
        
        # 逐点卷积
        self.conv3 = nn.Conv2d(exp_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = activation()

        # Residual Connection
        # 不同于MobileNetV2仅在stride=1且输入输出维度一致是进行残差链接
        # MobileNetV3增加在stride=2时的残差
        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )
    
    def forward(self, x):
        origin_input = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation2(out)

        if self.use_se:
            out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.skip is not None:
            skip = self.skip(origin_input)
        else:
            skip = 0
        
        out = self.act3(out + skip)
        
        return out

class MobileNetV3(nn.Module):
    # def __init__(self, num_classes=100, act=nn.Hardswish)
    def __init__(self, num_classes=100, act=hard_swish):
        super(MobileNetV3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = act()

        self.bneck = nn.Sequential(
            # Inverted Residual Block settings:
            # (kernel_size, in_size, expand_ratio, out_size, stride, use_se, activation):
            InvertedResidualBlock(3, 16, 16, 16, 2, True, nn.ReLU),
            InvertedResidualBlock(3, 16, 72, 24, 2, False, nn.ReLU),
            InvertedResidualBlock(3, 24, 88, 24, 1, False, nn.ReLU),
            InvertedResidualBlock(5, 24, 96, 40, 2, True, act),
            InvertedResidualBlock(5, 40, 240, 40, 1, True, act),
            InvertedResidualBlock(5, 40, 240, 40, 1, True, act),
            InvertedResidualBlock(5, 40, 120, 48, 1, True, act),
            InvertedResidualBlock(5, 48, 144, 48, 1, True, act),
            InvertedResidualBlock(5, 48, 288, 96, 2, True, act),
            InvertedResidualBlock(5, 96, 576, 96, 1, True, act),
            InvertedResidualBlock(5, 96, 576, 96, 1, True, act),
        )
        
        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.act2 = act()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear3 = nn.Linear(576, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.act3 = act()
        self.dropout = nn.Dropout(0.2)

        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.bneck(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        out = self.linear3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.dropout(out)

        out = self.linear4(out)
        return out



    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)




