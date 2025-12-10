import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange

def conv_bn(input_channel, output_channel, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size,
                  stride=stride, padding=kernel_size // 2),
        nn.BatchNorm2d(output_channel),
        nn.SiLU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class MV2Block(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        hidden_dim = input_channel * expansion
        self.use_res_connection = (stride == 1 and input_channel == output_channel)

        if expansion == 1:
            self.conv = nn.Sequential(
                ## 3x3 DepthWise Conv -> BN -> SiLU -> 1x1 PointWise Conv -> BN
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=self.stride,
                          padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, output_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(output_channel)
            )
        else:
            self.conv = nn.Sequential(
                 ## 1x1 Conv -> BN -> SiLU -> 3x3 DepthWise Conv -> BN -> SiLU -> 1x1 PointWise Conv -> BN
                nn.Conv2d(input_channel, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                          padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, output_channel, kernel_size=1, stride=1, bias=False),
                nn.SiLU(),
                nn.BatchNorm2d(output_channel)
            )

    def forward(self, x):
        if (self.use_res_connection):
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        return out

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # b：batch size , 一批有多少张图 / 样本
        # p：patch size
        # n：sequence length (一个Patch中有多少位置)
        # h：heads(self.heads)
        # d：dimension per head(dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)
        # (b, p, n, h*d) -> (b, p, h, n, d)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MobileViTAttention(nn.Module):
    def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=7, depth=3, mlp_dim=1024):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size
        # DeepWise Conv 3x3
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)
        # PointWise Conv 1x1
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)

        self.trans = Transformer(dim=dim, depth=depth, heads=8, dim_head=64, mlp_dim=mlp_dim)
        # PointWise Conv 1x1
        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)
        self.conv4 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        y = x.clone()  # bs,c,h,w

        ## Local Representation
        y = self.conv2(self.conv1(x))  # bs,dim,h,w

        ## Global Representation
        _, _, h, w = y.shape
        y = rearrange(y, 'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim', ph=self.ph, pw=self.pw)  # bs,h,w,dim
        y = self.trans(y)
        y = rearrange(y, 'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)', ph=self.ph, pw=self.pw, nh=h // self.ph,
                      nw=w // self.pw)  # bs,dim,h,w

        ## Fusion
        y = self.conv3(y)  # bs,dim,h,w
        y = torch.cat([x, y], 1)  # bs,2*dim,h,w
        y = self.conv4(y)  # bs,c,h,w

        return y


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, depths=[2, 4, 3], expansion=4, kernel_size=3,
                 patch_size=2):
        super().__init__()
        ih, iw = image_size, image_size
        ph, pw = patch_size, patch_size
        assert iw % pw == 0 and ih % ph == 0

        self.conv1 = conv_bn(3, channels[0], kernel_size=3, stride=patch_size)
        self.mv2 = nn.ModuleList([])
        self.m_vits = nn.ModuleList([])

        self.mv2.append(MV2Block(channels[0], channels[1], 1))
        self.mv2.append(MV2Block(channels[1], channels[2], 2))
        self.mv2.append(MV2Block(channels[2], channels[3], 1))
        self.mv2.append(MV2Block(channels[2], channels[3], 1))  # x2
        self.mv2.append(MV2Block(channels[3], channels[4], 2))
        self.m_vits.append(MobileViTAttention(channels[4], dim=dims[0], kernel_size=kernel_size, patch_size=patch_size,
                                              depth=depths[0], mlp_dim=int(2 * dims[0])))
        self.mv2.append(MV2Block(channels[4], channels[5], 2))
        self.m_vits.append(MobileViTAttention(channels[5], dim=dims[1], kernel_size=kernel_size, patch_size=patch_size,
                                              depth=depths[1], mlp_dim=int(4 * dims[1])))
        self.mv2.append(MV2Block(channels[5], channels[6], 2))
        self.m_vits.append(MobileViTAttention(channels[6], dim=dims[2], kernel_size=kernel_size, patch_size=patch_size,
                                              depth=depths[2], mlp_dim=int(4 * dims[2])))

        head_index = min(len(channels) - 2, 6)
        head_in_channels = channels[head_index]
        self.conv2 = conv_bn(head_in_channels, channels[-1], kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        y = self.conv1(x)
        y = self.mv2[0](y)
        y = self.mv2[1](y)
        y = self.mv2[2](y)
        y = self.mv2[3](y)
        y = self.mv2[4](y)
        y = self.m_vits[0](y)

        y = self.mv2[5](y)
        y = self.m_vits[1](y)

        y = self.mv2[6](y)
        y = self.m_vits[2](y)

        y = self.conv2(y)
        y = self.pool(y).view(y.shape[0], -1)
        y = self.fc(y)
        return y
    

# 针对CIFAR100, 做出如下修改:
def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT(32, dims, channels, num_classes=100, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT(32, dims, channels, num_classes=100)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT(32, dims, channels, num_classes=100)

# def mobilevit_xxs():
#     dims = [64, 80, 96]
#     channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
#     return MobileViT((256, 256), dims, channels, num_classes=1000, expansion=2)


# def mobilevit_xs():
#     dims = [96, 120, 144]
#     channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
#     return MobileViT((256, 256), dims, channels, num_classes=1000)


# def mobilevit_s():
#     dims = [144, 192, 240]
#     channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
#     return MobileViT((256, 256), dims, channels, num_classes=1000)