import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ConvBn2d import *
from ResBlocks import *
from ResLayers import *

def downsample_add(x: Tensor, y: Tensor) -> Tensor:
    _, _, h, w = x.size()
    return F.interpolate(y, (h, w), mode='bilinear') + x

class Prefilter(nn.Module):
    def __init__(self, channels_out: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.convbn1 = ConvBn(3, channels_out / 2, kernel_size, stride)
        self.convbn2 = ConvBn(channels_out / 2, channels_out, kernel_size, stride)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.convbn2(self.convbn1(x))

class YoloNet(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.filter = Prefilter(64, 3, 1)
        self.rl1 = ResidualLayer1(64, 128, 2)
        self.rl2 = ResidualLayer1(128, 256, 2)
        self.rl3 = ResidualLayer1(256, 512, 2)
        self.rl4 = ResidualLayer1(512, 1024, 2)
        self.lat1layer1 = nn.Conv2d(256, 128, 1, 1)
        self.lat1layer2 = nn.Conv2d(512, 256, 1, 1)
        self.lat1layer3 = nn.Conv2d(1024, 512, 1, 1)
        self.fpn01 = ConvBnDropout(128, 64, 1, 1)
        self.fpn02 = ConvBnDropout(256, 64, 1, 1)
        self.fpn03 = ConvBnDropout(512, 64, 1, 1)
        self.out1 = ConvBn(64, channels, 1, 1)
        self.out2 = ConvBn(64, channels, 1, 1)
        self.out3 = ConvBn(64, channels, 1, 1)

    def forward(self, x: Tensor):
        f = self.filter(x)
        c1 = self.rl1(f)
        c2 = self.rl2(c1)
        c3 = self.rl3(c2)
        c4 = self.rl4(c3)
        fpn03 = self.fpn03(c3 + self.lat1layer2(c4))
        fpn02 = self.fpn02(c2 + self.lat1layer2(c3) + fpn03)
        fpn01 = self.fpn01(c1 + self.lat1layer1(c2) + fpn02)
        out1 = self.out1(fpn01)
        out2 = self.out2(fpn02 + out1)
        out3 = self.out3(fpn03 + out2)
        return out1, out2, out3


if __name__ == '__main__':
    rl = YoloNet(10).cuda()
    # rl = ResidualLayer1(64, 128, 8)
    # rl = ParallelFuseBn(64, *[DenseBlock1(64, 64, 64) for i in range(0, 8)]).cuda()
    # rl = DenseBlock1(64, 64, 64).cuda()
    x: Tensor = torch.rand((1, 64, 600, 600), requires_grad=True).float().cuda().abs()
    x = 255 * x / x.max()
    print(type(rl.forward(x)))
