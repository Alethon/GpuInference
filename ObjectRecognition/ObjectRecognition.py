import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ConvBn2d import *
from ResidualBlocks import *
from Layers import *

def upsample(t: Tensor, h: int, w: int) -> Tensor:
    return F.interpolate(t, (h, w), mode='bilinear')

class DarknetTiny3(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # prefilter
        self.pf = DarknetTiny3Block(3, 32)
        # layer 1
        self.dntl1 = DarknetTiny3Layer(32, 64)
        # layer 2
        self.dntl2 = DarknetTiny3Layer(64, 128)
        # layer 3
        self.dntl3 = DarknetTiny3Layer(128, 256)
        # layer 4
        self.dntl4 = DarknetTiny3Layer(256, 512)
        # layer 5
        self.dntl5 = DarknetTiny3Layer(512, 1024)
        # combobs
        self.cb1 = ConvBnLeaky(1024, 512, 1, 1)
        self.cb2 = ConvBnLeaky(1024, 512, 1, 1)
        self.cb3 = ConvBnLeaky(1024, 512, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        # prefilter
        out: Tensor = self.pf(x)
        # layer 1
        out = self.dntl1(out)
        # layer 2
        p1 = self.dntl2(out)
        # layer 3
        p2 = self.dntl3(out)
        # layer 4
        p3 = self.dntl4(out)
        # layer 5
        p4 = self.dntl5(out)
        out = upsample(self.cb1(p4), 120, 67)
        out = upsample(self.cb2(out + p3), 120, 67)
        return out


if __name__ == '__main__':
    rl = DarknetTiny3(10).cuda()
    # rl = ResidualLayer1(64, 128, 8)
    # rl = ParallelFuseBn(64, *[DenseBlock1(64, 64, 64) for i in range(0, 8)]).cuda()
    # rl = DenseBlock1(64, 64, 64).cuda()
    x: Tensor = torch.rand((1, 3, 1920, 1080), requires_grad=True).float().cuda().abs()
    x = 255 * x / x.max()
    rl.forward(x).shape
