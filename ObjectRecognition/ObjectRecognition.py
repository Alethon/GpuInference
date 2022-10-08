import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ConvBn2d import *
from ResidualBlocks import *
from Layers import *

def downsample_add(x: Tensor, y: Tensor) -> Tensor:
    _, _, h, w = x.size()
    return F.interpolate(y, (h, w), mode='bilinear') + x

class DarknetTiny3(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # prefilter
        self.pf = DarknetTiny3Layer(3, 32)
        # layer 1
        self.dntl1 = DarknetTiny3Layer(32, 64)
        # layer 2
        self.dntl2 = DarknetTiny3Layer(64, 128)
        # layer 3
        self.dntl3 = DarknetTiny3Layer(128, 256)
        # layer 4
        self.dntl4 = DarknetTiny3Layer(256, 512)
        # layer 5
        self.dntl5 = DarknetTiny3Block(512, 1024)

    def forward(self, x: Tensor) -> Tensor:
        # prefilter
        out = self.pf(x)
        # layer 1
        out = self.dntl1(out)
        # layer 2
        out = self.dntl2(out)
        # layer 3
        out = self.dntl3(out)
        # layer 4
        out = self.dntl4(out)
        # layer 5
        out = self.dntl5(out)
        return out


if __name__ == '__main__':
    rl = DarknetTiny3(10).cuda()
    # rl = ResidualLayer1(64, 128, 8)
    # rl = ParallelFuseBn(64, *[DenseBlock1(64, 64, 64) for i in range(0, 8)]).cuda()
    # rl = DenseBlock1(64, 64, 64).cuda()
    x: Tensor = torch.rand((1, 3, 512, 512), requires_grad=True).float().cuda().abs()
    x = 255 * x / x.max()
    print(rl.forward(x).shape)
