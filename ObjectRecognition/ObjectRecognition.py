import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ConvBn2d import *
from ResidualBlocks import *
from Layers import *
from YOLOv3 import *

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

class Darknet3(nn.Module):
    def __init__(self, nC: int = 80) -> None:
        super().__init__()
        # the first layers before the first intermediate result needs to be retained
        self.dnrl1 = nn.Sequential(
            ConvBnLeaky(3, 32, 3, 1, 1),
            ConvBnLeaky(32, 64, 3, 2, 1),
            Darknet3ResidualLayer(64, 32, 1),
            ConvBnLeaky(64, 128, 3, 2, 1),
            Darknet3ResidualLayer(128, 64, 2),
            ConvBnLeaky(128, 256, 3, 2, 1),
            Darknet3ResidualLayer(256, 128, 8)
        )
        # next retained result
        self.dnrl2 = Darknet3ResidualLayer(512, 256, 8, prefix=[ConvBnLeaky(256, 512, 3, 2, 1)])
        # next retained result
        self.dnrl3 = nn.Sequential(
            ConvBnLeaky(512, 1024, 3, 2, 1),
            Darknet3ResidualLayer(1024, 512, 4),
            Darknet3Layer(1024, 512, 4),
            ConvBnLeaky(1024, 512, 1, 1)
        )
        self.py1 = nn.Sequential(ConvBnLeaky(512, 1024, 3, 1, 1), ConvBn(1024, 255, 1, 1))
        self.yolo1 = YoloLayer([6, 7, 8], nC)
        self.cbu1 = ConvBnLeaky(512, 256, 1, 1, suffix=[Upsample()])
        self.dnl2 = nn.Sequential(ConvBnLeaky(768, 256, 1, 1),
            ConvBnLeaky(256, 512, 3, 1, 1), ConvBnLeaky(512, 256, 1, 1),
            ConvBnLeaky(256, 512, 3, 1, 1), ConvBnLeaky(512, 256, 1, 1))
        self.py2 = nn.Sequential(ConvBnLeaky(256, 512, 3, 1, 1), ConvBn(512, 255, 1, 1))
        self.yolo2 = YoloLayer([3, 4, 5], nC)
        self.cbu2 = ConvBnLeaky(256, 128, 1, 1, suffix=[Upsample()])
        self.py3 = nn.Sequential(ConvBnLeaky(384, 128, 1, 1), ConvBnLeaky(128, 256, 3, 1, 1),
            Darknet3Layer(256, 128, 2), ConvBn(256, 255, 1, 1))
        self.yolo3 = YoloLayer([0, 1, 2], nC)

    def forward(self, x: Tensor) -> List[Tensor] | Tensor:
        yolo: List[Tensor] = [None, None, None]
        imgSize: int = x.shape[-1]
        yolo[2] = self.dnrl1(x)
        yolo[1] = self.dnrl2(yolo[2])
        temp: Tensor = self.dnrl3(yolo[1])
        yolo[0] = self.yolo1(self.py1(temp), imgSize)
        temp = self.dnl2(torch.cat([self.cbu1(temp), yolo[1]], 1))
        yolo[1] = self.yolo2(self.py2(temp), imgSize)
        yolo[2] = self.yolo3(self.py3(torch.cat([self.cbu2(temp), yolo[2]], 1)), imgSize)
        return yolo if self.training else torch.cat(yolo, 1)

if __name__ == '__main__':
    rl = Darknet3(80).cuda()
    # rl = ResidualLayer1(64, 128, 8)
    # rl = ParallelFuseBn(64, *[DenseBlock1(64, 64, 64) for i in range(0, 8)]).cuda()
    # rl = DenseBlock1(64, 64, 64).cuda()
    x: Tensor = torch.rand((1, 3, 416, 416), requires_grad=True).float().cuda().abs()
    x = 255 * x / x.max()
    res = rl.forward(x)
    print([r.shape for r in res])
