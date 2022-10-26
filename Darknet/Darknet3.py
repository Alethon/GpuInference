import torch
import torch.nn as nn
from torch import Tensor

from Darknet.ConvBn2d import *
from Darknet.Blocks import *
from Darknet.ResidualBlocks import *
from Darknet.Layers import *

from Darknet.YOLOv3 import *

class DarknetTiny3(nn.Module):
    def __init__(self, nC: int, scale: float = 1.0) -> None:
        super().__init__()
        # intermediate channel count
        icc = 3 * nC + 15

        # intermediate result
        self.dntl1 = nn.Sequential(DarknetTiny3Block(3, int(32 * scale)),
                                   DarknetTiny3Layer(int(32 * scale), int(64 * scale)),
                                   DarknetTiny3Layer(int(64 * scale), int(128 * scale)),
                                   DarknetTiny3Layer(int(128 * scale), int(256 * scale)),
                                   DarknetTiny3Layer(int(256 * scale), int(512 * scale)))

        # intermediate result
        self.dntl2 = DarknetTiny3Layer(int(512 * scale), int(1024 * scale), suffix=[ConvBnLeaky(int(1024 * scale), 256, 1, 1)])

        # yolo 1
        self.py1 = nn.Sequential(DarknetTiny3Block(256, 512), ConvBn(512, icc, 1, 1))
        self.yolo1 = YoloLayer([3, 4, 5], nC)

        self.cblu = ConvBnLeaky(256, 128, 1, 1, suffix=[Upsample()])

        # yolo 2
        self.py2 = nn.Sequential(DarknetTiny3Block(int(512 * scale) + 128, 256), ConvBn(256, icc, 1, 1))
        self.yolo2 = YoloLayer([0, 1, 2], nC)

    def forward(self, x: Tensor) -> list[Tensor] | Tensor:
        yolo: list[Tensor] = [None, None]
        imgSize: tuple[int, int] = tuple((x.shape[-1], x.shape[-2]))
        yolo[1] = self.dntl1(x)
        temp: Tensor = self.dntl2(yolo[1])
        yolo[0] = self.yolo1(self.py1(temp), imgSize)
        yolo[1] = self.yolo2(self.py2(torch.cat([self.cblu(temp), yolo[1]], 1)), imgSize)
        return yolo if self.training else torch.cat(yolo, 1)

class Darknet3(nn.Module):
    def __init__(self, nC: int) -> None:
        super().__init__()
        # intermediate channel count
        icc = 3 * nC + 15

        # intermediate result
        self.dnrl1 = nn.Sequential(ConvBnLeaky(3, 32, 3, 1, 1),
                                   ConvBnLeaky(32, 64, 3, 2, 1),
                                   Darknet3ResidualLayer(64, 32, 1),
                                   ConvBnLeaky(64, 128, 3, 2, 1),
                                   Darknet3ResidualLayer(128, 64, 2),
                                   ConvBnLeaky(128, 256, 3, 2, 1),
                                   Darknet3ResidualLayer(256, 128, 8))
        
        # intermediate result
        self.dnrl2 = Darknet3ResidualLayer(512, 256, 8, prefix=[ConvBnLeaky(256, 512, 3, 2, 1)])

        # intermediate result
        self.dnrl3 = nn.Sequential(ConvBnLeaky(512, 1024, 3, 2, 1),
                                   Darknet3ResidualLayer(1024, 512, 4),
                                   Darknet3Layer(1024, 512, 4),
                                   ConvBnLeaky(1024, 512, 1, 1))
        
        # the 1st yolo
        self.yolo1 = YoloLayer([6, 7, 8], nC)
        self.py1 = nn.Sequential(ConvBnLeaky(512, 1024, 3, 1, 1),
                                 ConvBn(1024, icc, 1, 1))

        # intermediate result
        self.cbu1 = ConvBnLeaky(512, 256, 1, 1, suffix=[Upsample()])
        self.dnl = nn.Sequential(ConvBnLeaky(768, 256, 1, 1),
                                 ConvBnLeaky(256, 512, 3, 1, 1),
                                 ConvBnLeaky(512, 256, 1, 1),
                                 ConvBnLeaky(256, 512, 3, 1, 1),
                                 ConvBnLeaky(512, 256, 1, 1))

        # the 2nd yolo
        self.yolo2 = YoloLayer([3, 4, 5], nC)
        self.py2 = nn.Sequential(ConvBnLeaky(256, 512, 3, 1, 1),
                                 ConvBn(512, icc, 1, 1))

        # intermediate result
        self.cbu2 = ConvBnLeaky(256, 128, 1, 1, suffix=[Upsample(adjust=(-1, 0))])

        # the 3rd yolo
        self.yolo3 = YoloLayer([0, 1, 2], nC)
        self.py3 = nn.Sequential(ConvBnLeaky(384, 128, 1, 1),
                                 ConvBnLeaky(128, 256, 3, 1, 1),
                                 Darknet3Layer(256, 128, 2),
                                 ConvBn(256, icc, 1, 1))

    def forward(self, x: Tensor) -> list[Tensor] | Tensor:
        yolo: list[Tensor] = [None, None, None]
        imgSize: tuple[int, int] = tuple((x.shape[-1], x.shape[-2]))
        yolo[2] = self.dnrl1(x)
        yolo[1] = self.dnrl2(yolo[2])
        temp: Tensor = self.dnrl3(yolo[1])
        yolo[0] = self.yolo1(self.py1(temp), imgSize)
        temp = self.dnl(torch.cat([self.cbu1(temp), yolo[1]], 1))
        yolo[1] = self.yolo2(self.py2(temp), imgSize)
        print(self.cbu2(temp).shape, yolo[2].shape)
        yolo[2] = self.yolo3(self.py3(torch.cat([self.cbu2(temp), yolo[2]], 1)), imgSize)
        return yolo if self.training else torch.cat(yolo, 1)

if __name__ == '__main__':
    rl = DarknetTiny3(10, 0.5).cuda()
    x: Tensor = torch.rand((1, 3, 416, 416), requires_grad=True).float().cuda().abs()
    x = 255 * x / x.max()
    res = rl.forward(x)
    print([r.shape for r in res])
