# CREDIT: https://github.com/ultralytics/yolov3/blob/v3.0/models.py

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class Upsample(nn.Module):
    def __init__(self, scaleFactor=(2, 2), adjust=(0, 0), mode='bilinear'):
        super().__init__()
        self.scaleFactor = scaleFactor
        self.adjust = adjust
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        _, _, w, h = x.shape
        return F.interpolate(x, scale_factor=(self.scaleFactor[0] + self.adjust[0] / (1.0 * w), self.scaleFactor[1] + self.adjust[1] / (1.0 * h)), mode=self.mode)

class YoloLayer(nn.Module):
    div = 1. # 416.
    coordinates = [(10 / div, 13 / div), (16 / div, 30 / div), (33 / div, 23 / div),
                   (30 / div, 61 / div), (62 / div, 45 / div), (59 / div, 119 / div),
                   (116 / div, 90 / div), (156 / div, 198 / div), (373 / div, 326 / div)]
    
    def __init__(self, mask: list[int], nC: int):
        super().__init__()
        self.anchors = torch.FloatTensor([YoloLayer.coordinates[m] for m in mask])
        self.nA = len(mask)  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        # print('nC: ' + str(self.nC))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.createGrids((32, 32), 1, 1)

    def forward(self, x: Tensor, imgSize: tuple[int, int]):
        bs, nGx, nGy = x.shape[0], x.shape[-1], x.shape[-2]
        if self.imgSize[0] != imgSize[0] or self.imgSize[1] != imgSize[1]:
            self.createGrids(imgSize, nGx, nGy)

        # x.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        x = x.view(bs, self.nA, self.nC + 5, nGy, nGx).permute(0, 1, 3, 4, 2).contiguous().to(self.device)  # prediction

        if self.training:
            return x
        else:  # inference
            x[..., 0:2] = torch.sigmoid(x[..., 0:2]) + self.gridXy  # xy
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchorWh  # wh yolo method
            # x[..., 2:4] = ((torch.sigmoid(x[..., 2:4]) * 2) ** 2) * self.anchor_wh  # wh power method
            x[..., 4] = torch.sigmoid(x[..., 4])  # p_conf
            x[..., 0] *= self.strideX
            x[..., 1] *= self.strideY
            x[..., 2] *= self.strideX
            x[..., 3] *= self.strideY
            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return x.view(bs, -1, 5 + self.nC)
        
    def createGrids(self, imgSize: tuple[int, int], nGx: int, nGy: int):
        self.imgSize = imgSize
        self.strideX = imgSize[0] / nGx
        self.strideY = imgSize[1] / nGy

        # build xy offsets
        gridX = torch.arange(nGx).repeat((nGy, 1)).view((1, 1, nGy, nGx)).float()
        gridY = torch.arange(nGy).repeat((nGx, 1)).view((1, 1, nGx, nGy)).permute(0, 1, 3, 2).float()
        # gridY = gridX.permute(0, 1, 3, 2)
        self.gridXy = torch.stack((gridX, gridY), 4).to(self.device)

        # build wh gains
        self.anchorVector = self.anchors.to(self.device)
        self.anchorVector[:, 0] = self.anchorVector[:, 0] / self.strideX
        self.anchorVector[:, 1] = self.anchorVector[:, 1] / self.strideY
        self.anchorWh = self.anchorVector.view(1, self.nA, 1, 1, 2).to(self.device)
        self.nGx = torch.FloatTensor([nGx]).to(self.device)
        self.nGy = torch.FloatTensor([nGy]).to(self.device)
    
    def cuda(self) -> nn.Module:
        self.device = torch.device('cuda')
        return super().cuda()
    
    def to(self, device) -> nn.Module:
        self.device = device
        return super().to(device=device)
