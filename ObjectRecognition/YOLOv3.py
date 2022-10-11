# CREDIT: https://github.com/ultralytics/yolov3/blob/v3.0/models.py

from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class Upsample(nn.Module):
    def __init__(self, scaleFactor=2, mode='bilinear'):
        super().__init__()
        self.scaleFactor = scaleFactor
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(x, scale_factor=self.scaleFactor, mode=self.mode)

class YoloLayer(nn.Module):
    div = 1. # 416.
    coordinates = [(10 / div, 13 / div), (16 / div, 30 / div), (33 / div, 23 / div),
                   (30 / div, 61 / div), (62 / div, 45 / div), (59 / div, 119 / div),
                   (116 / div, 90 / div), (156 / div, 198 / div), (373 / div, 326 / div)]
    
    def __init__(self, mask: List[int], nC: int):
        super().__init__()
        self.anchors = torch.FloatTensor([YoloLayer.coordinates[m] for m in mask])
        self.nA = len(mask)  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        # print('nC: ' + str(self.nC))
        self.device = torch.device('cpu')
        self.createGrids(32, 1)

    def forward(self, x: Tensor, imgSize: int):
        bs, nG = x.shape[0], x.shape[-1]
        if self.imgSize != imgSize:
            self.createGrids(imgSize, nG)

        # x.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        x = x.view(bs, self.nA, self.nC + 5, nG, nG).permute(0, 1, 3, 4, 2).contiguous().to(self.device)  # prediction

        if self.training:
            return x
        else:  # inference
            x[..., 0:2] = torch.sigmoid(x[..., 0:2]) + self.gridXy  # xy
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchorWh  # wh yolo method
            # x[..., 2:4] = ((torch.sigmoid(x[..., 2:4]) * 2) ** 2) * self.anchor_wh  # wh power method
            x[..., 4] = torch.sigmoid(x[..., 4])  # p_conf
            x[..., :4] *= self.stride
            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return x.view(bs, -1, 5 + self.nC)
        
    def createGrids(self, imgSize: int, nG: int):
        self.imgSize = imgSize
        self.stride = imgSize / nG

        # build xy offsets
        gridX = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
        gridY = gridX.permute(0, 1, 3, 2)
        self.gridXy = torch.stack((gridX, gridY), 4).to(self.device)

        # build wh gains
        self.anchorVector = self.anchors.to(self.device) / self.stride
        self.anchorWh = self.anchorVector.view(1, self.nA, 1, 1, 2).to(self.device)
        self.nG = torch.FloatTensor([nG]).to(self.device)
    
    def cuda(self) -> nn.Module:
        self.device = torch.device('cuda')
        return super().cuda()
