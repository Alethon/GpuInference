import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torch import Tensor

torch.set_grad_enabled(False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

def readLabel(labelPath: str) -> np.ndarray:
    with open(labelPath, 'r') as f:
        lines = f.read().splitlines()
    label: np.ndarray = np.array([l.split() for l in lines], dtype=np.float32)
    # to xyxy
    lc: np.ndarray = label.copy()
    label[:, 1] = lc[:, 1] - lc[:, 3] / 2
    label[:, 3] = lc[:, 1] + lc[:, 3] / 2
    label[:, 2] = lc[:, 2] - lc[:, 4] / 2
    label[:, 4] = lc[:, 2] + lc[:, 4] / 2
    return label

def sum_pool2d(x: Tensor, kernel_size) -> Tensor:
    return F.lp_pool2d(x, 1, kernel_size=kernel_size, stride=1)
    
def saveMask(mask: Tensor, segmentPath: str) -> None:
    npMask = (255 * mask).type(dtype=torch.uint8).cpu().numpy()
    Image.fromarray(npMask).save(segmentPath)
    
def saveTensor(t: Tensor, segmentPath: str) -> None:
    t.numpy().tofile(segmentPath)

class SegmentData:
    def __init__(self, basePath: str = '.', inputSize: tuple[int, int] = (1056, 1056), segmentSize: tuple[int, int] = (416, 416), totalThresh: float = 0.9) -> None:
        self.inputSize: tuple[int, int] = inputSize
        self.segmentSize: tuple[int, int] = segmentSize

        self.totalThresh: float = totalThresh

        labelPath: str = os.path.join(basePath, 'labels')
        segmentPath: str = os.path.join(basePath, 'segments', str(self.inputSize), str(self.segmentSize))
        if not os.path.exists(segmentPath):
            os.makedirs(segmentPath)
        self.paths: list[str] = [(os.path.join(labelPath, f), os.path.join(segmentPath, f.replace('.txt', ''))) for f in os.listdir(labelPath)]
    
    def imageMask(self, labelPath: str) -> tuple[Tensor, Tensor]:
        label: np.ndarray = readLabel(labelPath)
        mask: Tensor = torch.zeros((label.shape[0], 1, self.inputSize[1], self.inputSize[0]))
        for i, (_, x1, y1, x2, y2) in enumerate(label):
            xmin: int = int((self.inputSize[0] - 1) * x1)
            xmax: int = int((self.inputSize[0] - 1) * x2)
            ymin: int = int((self.inputSize[1] - 1) * y1)
            ymax: int = int((self.inputSize[1] - 1) * y2)
            mask[i, 0, ymin:ymax+1, xmin:xmax+1] = 1.0
        minTotal: Tensor = self.totalThresh * mask.sum(3).sum(2).unsqueeze(2)
        mask = sum_pool2d(mask, (self.segmentSize[1], self.segmentSize[0])).squeeze(1)
        labelsMask: Tensor = (mask > 0)
        okayPoints: Tensor = (mask == 0)
        mask = (mask >= minTotal)
        okayPoints = (torch.logical_or(mask, okayPoints).sum(0) == okayPoints.shape[0])
        mask = torch.logical_and(mask.sum(0) > 0, okayPoints)
        return mask.cpu(), labelsMask.cpu()
    
    def extractAndSaveMasks(self) -> None:
        for labelPath, segmentPath in self.paths:
            mask, labelsMask = self.imageMask(labelPath)
            saveTensor(mask, segmentPath + '.mask')
            saveTensor(labelsMask, segmentPath + '.labels')


if __name__ == '__main__':
    # sd = SegmentData(segmentSize=(416, 416))
    # sd = SegmentData(segmentSize=(1056, 1056))
    sd = SegmentData(basePath = os.path.join('.', 'Lego'), inputSize=(1920, 1080), segmentSize=(1920, 1080))
    sd.extractAndSaveMasks()
    sd = SegmentData(basePath = os.path.join('.', 'Lego'), inputSize=(1056, 1056), segmentSize=(1056, 1056))
    sd.extractAndSaveMasks()
    sd = SegmentData(basePath = os.path.join('.', 'Lego'), inputSize=(1056, 1056), segmentSize=(416, 416))
    sd.extractAndSaveMasks()
