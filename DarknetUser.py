import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from numpy import ndarray

from Darknet.Darknet3 import *
from Darknet3Data import *

def image_to_tensor(imageNp: ndarray, device: torch.device, size: tuple[int, int]) -> Tensor:
    image: Tensor = torch.from_numpy(imageNp)[None].cuda().float().permute(0, 3, 1, 2) / 255.0
    return F.interpolate(image, size=(size[1], size[0]), mode='bilinear').to(device)

class DarknetUser:
    def __init__(self, datasetInfoPath: str) -> None:
        self.info: dict[str, any] = readDatasetInfo(datasetInfoPath)
        
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classCount: int = self.info['classes']
        
        if self.info['useTiny']:
            self.model: DarknetTiny3 = DarknetTiny3(self.classCount).to(self.device)
            self.yolos: list[YoloLayer] = [self.model.yolo1, self.model.yolo2]
            self.weightsPath: str = WEIGHTS_TINY
        else:
            self.model: Darknet3 = Darknet3(self.classCount).to(self.device)
            self.yolos: list[YoloLayer] = [self.model.yolo1, self.model.yolo2, self.model.yolo3]
            self.weightsPath: str = WEIGHTS

        if not os.path.isdir(self.weightsPath):
            os.makedirs(self.weightsPath)

        self.latestWeightsPath: str = os.path.join(self.weightsPath, 'latest.pt')
        
        with open(self.info['names'], 'r') as f:
            self.names = f.read().strip().split('\n')

    def loadCheckpoint(self, checkPointPath: str) -> dict:
        self.model = self.model.cpu()
        checkpoint: dict = torch.load(checkPointPath)
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(self.device)
        return checkpoint
    
    def evaluate(self, imageNp: ndarray, size: tuple[int, int] = (416, 416)) -> Tensor:
        with torch.no_grad():
            image: Tensor = image_to_tensor(imageNp, self.device, size)
            return self.model(image)

