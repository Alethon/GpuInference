import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time
from torch import Tensor
from numpy import ndarray

from utils.utils import *
from DarknetUser import *

from DetectionIterable import *

def image_to_tensor(imageNp: ndarray, device: torch.device, size: tuple[int, int]) -> Tensor:
    image: Tensor = torch.from_numpy(imageNp)[None].cuda().float().permute(0, 3, 1, 2) / 255.0
    return F.interpolate(image, size=(size[1], size[0]), mode='bilinear').to(device)

class CameraUser(DetectionIterable):
    def __init__(self, captureInfo, datasetInfoPath: str = ..., weightFilePath: str = ..., streamSize: tuple[int, int] = (1920, 1080)) -> None:
        super().__init__()

        self.showResults = True
        self.realTimePlayback = True

        self.captureInfo = captureInfo
        self.streamSize: tuple[int, int] = streamSize
        self.capture: cv2.VideoCapture = None
    
    def __iter__(self):
        self.capture = cv2.VideoCapture(self.captureInfo)
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.streamSize[0])
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.streamSize[1])
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.capture.set(cv2.CAP_PROP_FPS, 30)
        return super().__iter__()
    
    def breakIteration(self):
        self.capture.release()
        super().breakIteration()
    
    def __len__(self) -> int:
        return 0
    
    def __next__(self) -> tuple[ndarray, Tensor, Tensor]:
        success, frame = self.capture.read()
        if not success:
            self.breakIteration()
        # frame[:, :, :] = frame[:, :, ::-1]
        image, predictions = self._getPredictions(frame)
        self._showPredictions(frame, predictions)
        return frame, image, predictions

if __name__ == '__main__':
    # cu = CameraUser(0)
    cu = CameraUser('rtsp://192.168.117.59:8554/unicast')
    for frame, image, predictions in cu:
        continue
        
