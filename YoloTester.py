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

class YoloTester(DetectionIterable):
    def __init__(self) -> None:
        super().__init__()

        imageDir = os.path.join('Lego', 'images')
        self.images: ndarray = np.stack([cv2.imread(os.path.join(imageDir, f)) for f in os.listdir(imageDir)])
        self.images[:, :, :, :] = self.images[:, :, :, ::-1]
        # self.images = self.images[np.random.permutation(self.images.shape[0])[:1000]] # select a 1000 image permutation
    
    def __len__(self) -> int:
        return self.images.shape[0]
    
    def __iter__(self):
        self.count = -1
        return super().__iter__()
    
    def __next__(self) -> tuple[ndarray, Tensor, Tensor]:
        self.count += 1
        if self.count == self.images.shape[0]:
            self.breakIteration()
        frame = self.images[self.count].copy()
        image, predictions = self._getPredictions(frame)
        self._showPredictions(frame, predictions)
        return frame, image, predictions

if __name__ == '__main__':
    yt = YoloTester()
    for _, _, _ in yt:
        continue