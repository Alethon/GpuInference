from typing import Tuple
from cv2 import VideoCapture
import numpy as np
import torch
from torch import Tensor
from ObjectRecognition import *
from time import time

class VideoDriver:
    def __init__(self, connectionAddress) -> None:
        self.capture: VideoCapture = VideoCapture(connectionAddress)
        result = self._getValidFrame()
        if not result[0]:
            raise ValueError('Could not get an initial frame.')
        self.lastFrame: np.ndarray = result[1]
    
    def _getValidFrame(self, timeoutTries: int = 10) -> Tuple[bool, np.ndarray]:
        for i in range(0, timeoutTries):
            result = self.capture.read()
            if result[0]:
                return result
        return result

    def GetNextFrame(self) -> Tuple[bool, np.ndarray]:
        validFrame = self._getValidFrame()
        while validFrame[0] and not np.array_equal(self.lastFrame, validFrame[1]):
            validFrame = self._getValidFrame()
        if validFrame[0]:
            self.lastFrame = validFrame[1]
        return validFrame[0], self.lastFrame

class VideoInference:
    def __init__(self, connectionAddress, weightPath: str, channelCount: int = 6, cuda: bool = True) -> None:
        self.driver: VideoDriver = VideoDriver(connectionAddress)
        self.nn: torch.nn.Module = YoloNet(channelCount)
        if weightPath is not None:
            self.nn.load_state_dict(torch.load(weightPath))
        if cuda:
            self.device = torch.device('cuda')
            self.nn = self.nn.cuda()
        else:
            self.device = torch.device('cpu')
    
    def GetNextResult(self) -> Tuple[bool, Tensor]:
        success, frame = self.driver.GetNextFrame()
        if success:
            return True, self.nn(torch.from_numpy(frame).to(dtype=torch.float32, device=self.device))[0]
        else:
            return False, None

if __name__ == '__main__':
    vi = VideoInference(0, None)
