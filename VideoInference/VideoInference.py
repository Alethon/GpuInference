from typing import Tuple
import cv2
import numpy as np
import torch
from torch import Tensor
from ObjectRecognition import *
from time import time

class VideoDriver:
    def __init__(self, connectionAddress, cuda: bool, resolution: Tuple[int, int] = (1920, 1080)) -> None:
        self.capture: cv2.VideoCapture = cv2.VideoCapture(connectionAddress)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        if cuda:
            self.device: torch.device = torch.device('cuda')
        else:
            self.device: torch.device = torch.device('cpu')
        
        result: Tuple[bool, Tensor] = self._getValidFrame()
        if not result[0]:
            raise ValueError('Could not get an initial frame.')
        
        self.lastFrame: Tensor = result[1]
    
    def _getValidFrame(self, timeoutTries: int = 10) -> Tuple[bool, Tensor]:
        for i in range(0, timeoutTries):
            result: Tuple[bool, np.ndarray] = self.capture.read()
            if result[0]:
                print(result[1].shape)
                return True, torch.from_numpy(result[1]).to(dtype=torch.float32, device=self.device)
        return False, None

    def GetNextFrame(self) -> Tuple[bool, Tensor]:
        validFrame: Tuple[bool, Tensor] = self._getValidFrame()
        while validFrame[0] and torch.equal(self.lastFrame, validFrame[1]):
            print(self.lastFrame)
            validFrame = self._getValidFrame()
        if validFrame[0]:
            self.lastFrame = validFrame[1]
        return validFrame[0], self.lastFrame

class VideoInference:
    def __init__(self, connectionAddress, weightPath: str, channelCount: int = 6, cuda: bool = True) -> None:
        self.driver: VideoDriver = VideoDriver(connectionAddress, cuda)
        self.nn: torch.nn.Module = YoloNet(channelCount)
        if weightPath is not None:
            self.nn.load_state_dict(torch.load(weightPath))
        if cuda:
            self.nn = self.nn.cuda()
    
    def GetNextResult(self) -> Tuple[bool, Tensor]:
        success, frame = self.driver.GetNextFrame()
        print(frame.transpose(0, 2)[None].shape)
        if success:
            return True, self.nn(frame.transpose(0, 2)[None])[0]
        else:
            return False, None

if __name__ == '__main__':
    vi = VideoInference(0, None)
    print(vi.GetNextResult()[1].shape)
    print(vi.driver.lastFrame.shape)
