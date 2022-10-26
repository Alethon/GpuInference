import os
import cv2
from cv2 import VideoCapture
import numpy as np
from PIL import Image

class FrameSaver:
    def __init__(self, connectionAddress, savePath: str, resolution: tuple[int, int] = (1920, 1080)) -> None:
        self.capture: VideoCapture = cv2.VideoCapture(connectionAddress)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.baseSavePath: str = savePath
        self.baseImageSavePath: str = os.path.join(savePath, 'images')
        if not os.path.exists(self.baseImageSavePath):
            os.makedirs(self.baseImageSavePath)
    
    def _getValidFrame(self, timeoutTries: int = 10) -> tuple[bool, np.ndarray]:
        for i in range(0, timeoutTries):
            result: tuple[bool, np.ndarray] = self.capture.read()
            if result[0]:
                return result
        return False, None

    def getNextFrame(self) -> bool:
        validFrame: tuple[bool, np.ndarray] = self._getValidFrame()
        while validFrame[0] and np.all(np.equal(self.lastFrame, validFrame[1])):
            validFrame = self._getValidFrame()
        if validFrame[0]:
            self.lastFrame = validFrame[1]
        return validFrame[0]

    def saveFrames(self, tryNum: int, frameCount: int) -> None:
        if frameCount < 1:
            return
        if tryNum < 0:
            raise ValueError('tryNum must be greater than or equal to 0.')
        result: tuple[bool, np.ndarray] = self._getValidFrame()
        if not result[0]:
            raise ValueError('Could not get an initial frame.')
        savePath = os.path.join(self.baseImageSavePath, '{:02d}-'.format(tryNum) + '{:03d}.png')
        self.lastFrame: np.ndarray = result[1]
        frames = [self.lastFrame]
        for i in range(1, frameCount):
            result = self.getNextFrame()
            if result:
                frames.append(self.lastFrame)
        for i, f in enumerate(frames):
            Image.fromarray(f).save(savePath.format(i + 1))

if __name__ == '__main__':
    fs = FrameSaver('rtsp://192.168.137.3:8554/unicast', '.')
    fs.saveFrames(1, 999)
