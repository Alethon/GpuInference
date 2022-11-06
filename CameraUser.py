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

def image_to_tensor(imageNp: ndarray, device: torch.device, size: tuple[int, int]) -> Tensor:
    image: Tensor = torch.from_numpy(imageNp)[None].cuda().float().permute(0, 3, 1, 2) / 255.0
    return F.interpolate(image, size=(size[1], size[0]), mode='bilinear').to(device)

class CameraUser(DarknetUser):
    def __init__(self, captureInfo, datasetInfoPath: str = os.path.join('cfg', 'obj.data'), weightFilePath: str = None, streamSize: tuple[int, int] = (1920, 1080), evaluationSize: tuple[int, int] = (416, 416)) -> None:
        super().__init__(datasetInfoPath)

        if weightFilePath is None:
            weightFilePath = self.latestWeightsPath
        
        self.loadCheckpoint(weightFilePath)
        self.model = self.model.eval()

        self.captureInfo = captureInfo
        self.streamSize: tuple[int, int] = streamSize
        self.capture: cv2.VideoCapture = None

        self.useImages: bool = True
        if self.useImages:
            imageDir = os.path.join('Lego', 'images')
            self.images = np.stack([cv2.imread(os.path.join(imageDir, f)) for f in os.listdir(imageDir)])

        self.evaluationSize: tuple[int, int] = evaluationSize

        self.showResults: bool = True
        self.windowName: str = 'Webcam Test'
        self.confThresh: float = 0.5
        self.nmsThresh: float = 0.5

        self.color = [[random.randint(0, 255) for _ in range(3)] for _ in range(self.classCount)]
    
    def __len__(self) -> int:
        return self.images.shape[0] if self.useImages else 0
    
    def __iter__(self):
        if self.useImages:
            self.count = -1
        else:
            self.capture = cv2.VideoCapture(self.captureInfo)
            # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.streamSize[0])
            # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.streamSize[1])
            # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # self.capture.set(cv2.CAP_PROP_FPS, 30)
        if self.showResults:
            cv2.namedWindow(self.windowName)
        return self
    
    def breakIteration(self):
        if not self.useImages:
            self.capture.release()
        if self.showResults:
            # print('Failed to get a new frame.')
            # print('Press any key to close the window...')
            # cv2.waitKey()
            cv2.destroyWindow(self.windowName)
        raise StopIteration
    
    def __next__(self) -> tuple[ndarray, Tensor, Tensor]:
        if self.useImages:
            self.count += 1
            success = (self.count < self.images.shape[0])
            frame = self.images[self.count % self.images.shape[0]].copy()
        else:
            success, frame = self.capture.read()

        if not success:
            self.breakIteration()
        
        frame[:, :, :] = frame[:, :, ::-1]
        
        with torch.no_grad():
            image: Tensor = image_to_tensor(frame, self.device, self.evaluationSize)
            predictions: Tensor = self.model(image)
            predictions = predictions[predictions[:, :, 4] > self.confThresh]
            if len(predictions) > 0:
                predictions = non_max_suppression(predictions.unsqueeze(0), self.confThresh, self.nmsThresh)[0]
                predictions[:, [0, 2]] *= 1.0 * frame.shape[1] / image.shape[3]
                predictions[:, [1, 3]] *= 1.0 * frame.shape[0] / image.shape[2]
                predictions[:, :4] = predictions[:, :4].round()
            else:
                predictions = torch.zeros((0, 7))

        if self.showResults:
            fc = frame.copy() # fc = frame_copy
            for x1, y1, x2, y2, conf, _, cls in predictions:
                plot_one_box([x1, y1, x2, y2], fc, color=self.color[int(cls)], label='{} {:.03f}'.format(self.names[int(cls)], conf))
            cv2.imshow(self.windowName, fc)
        
        if cv2.waitKey(1) == 27:
            self.breakIteration()

        return frame, image, predictions

if __name__ == '__main__':
    cu = CameraUser(0)
    t = time()
    for _, _, _ in cu:
        # print(1 / (time() - t))
        t = time()
        
