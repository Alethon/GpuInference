import os
import random
from abc import *

import cv2

import torch
import torch.nn.functional as F

from torch import Tensor
from numpy import ndarray

from DarknetUser import DarknetUser
from utils.utils import non_max_suppression, plot_one_box

def image_to_tensor(imageNp: ndarray, device: torch.device, size: tuple[int, int]) -> Tensor:
    image: Tensor = torch.from_numpy(imageNp)[None].cuda().float().permute(0, 3, 1, 2) / 255.0
    return F.interpolate(image, size=(size[1], size[0]), mode='bilinear').to(device)

class DetectionIterable(ABC):
    def __init__(self, datasetInfoPath: str = os.path.join('cfg', 'obj.data'), weightFilePath: str = None, evaluationSize: tuple[int, int] = (416, 416)) -> None:
        super().__init__()

        user: DarknetUser = DarknetUser(datasetInfoPath)

        if weightFilePath is None:
            weightFilePath = user.latestWeightsPath

        user.loadCheckpoint(weightFilePath)

        self.model = user.model
        self.model = self.model.eval()

        self.names = user.names
        self.device = user.device
        self.classCount = user.classCount

        self.evaluationSize: tuple[int, int] = evaluationSize
        self.confThresh: float = 0.5
        self.nmsThresh: float = 0.5
        
        self.showResults: bool = True
        self.realTimePlayback: bool = False
        self.windowName: str = 'Detection Test'
        self.color = [[random.randint(0, 255) for _ in range(3)] for _ in range(self.classCount)]
    
    def _getPredictions(self, frame: ndarray) -> tuple[Tensor, Tensor]:
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
        return image, predictions
    
    def _showPredictions(self, frame: ndarray, predictions: Tensor) -> None:
        if self.showResults:
            fc = frame.copy() # fc = frame_copy
            for x1, y1, x2, y2, conf, _, cls in predictions:
                plot_one_box([x1, y1, x2, y2], fc, color=self.color[int(cls)], label='{} {:.03f}'.format(self.names[int(cls)], conf))
            cv2.imshow(self.windowName, fc)
            if self.realTimePlayback:
                if cv2.waitKey(1) == 27:
                    self.breakIteration()
            else:
                if cv2.waitKey() == 27:
                    self.breakIteration()
    
    def breakIteration(self):
        if self.showResults:
            # print('Failed to get a new frame.')
            # print('Press any key to close the window...')
            # cv2.waitKey()
            cv2.destroyWindow(self.windowName)
        raise StopIteration
    
    def __iter__(self):
        if self.showResults:
            cv2.namedWindow(self.windowName)
        return self
    
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __next__(self) -> tuple[ndarray, Tensor, Tensor]:
        pass
