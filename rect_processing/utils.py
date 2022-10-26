import cv2
import numpy as np
from numpy import ndarray

import torch
import torch.nn.functional as F

WINDOW_NAME: str = '16 x 9 test'

def getWebcamCapture() -> cv2.VideoCapture:
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return capture

def get16x9(capture: cv2.VideoCapture) -> ndarray:
    return capture.read()[1]

def makeWindow() -> None:
    cv2.namedWindow(WINDOW_NAME)

def showImage(img: ndarray):
    cv2.imshow(WINDOW_NAME, img)

def scaleImage(img: ndarray, img_size: tuple[int, int]) -> ndarray:
    out = F.interpolate(torch.from_numpy(img).cuda().float().permute(2, 1, 0)[None], size=img_size, mode='bilinear')[0].permute(2, 1, 0)
    print(out.shape)
    out = out.type(torch.uint8)
    print(out.shape)
    out = out.cpu()
    print(out.shape)
    out = out.numpy()
    print(out.shape)
    return out

makeWindow()

cam = getWebcamCapture()

img0 = get16x9(cam)
print(img0.shape)
showImage(img0)
# cv2.waitKey()

img1 = scaleImage(img0, (1920 // 4, 1080 // 4))
showImage(img1)
cv2.waitKey()
