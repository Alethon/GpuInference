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
        # self.capture: cv2.VideoCapture = None
        self.camera: Camera = None
    
    def __iter__(self):
        # self.capture = cv2.VideoCapture(self.captureInfo)

        self.camera = Camera(self.captureInfo)
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.streamSize[0])
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.streamSize[1])
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        return super().__iter__()
    
    def breakIteration(self):
        # self.capture.release()
        self.camera.end()
        super().breakIteration()
    
    def __len__(self) -> int:
        return 0
    
    def __next__(self) -> tuple[ndarray, Tensor, Tensor]:
        # success, frame = self.capture.read()
        # print(success, frame)
        # if not success:
        #     self.breakIteration()
        frame = self.camera.get_frame()
        # frame[:, :, :] = frame[:, :, ::-1]
        image, predictions = self._getPredictions(frame)
        # self._showPredictions(frame, predictions)
        return frame, image, predictions

class Camera:
    def __init__(self,rtsp_url):        
        #load pipe for data transmittion to the process
        self.parent_conn, child_conn = mp.Pipe()
        #load process
        self.p = mp.Process(target=self.update, args=(child_conn,rtsp_url))        
        #start process
        self.p.daemon = True
        self.p.start()
        
    def end(self):
        self.parent_conn.send(2)
        
    def update(self,conn,rtsp_url):
        print("Cam Loading...")
        # cap = cv2.VideoCapture(rtsp_url,cv2.CAP_FFMPEG)   
        cap = cv2.VideoCapture(rtsp_url)
        print("Cam Loaded...")
        run = True
        
        while run:
            
            #grab frames from the buffer
            ret,frame = cap.read()
            
            #recieve input data
            rec_dat = conn.recv()
            
            if rec_dat == 1:
                #if frame requested
                if ret:
                    conn.send(frame)
            elif rec_dat ==2:
                #if close requested
                cap.release()
                run = False
                
        print("Camera Connection Closed")        
        conn.close()
    
    def get_frame(self):
        ###used to grab frames from the cam connection process
        
        ##[resize] param : % of size reduction or increase i.e 0.65 for 35% reduction  or 1.5 for a 50% increase
             
        #send request
        self.parent_conn.send(1)
        frame = self.parent_conn.recv()
        
        #reset request 
        self.parent_conn.send(0)

        return frame

if __name__ == '__main__':
    # cu = CameraUser(0)
    cu = CameraUser('rtsp://192.168.137.3:8554/unicast')
    for frame, image, predictions in cu:
        continue
        
