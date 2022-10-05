from ObjectRecognition import *
from CameraDriver import LowLatencyCapture

class VideoInference:
    def __init__(self, connectionAddress: str) -> None:
        self.llc = LowLatencyCapture(connectionAddress)
    
    def GetNextFrame():
        pass
