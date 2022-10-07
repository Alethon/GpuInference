from CameraDriver import LowLatencyCapture
import cv2

rtsp = 0
frameCount = 100

if rtsp == 1:
    llc = LowLatencyCapture('rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4')
else:
    llc = LowLatencyCapture()

llc.Test(frameCount)