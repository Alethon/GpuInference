#pragma once

#include <Windows.h>
#include <process.h>
#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

class LowLatencyCapture {
public:
	LowLatencyCapture();
	~LowLatencyCapture();
	bool GetNewFrame(cv::OutputArray);
	DWORD WINAPI FrameUpdateLoop();
protected:
	const std::string& windowName = "First OpenCV Application";
	cv::VideoCapture capture;
	bool frameUpdated;
	cv::Mat currentFrame;
	// errors because semaphore resources are being released and waited by different threads
	HANDLE waitingForFrame;// semaphore
	HANDLE frameReady;// semaphore
	HANDLE frameLock;// mutex
	HANDLE captureThread;
};
