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
	LowLatencyCapture(const string&);
	~LowLatencyCapture();
	bool GetNewFrame(cv::OutputArray);
	DWORD WINAPI FrameUpdateLoop();
	void Test(int);

protected:
	void Init();
	cv::VideoCapture capture;
	cv::Mat currentFrame;
	HANDLE frameReady;// semaphore
	HANDLE frameLock;// mutex
	HANDLE captureThread;

private:
	const std::string& windowName = "LowLatencyCapture Test";
};
