#include "LowLatencyCapture.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

DWORD WINAPI FrameUpdateLoopCall(LPVOID lpParam) {
    LowLatencyCapture* llcap = (LowLatencyCapture*)lpParam;
    return llcap->FrameUpdateLoop();
}

void LowLatencyCapture::Init() {
    frameReady = CreateSemaphore(NULL, 0, 1, NULL);
    frameLock = CreateMutex(NULL, false, NULL);
    captureThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)FrameUpdateLoopCall, this, NULL, NULL);
}

// for webcam testing
LowLatencyCapture::LowLatencyCapture() {
    capture = cv::VideoCapture(0);
    Init();
}

// for rtsp stream testing
LowLatencyCapture::LowLatencyCapture(const string& address) {
    capture = cv::VideoCapture(address);
    Init();
}

LowLatencyCapture::~LowLatencyCapture() {
    CloseHandle(frameReady);
    CloseHandle(frameLock);
    WaitForSingleObject(captureThread, INFINITE);
    CloseHandle(captureThread);
}

bool LowLatencyCapture::GetNewFrame(cv::OutputArray image) {
    // wait for a new frame to be ready
    DWORD waitResult = WaitForSingleObject(frameReady, INFINITE);
    if (waitResult != WAIT_OBJECT_0) {
        cout << "Failed to wait for frameReady: " << GetLastError() << endl;
        return false;
    }

    // lock the frame object
    waitResult = WaitForSingleObject(frameLock, INFINITE);
    if (waitResult != WAIT_OBJECT_0) {
        cout << "Failed to wait for frameLock: " << GetLastError() << endl;
        return false;
    }

    currentFrame.copyTo(image);

    // release the lock
    ReleaseMutex(frameLock);

    return true;
}

DWORD WINAPI LowLatencyCapture::FrameUpdateLoop() {
    cv::Mat tempFrame;
    vector<cv::Mat> rgb(3);
    DWORD waitResult;

    while (!capture.read(currentFrame));
    ReleaseSemaphore(frameReady, 1, NULL);

    while (1) {
        while (!capture.read(tempFrame));

        cv::split((tempFrame != currentFrame), rgb);

        if (cv::countNonZero(rgb[0]) + cv::countNonZero(rgb[1]) + cv::countNonZero(rgb[2]) > 0) {
            waitResult = WaitForSingleObject(frameLock, INFINITY);

            switch (waitResult) {
            case WAIT_OBJECT_0:
                tempFrame.copyTo(currentFrame);
                ReleaseSemaphore(frameReady, 1, NULL);
                break;

            default:
                return false;
            }

            ReleaseMutex(frameLock);
        }
    }
    return true;
}

void LowLatencyCapture::Test(int frameCount) {
    cv::Mat frame;

    namedWindow(windowName, WINDOW_AUTOSIZE);
    cv::moveWindow(windowName, 0, 45);

    // display frames
    for (int i = 0; i < frameCount; i++) {
        cv::waitKey(1);// required to give the display window rendering time
        if (!GetNewFrame(frame)) {
            cout << "This should never fail under webcam test. Something is wrong." << endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
            return;
        }
        cv::imshow(windowName, frame);
    }

    cv::waitKey(0);
    cv::destroyAllWindows();
}
