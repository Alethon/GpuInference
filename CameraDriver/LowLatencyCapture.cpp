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

LowLatencyCapture::LowLatencyCapture() {
    frameReady = CreateSemaphore(NULL, 0, 1, NULL);

    waitingForFrame = CreateSemaphore(NULL, 0, 1, NULL);

    frameLock = CreateMutex(NULL, true, NULL);

    capture = cv::VideoCapture(0);

    namedWindow(windowName, WINDOW_AUTOSIZE);
    cv::moveWindow(windowName, 0, 45);
    captureThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)FrameUpdateLoopCall, this, NULL, NULL);

    cv::Mat frame;
    for (int i = 0; i < 1000; i++) {
        cv::waitKey(20);
        GetNewFrame(frame);
        cv::imshow(windowName, frame);
    }
}

LowLatencyCapture::~LowLatencyCapture() {
    CloseHandle(frameReady);
    CloseHandle(waitingForFrame);
    CloseHandle(frameLock);
    WaitForSingleObject(captureThread, INFINITE);
    CloseHandle(captureThread);
    cv::destroyAllWindows();
}

bool LowLatencyCapture::GetNewFrame(cv::OutputArray image) {
    while (ReleaseSemaphore(waitingForFrame, 1, NULL) == 0) {
        cout << "Failed to release waitingForFrame: " << GetLastError() << endl;
        WaitForSingleObject(waitingForFrame, 0);
        //return false;
    }

    DWORD waitResult = WaitForSingleObject(frameReady, INFINITE);

    if (waitResult != WAIT_OBJECT_0) {
        cout << "Failed to wait for frameReady: " << GetLastError() << endl;
        return false;
    }

    currentFrame.copyTo(image);

    while (ReleaseSemaphore(waitingForFrame, 1, NULL) == 0) {
        cout << "Failed to release waitingForFrame: " << GetLastError() << endl;
        WaitForSingleObject(waitingForFrame, 0);
        //return false;
    }

    return true;
}

DWORD WINAPI LowLatencyCapture::FrameUpdateLoop() {
    cv::Mat tempFrame;
    DWORD waitResult;

    while(!capture.read(currentFrame));
    frameUpdated = true;
    ReleaseMutex(frameLock);
    cout << "success" << endl;

    while (1) {
        cout << "loop" << endl;
        waitResult = WaitForSingleObject(frameLock, INFINITY);

        switch (waitResult) {
        case WAIT_OBJECT_0:
            if (capture.read(tempFrame)) {
                if (!frameUpdated) {
                    frameUpdated = (cv::countNonZero(tempFrame != currentFrame) > 0);
                }

                if (frameUpdated) {
                    tempFrame.copyTo(currentFrame);
                }
            }
            break;

        case WAIT_TIMEOUT:
            if (frameUpdated) {
                if (ReleaseSemaphore(frameReady, 1, NULL) == 0) {
                    cout << "Failed to release frameReady: " << GetLastError() << endl;
                    return false;
                }

                waitResult = WaitForSingleObject(waitingForFrame, INFINITY);

                if (waitResult != WAIT_OBJECT_0) {
                    cout << "Failed to wait for waitingForFrame: " << waitResult << endl;
                    return false;
                }

                frameUpdated = false;
            }
            break;

        default:
            cout << "Failed to wait for waitingForFrame" << endl;
            return false;
        }

        ReleaseMutex(frameLock);
    }
    return true;
}