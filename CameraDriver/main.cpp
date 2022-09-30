#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <process.h>
#include "LowLatencyCapture.h"

using namespace std;
using namespace cv;

void showImg() {
    cv::Mat img = cv::imread("D:/Pictures/meme/blushy.jpg");
    namedWindow("First OpenCV Application", WINDOW_AUTOSIZE);
    cv::imshow("First OpenCV Application", img);
    cv::moveWindow("First OpenCV Application", 0, 45);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    bool ll = true;
    if (ll) {
        LowLatencyCapture llc = LowLatencyCapture();
    }
    else {
        showImg();
    }
    return 0;
}
