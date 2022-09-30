#include "LowLatencyCapture.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(CameraDriver, m) {
    py::class_<LowLatencyCapture>(m, "LowLatencyCapture")
        // .def(py::init<const string&>())
        .def(py::init<>())
        .def("GetNewFrame", &LowLatencyCapture::GetNewFrame)
        .def("Test", &LowLatencyCapture::Test);
    
    // py::class_<Hi>(m, "Hi")
    //     .def(py::init<>())
    //     .def("SayHi", &Hi::SayHi);

// #ifdef VERSION_INFO
//     m.attr("__version__") = VERSION_INFO;
// #else
//     m.attr("__version__") = "dev";
// #endif
}

// #include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
// #include <iostream>
// #include <process.h>
// #include "LowLatencyCapture.h"

// using namespace std;
// using namespace cv;

// void showImg() {
//     cv::Mat img = cv::imread("D:/Pictures/meme/blushy.jpg");
//     namedWindow("First OpenCV Application", WINDOW_AUTOSIZE);
//     cv::imshow("First OpenCV Application", img);
//     cv::moveWindow("First OpenCV Application", 0, 45);
//     cv::waitKey(0);
//     cv::destroyAllWindows();
// }

// //cv::VideoWriter 

// int main() {
//     bool ll = true;
//     if (ll) {
//         LowLatencyCapture llc = LowLatencyCapture();
//         llc.Test();
//     }
//     else {
//         showImg();
//     }
//     return 0;
// }

