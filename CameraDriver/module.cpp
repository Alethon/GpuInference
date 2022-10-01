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
        .def(py::init<>())
        .def(py::init<const string&>())
        .def("GetNewFrame", &LowLatencyCapture::GetNewFrame)
        .def("Test", &LowLatencyCapture::Test);

// #ifdef VERSION_INFO
//     m.attr("__version__") = VERSION_INFO;
// #else
//     m.attr("__version__") = "dev";
// #endif
}


