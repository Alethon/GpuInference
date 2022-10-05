# GpuInference

## CUDA Setup
Install CUDA 11.6.2 from the installer that can be downloaded from https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_511.65_windows.exe

## Python Setup
1. Install Python 3.10.6 using the installer that can be downloaded from https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
2. Install pytorch 1.2.1 using the pip command `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`

## Visual Studio Setup
1. Install Visual Studio
2. Add the featurepack "Desktop Development for C++"

## OpenCV Setup
1. Download OpenCV 4.6 from https://sourceforge.net/projects/opencvlibrary/files/4.6.0/opencv-4.6.0-vc14_vc15.exe/download
2. When prompted for a directory to extract to, extract to `C:\Program Files (x86)\libraries` (you will probably need to create the libraries folder)
3. Add `C:\Program Files (x86)\libraries\opencv\build\x64\vc15\bin` to PATH
