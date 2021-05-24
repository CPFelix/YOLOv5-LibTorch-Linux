Edit from https://github.com/Nebula4869/YOLOv5-LibTorch.
# YOLOv5 LibTorch
Real time object detection with deployment of YOLOv5 through LibTorch C++ API With CUDA10.2.

### Environment

- Ubuntu 16.04
- OpenCV 3.2.0
- LibTorch 1.7.0
- CMake 3.20.0
- Nvidia Driver 460.67
- CUDA10.2

### Getting Started

1. Install OpenCV.

   ```shell
   sudo apt-get install libopencv-dev
   ```
   recommend using cmake to build opencv to specify the version. 

2. Download LibTorch.

   https://blog.csdn.net/weixin_43742643/article/details/114156298

3. Edit "CMakeLists.txt" to configure OpenCV and LibTorch correctly.

4. Test model download

truck detect model.
链接：https://pan.baidu.com/s/1PETg82SDpNb52Hjbh71BZQ 
提取码：36ax 

5. Compile and run.

   ```shell
   cd build
   cmake ..
   make
   ./../bin/YOLOv5LibTorch
   ```
