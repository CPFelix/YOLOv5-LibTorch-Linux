# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/felix/Felix/YOLOv5-LibTorch-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/felix/Felix/YOLOv5-LibTorch-master/build

# Include any dependencies generated for this target.
include CMakeFiles/YOLOv5LibTorch.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/YOLOv5LibTorch.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/YOLOv5LibTorch.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/YOLOv5LibTorch.dir/flags.make

CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.o: CMakeFiles/YOLOv5LibTorch.dir/flags.make
CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.o: ../src/YOLOv5LibTorch.cpp
CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.o: CMakeFiles/YOLOv5LibTorch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/felix/Felix/YOLOv5-LibTorch-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.o -MF CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.o.d -o CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.o -c /home/felix/Felix/YOLOv5-LibTorch-master/src/YOLOv5LibTorch.cpp

CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/felix/Felix/YOLOv5-LibTorch-master/src/YOLOv5LibTorch.cpp > CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.i

CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/felix/Felix/YOLOv5-LibTorch-master/src/YOLOv5LibTorch.cpp -o CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.s

# Object files for target YOLOv5LibTorch
YOLOv5LibTorch_OBJECTS = \
"CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.o"

# External object files for target YOLOv5LibTorch
YOLOv5LibTorch_EXTERNAL_OBJECTS =

../bin/YOLOv5LibTorch: CMakeFiles/YOLOv5LibTorch.dir/src/YOLOv5LibTorch.cpp.o
../bin/YOLOv5LibTorch: CMakeFiles/YOLOv5LibTorch.dir/build.make
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_dnn.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_highgui.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_ml.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_objdetect.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_shape.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_stitching.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_superres.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_videostab.so.3.4.12
../bin/YOLOv5LibTorch: /home/felix/Felix/yolov5-linux/libtorch/lib/libtorch.so
../bin/YOLOv5LibTorch: /home/felix/Felix/yolov5-linux/libtorch/lib/libc10.so
../bin/YOLOv5LibTorch: /usr/local/cuda/lib64/stubs/libcuda.so
../bin/YOLOv5LibTorch: /usr/local/cuda/lib64/libnvrtc.so
../bin/YOLOv5LibTorch: /usr/local/cuda/lib64/libnvToolsExt.so
../bin/YOLOv5LibTorch: /usr/local/cuda/lib64/libcudart.so
../bin/YOLOv5LibTorch: /home/felix/Felix/yolov5-linux/libtorch/lib/libc10_cuda.so
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_calib3d.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_features2d.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_flann.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_photo.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_video.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_videoio.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_imgcodecs.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_imgproc.so.3.4.12
../bin/YOLOv5LibTorch: /usr/local/lib/libopencv_core.so.3.4.12
../bin/YOLOv5LibTorch: /home/felix/Felix/yolov5-linux/libtorch/lib/libc10_cuda.so
../bin/YOLOv5LibTorch: /home/felix/Felix/yolov5-linux/libtorch/lib/libc10.so
../bin/YOLOv5LibTorch: /usr/local/cuda/lib64/libcufft.so
../bin/YOLOv5LibTorch: /usr/local/cuda/lib64/libcurand.so
../bin/YOLOv5LibTorch: /usr/lib/x86_64-linux-gnu/libcublas.so
../bin/YOLOv5LibTorch: /usr/local/cuda/lib64/libcudnn.so
../bin/YOLOv5LibTorch: /usr/local/cuda/lib64/libnvToolsExt.so
../bin/YOLOv5LibTorch: /usr/local/cuda/lib64/libcudart.so
../bin/YOLOv5LibTorch: CMakeFiles/YOLOv5LibTorch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/felix/Felix/YOLOv5-LibTorch-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/YOLOv5LibTorch"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/YOLOv5LibTorch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/YOLOv5LibTorch.dir/build: ../bin/YOLOv5LibTorch
.PHONY : CMakeFiles/YOLOv5LibTorch.dir/build

CMakeFiles/YOLOv5LibTorch.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/YOLOv5LibTorch.dir/cmake_clean.cmake
.PHONY : CMakeFiles/YOLOv5LibTorch.dir/clean

CMakeFiles/YOLOv5LibTorch.dir/depend:
	cd /home/felix/Felix/YOLOv5-LibTorch-master/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/felix/Felix/YOLOv5-LibTorch-master /home/felix/Felix/YOLOv5-LibTorch-master /home/felix/Felix/YOLOv5-LibTorch-master/build /home/felix/Felix/YOLOv5-LibTorch-master/build /home/felix/Felix/YOLOv5-LibTorch-master/build/CMakeFiles/YOLOv5LibTorch.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/YOLOv5LibTorch.dir/depend

