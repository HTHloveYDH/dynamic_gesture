# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

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
CMAKE_SOURCE_DIR = /data_ws/Data_1/tinghao/hand_pipe_2/load_tflite

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data_ws/Data_1/tinghao/hand_pipe_2/load_tflite/build

# Include any dependencies generated for this target.
include CMakeFiles/TFLiteCheck.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/TFLiteCheck.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/TFLiteCheck.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/TFLiteCheck.dir/flags.make

CMakeFiles/TFLiteCheck.dir/main.cpp.o: CMakeFiles/TFLiteCheck.dir/flags.make
CMakeFiles/TFLiteCheck.dir/main.cpp.o: /data_ws/Data_1/tinghao/hand_pipe_2/load_tflite/main.cpp
CMakeFiles/TFLiteCheck.dir/main.cpp.o: CMakeFiles/TFLiteCheck.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data_ws/Data_1/tinghao/hand_pipe_2/load_tflite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/TFLiteCheck.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/TFLiteCheck.dir/main.cpp.o -MF CMakeFiles/TFLiteCheck.dir/main.cpp.o.d -o CMakeFiles/TFLiteCheck.dir/main.cpp.o -c /data_ws/Data_1/tinghao/hand_pipe_2/load_tflite/main.cpp

CMakeFiles/TFLiteCheck.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TFLiteCheck.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data_ws/Data_1/tinghao/hand_pipe_2/load_tflite/main.cpp > CMakeFiles/TFLiteCheck.dir/main.cpp.i

CMakeFiles/TFLiteCheck.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TFLiteCheck.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data_ws/Data_1/tinghao/hand_pipe_2/load_tflite/main.cpp -o CMakeFiles/TFLiteCheck.dir/main.cpp.s

# Object files for target TFLiteCheck
TFLiteCheck_OBJECTS = \
"CMakeFiles/TFLiteCheck.dir/main.cpp.o"

# External object files for target TFLiteCheck
TFLiteCheck_EXTERNAL_OBJECTS =

TFLiteCheck: CMakeFiles/TFLiteCheck.dir/main.cpp.o
TFLiteCheck: CMakeFiles/TFLiteCheck.dir/build.make
TFLiteCheck: /data_ws/Data_1/tinghao/hand_pipe_2/load_tflite/lib/linux_x86/libtensorflowlite.so
TFLiteCheck: /usr/local/lib/libopencv_gapi.so.4.5.4
TFLiteCheck: /usr/local/lib/libopencv_highgui.so.4.5.4
TFLiteCheck: /usr/local/lib/libopencv_ml.so.4.5.4
TFLiteCheck: /usr/local/lib/libopencv_objdetect.so.4.5.4
TFLiteCheck: /usr/local/lib/libopencv_photo.so.4.5.4
TFLiteCheck: /usr/local/lib/libopencv_stitching.so.4.5.4
TFLiteCheck: /usr/local/lib/libopencv_video.so.4.5.4
TFLiteCheck: /usr/local/lib/libopencv_videoio.so.4.5.4
TFLiteCheck: /usr/local/lib/libopencv_imgcodecs.so.4.5.4
TFLiteCheck: /usr/local/lib/libopencv_dnn.so.4.5.4
TFLiteCheck: /usr/local/lib/libopencv_calib3d.so.4.5.4
TFLiteCheck: /usr/local/lib/libopencv_features2d.so.4.5.4
TFLiteCheck: /usr/local/lib/libopencv_flann.so.4.5.4
TFLiteCheck: /usr/local/lib/libopencv_imgproc.so.4.5.4
TFLiteCheck: /usr/local/lib/libopencv_core.so.4.5.4
TFLiteCheck: CMakeFiles/TFLiteCheck.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data_ws/Data_1/tinghao/hand_pipe_2/load_tflite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable TFLiteCheck"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TFLiteCheck.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/TFLiteCheck.dir/build: TFLiteCheck
.PHONY : CMakeFiles/TFLiteCheck.dir/build

CMakeFiles/TFLiteCheck.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/TFLiteCheck.dir/cmake_clean.cmake
.PHONY : CMakeFiles/TFLiteCheck.dir/clean

CMakeFiles/TFLiteCheck.dir/depend:
	cd /data_ws/Data_1/tinghao/hand_pipe_2/load_tflite/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data_ws/Data_1/tinghao/hand_pipe_2/load_tflite /data_ws/Data_1/tinghao/hand_pipe_2/load_tflite /data_ws/Data_1/tinghao/hand_pipe_2/load_tflite/build /data_ws/Data_1/tinghao/hand_pipe_2/load_tflite/build /data_ws/Data_1/tinghao/hand_pipe_2/load_tflite/build/CMakeFiles/TFLiteCheck.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/TFLiteCheck.dir/depend
