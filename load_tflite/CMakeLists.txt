cmake_minimum_required(VERSION 3.17)
project(TFLiteCheck)

set(CMAKE_CXX_STANDARD 14)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# include has 2 subdirectories: tensorflow and flatbuffers
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include/)

# lib has 1 file: libtensorflowlite.so
ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/linux_x86/libtensorflowlite.so)

add_executable(TFLiteCheck main.cpp)
target_link_libraries(TFLiteCheck tensorflowlite)
target_link_libraries(TFLiteCheck ${OpenCV_LIBRARIES})