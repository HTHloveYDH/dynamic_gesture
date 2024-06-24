cmake_minimum_required(VERSION 3.5)
PROJECT(dynamic_gesture VERSION 1.0.0 LANGUAGES C CXX CUDA)
set(CMAKE_BUILD_TYPE Debug)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/common.cmake)

# 源文件
file(GLOB CPPS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${SAMPLES_COMMON_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../util/*.cpp)
find_package(OpenCV REQUIRED)

include_directories(${INCLUDE_DRIS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
# include_directories(/usr/local/include/eigen3)
# include_directories(/opt/eigen-3.4.0)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../util/eigen-3.4.0)
add_executable(${PROJECT_NAME} ${CPPS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS} ${OpenCV_LIBRARIES})