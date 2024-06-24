cmake_minimum_required(VERSION 3.5)
PROJECT(sampleOnnxMNIST VERSION 1.0.0 LANGUAGES C CXX CUDA)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/common.cmake)

# 源文件
file(GLOB CPPS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${SAMPLES_COMMON_DIR}/*.cpp)

message(STATUS "ALL_LIBS:${ALL_LIBS}")
message(STATUS "INCLUDE_DRIS:${INCLUDE_DRIS}")

include_directories(${INCLUDE_DRIS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
add_executable(${PROJECT_NAME} ${CPPS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS})