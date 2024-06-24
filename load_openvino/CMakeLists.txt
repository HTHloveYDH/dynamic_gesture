cmake_minimum_required(VERSION 3.17)
project(OpenVinoCheck)

set(CMAKE_CXX_STANDARD 14)

file(GLOB DYNAMIC_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/lib/intel64/libopenvino*.so)
file(GLOB STATIC_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/lib/intel64/lib*.a)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# include has 4 subdirectories: include, include/ie, include/ngraph and include/openvino
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include/ie 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ngraph ${CMAKE_CURRENT_SOURCE_DIR}/include/openvino
)

add_executable(OpenVinoCheck main.cpp)
target_link_libraries(OpenVinoCheck ${DYNAMIC_LIBS})
target_link_libraries(OpenVinoCheck ${STATIC_LIBS})
target_link_libraries(OpenVinoCheck ${OpenCV_LIBRARIES})