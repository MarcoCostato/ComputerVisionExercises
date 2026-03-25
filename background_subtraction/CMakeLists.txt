cmake_minimum_required(VERSION 3.10.0)
project(simple_segmentation VERSION 0.1.0 LANGUAGES C CXX)

set(OpenCV_DIR "$ENV{HOME/OpenCV/build}")
find_package(OpenCV REQUIRED)
add_executable(simple_segmentation main.cpp)

target_include_directories( simple_segmentation PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries( simple_segmentation PRIVATE ${OpenCV_LIBS})
