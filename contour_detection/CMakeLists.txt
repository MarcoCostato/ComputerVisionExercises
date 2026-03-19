cmake_minimum_required(VERSION 3.10.0)
project(contour_detection VERSION 0.1.0 LANGUAGES C CXX)

set(OpenCV_DIR "$ENV{HOME/OpenCV/build}")
find_package(OpenCV REQUIRED)
add_executable(contour_detection main.cpp)

target_include_directories( contour_detection PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries( contour_detection PRIVATE ${OpenCV_LIBS})
