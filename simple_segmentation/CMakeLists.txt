cmake_minimum_required(VERSION 3.10.0)
project(test_opencv_cpp_project VERSION 0.1.0 LANGUAGES C CXX)

set(OpenCV_DIR "$ENV{HOME/OpenCV/build}")
find_package(OpenCV REQUIRED)
add_executable(test_opencv_cpp_project main.cpp)

target_include_directories( test_opencv_cpp_project PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries( test_opencv_cpp_project PRIVATE ${OpenCV_LIBS})
