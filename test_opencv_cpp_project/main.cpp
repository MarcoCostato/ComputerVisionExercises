#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
int main(int, char**){
    Mat image;
    image = imread("/home/marco/Desktop/ComputerVisionExercises/test_opencv_cpp_project/pharaoh.jpg");
    if( !image.data )
    {
        printf("No Image Data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", image);
    waitKey(0);
    return 0;

}
