#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // Read the image file
    Mat image = imread("/home/marco/Desktop/ComputerVisionExercises/contour_detection/shapes.jpeg");
    imshow("Original Image", image);
    Mat bwImage;
    cvtColor(image, bwImage, COLOR_BGR2GRAY);
    imshow("Grayscale Image", bwImage);
    // Otsu binarization
    Mat smoothImage, binaryImage;
    GaussianBlur(bwImage, smoothImage, Size(3, 3), 0, 0);
    threshold(smoothImage, binaryImage, 0, 255, THRESH_BINARY + THRESH_OTSU);
    // invert the binary image
    bitwise_not(binaryImage, binaryImage);
    imshow("Binary Image", binaryImage);
    // Find contours
    std::vector<std::vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // Print the number of contours found
    cout << "Number of contours found: " << contours.size() << endl;
    // Draw contours on the original image
    Mat contourImage = image.clone();
    drawContours(contourImage, contours, -1, Scalar(0, 255, 0), 2);
    imshow("Contours", contourImage);

    waitKey(0);
    return 0;
}