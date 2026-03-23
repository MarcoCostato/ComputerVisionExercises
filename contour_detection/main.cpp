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
    //GaussianBlur(bwImage, smoothImage, Size(5, 5), 0, 0);
    medianBlur(bwImage, smoothImage, 5);
    //bilateralFilter(bwImage, smoothImage, 9, 75, 75);
    imshow("Smoothed Image", smoothImage);
    threshold(smoothImage, binaryImage, 0, 255, THRESH_BINARY + THRESH_OTSU);
    // invert the binary image
    bitwise_not(binaryImage, binaryImage);
    imshow("Binary Image", binaryImage);
    // Find contours
    std::vector<std::vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    // Print the number of contours found
    cout << "Number of contours found: " << contours.size() << endl;
    // Draw contours on the original image with teir information
    Mat contourImage = image.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(contourImage, contours, (int)i, Scalar(0, 255, 0), 2);
        // Print contour information on the image
        Moments m = moments(contours[i]);
        int cx = (int)(m.m10 / m.m00);
        int cy = (int)(m.m01 / m.m00);
        putText(contourImage, "Contour " + to_string(i), Point(cx - 50, cy), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
    }
    imshow("Contours", contourImage);

    // Extract largest contour (area)
    double maxArea = 0;
    int maxAreaIndex = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxAreaIndex = (int)i;
        }
    }
    // Draw the largest contour on the original image
    Mat largestContourImage = image.clone();
    if (maxAreaIndex >= 0) {
        drawContours(largestContourImage, contours, maxAreaIndex, Scalar(0, 0, 255), 2);
    }
    imshow("Largest Contour", largestContourImage); 

    // Get the minimum bounding box of the countours
    Mat boundingBoxImage = image.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        RotatedRect boundingBox = minAreaRect(contours[i]);
        // Draw the bounding box on the image
        Point2f vertices[4];
        boundingBox.points(vertices);
        for (int j = 0; j < 4; j++) {
            line(boundingBoxImage, vertices[j], vertices[(j + 1) % 4], Scalar(255, 0, 255), 2);
        }
    }
    imshow("Bounding Boxes", boundingBoxImage);

    // Get convex hull of the countours
    Mat convexHullImage = image.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        vector<Point> hull;
        convexHull(contours[i], hull);
        // Draw the convex hull on the image
        for (size_t j = 0; j < hull.size(); j++) {
            line(convexHullImage, hull[j], hull[(j + 1) % hull.size()], Scalar(255, 255, 0), 2);
        }
    }
    imshow("Convex Hulls", convexHullImage);

    // Compute the moments of the contours
    Mat momentsImage = image.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        Moments m = moments(contours[i]);
        // compute the centroid of the contour
        int cx = (int)(m.m10 / m.m00);
        int cy = (int)(m.m01 / m.m00);
        // Draw the centroid on the image
        circle(momentsImage, Point(cx, cy), 5, Scalar(0, 255, 255), -1);
    }
    imshow("Moments", momentsImage);


    // Read and extract the contours of a new image using edge detection
    image = imread("/home/marco/Desktop/ComputerVisionExercises/contour_detection/sword.png");
    imshow("Sword Image", image);
    Mat grayImage, edges;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    GaussianBlur(grayImage, grayImage, Size(5, 5), 0);
    Canny(grayImage, edges, 50, 150);
    // Close the edges using morphological closing
    morphologyEx(edges, edges, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
    imshow("Closed Edges", edges);
    // Find contours
    contours.clear();
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    // Draw contours on the original image
    contourImage = image.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(contourImage, contours, (int)i, Scalar(0, 255, 0), 2);
    }
    //imshow("Sword Contours", contourImage);
    // Compute the elliptical fitting of the contours
    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() >= 5) { // fitEllipse requires at least 5 points
            RotatedRect fittedEllipse = fitEllipse(contours[i]);
            cout << "angle: " << fittedEllipse.angle << endl;
            // Draw the ellipse on the image
            ellipse(contourImage, fittedEllipse, Scalar(255, 0, 255), 2);
        }
    }
    imshow("Fitted Ellipses", contourImage);

    waitKey(10000);
    return 0;
}