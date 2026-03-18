#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;



Mat computeHistImageBW8bit(Mat inputImage){
    int histSize = 256;
    float range[] = {0,256};
    const float* histRange[] = {range};
    bool uniform = true, accumulate = false;
    Mat hist;
    calcHist(&inputImage, 1, 0, Mat(), hist, 1, &histSize, histRange, uniform, accumulate);
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat outputImage(hist_h, hist_w, CV_8UC3, Scalar(30,30,30));
    normalize(hist, hist, 0, outputImage.rows, NORM_MINMAX);
    for (int i = 0; i < histSize; i++){
        line(outputImage,
        Point(bin_w * (i-1), hist_h - cvRound(hist.at<float>(i-1))),
        Point(bin_w * i, hist_h - cvRound(hist.at<float>(i))),
        Scalar(0,0,255), 2);
    }

    return outputImage;
}

Mat computeHistImageBRG8bit(Mat inputImage){
    std::vector<Mat> bgr_planes;
    split(inputImage, bgr_planes);
    int histSize = 256;
    float range[] = {0,256};
    const float* histRange[] = {range};
    bool uniform = true, accumulate = false;
    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat outputImage(hist_h, hist_w, CV_8UC3, Scalar(30,30,30));
    normalize(b_hist, b_hist, 0, outputImage.rows, NORM_MINMAX);
    normalize(g_hist, g_hist, 0, outputImage.rows, NORM_MINMAX);
    normalize(r_hist, r_hist, 0, outputImage.rows, NORM_MINMAX);
    for (int i = 1; i < histSize; i++){
        line(outputImage,
        Point(bin_w * (i-1), hist_h - cvRound(b_hist.at<float>(i-1))),
        Point(bin_w * i, hist_h - cvRound(b_hist.at<float>(i))),
        Scalar(255,0,0), 2);
        line(outputImage,
        Point(bin_w * (i-1), hist_h - cvRound(g_hist.at<float>(i-1))),
        Point(bin_w * i, hist_h - cvRound(g_hist.at<float>(i))),
        Scalar(0,255,0), 2);
        line(outputImage,
        Point(bin_w * (i-1), hist_h - cvRound(r_hist.at<float>(i-1))),
        Point(bin_w * i, hist_h - cvRound(r_hist.at<float>(i))),
        Scalar(0,0,255), 2);
    }

    return outputImage;
}

int main(int, char**){

    // Read the image
    Mat image, imageResized;
    image = imread("/home/marco/Desktop/ComputerVisionExercises/simple_segmentation/Coins.jpg");
    resize(image, imageResized, Size(256, 256));
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", imageResized);
    // BW conversion and histogram
    Mat bwImage, thresholdImage;
    cvtColor(imageResized, bwImage, COLOR_BGR2GRAY);
    imshow("BW Image", bwImage);
    Mat histImage = computeHistImageBW8bit(bwImage);

    imshow("Histogram", histImage);
    // BW histogram has a single peak, compute BGR histogram
    Mat histImageBRG = computeHistImageBRG8bit(imageResized);
    imshow("Histogram BRG", histImageBRG);
    // Blue channel has two peaks, show blue channel
    Mat blueChannel;
    std::vector<Mat> bgr_planes;
    split(imageResized, bgr_planes);
    blueChannel = bgr_planes[0];
    imshow("Blue Channel", blueChannel);
    // compute Otsu thresholding on blue channel
    threshold(blueChannel, thresholdImage, 0, 255, THRESH_BINARY + THRESH_OTSU);
    imshow("Threshold Image", thresholdImage);
    // Lighting causes problems, let's try HSV tresholding instead.

    //Convert to HSV
    Mat hsvImage;
    cvtColor(imageResized, hsvImage, COLOR_BGR2HSV);
    imshow("HSV Image", hsvImage);
    //Let's use a trackbar to find the best thresholds for the hsv channels
    int hMin = 24, hMax = 42, sMin = 0, sMax = 93, vMin = 54, vMax = 158;
    /*
    namedWindow("Trackbars", WINDOW_AUTOSIZE);
    createTrackbar("Hue Min", "Trackbars", &hMin, 179);
    createTrackbar("Hue Max", "Trackbars", &hMax, 179);
    createTrackbar("Sat Min", "Trackbars", &sMin, 255);
    createTrackbar("Sat Max", "Trackbars", &sMax, 255);
    createTrackbar("Val Min", "Trackbars", &vMin, 255);
    createTrackbar("Val Max", "Trackbars", &vMax, 255);

    while (true){
        Mat mask;
        inRange(hsvImage, Scalar(hMin, sMin, vMin), Scalar(hMax, sMax, vMax), mask);
        imshow("Mask", mask);
        if (waitKey(30) >= 0) break;
    }
    */
    inRange(hsvImage, Scalar(hMin, sMin, vMin), Scalar(hMax, sMax, vMax), thresholdImage);
    imshow("Threshold Image HSV", thresholdImage);

    // Remove noise with morphological operations
    Mat morphImage;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
    morphologyEx(thresholdImage, morphImage, MORPH_OPEN, kernel);
    imshow("Morphological Opening", morphImage);
    kernel = getStructuringElement(MORPH_ELLIPSE, Size(15,15));
    morphologyEx(morphImage, morphImage, MORPH_CLOSE, kernel);
    imshow("Morphological Closing", morphImage);

    // Display the final result over the original image
    Mat resultImage;
    bitwise_and(imageResized, imageResized, resultImage, morphImage);
    imshow("Result Image", resultImage);




    waitKey();


    return 0;
}