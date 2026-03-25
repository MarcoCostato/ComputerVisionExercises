#include <iostream>
#include <opencv2/opencv.hpp>

int extractFrameNumber(const std::string& filename){
    // Extracts the frame number from the filename
    std::string digits;
    for (char c : filename){
        if (std::isdigit(c)){
            digits += c;
        }
    }
    return std::stoi(digits);
}

int main() {
    int kernelSize=5, varThreshold=24, sizeTreshold=100;
    // Read the video frames
    std::string videoPath = "/home/marco/Desktop/ComputerVisionExercises/background_subtraction/frames/JPEGS/peds";
    bool closeFlag = false;
    std::vector<cv::String> filenames;
    cv::glob(videoPath += "/*.jpg", filenames);
    // Sort the filenames based on the extracted frame numbers to ensure correct order
    // without this, frame 10 would come before frame 2, etc.
    std::sort(filenames.begin(), filenames.end(), [](const cv::String& a, const cv::String& b){
        return extractFrameNumber(a) < extractFrameNumber(b);
    });
    // Create background subtractor object
    cv::Ptr<cv::BackgroundSubtractor> pBackSub;
    while(true){
        // Create MOG2 background subtractor
        pBackSub = cv::createBackgroundSubtractorMOG2(500, varThreshold, false);
        for (const auto& filename : filenames) {
            if(kernelSize <= 0) kernelSize = 1; // Ensure kernel size is positive
            cv::Mat frame = cv::imread(filename);
            // Apply background subtraction to get the foreground mask
            cv::Mat fgMask;
            pBackSub->apply(frame, fgMask);
            // Morphological operations to remove noise
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize));
            cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(fgMask, fgMask, cv::MORPH_DILATE, kernel);
            // Compute the contours of the foreground mask
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            // Filter contours based on area to remove small noise
            for (const auto& contour : contours) {
                // Delete small contours (draw them black) that are likely noise 
                if (cv::contourArea(contour) < sizeTreshold) { 
                    cv::drawContours(fgMask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0), cv::FILLED);
                }
            }
            // Draw bounding boxes and center points (using moments) around the detected objects
            cv::Mat detectionFrame = frame.clone();
            for (const auto& contour : contours) {
                if (cv::contourArea(contour) < sizeTreshold) continue; // Skip small contours
                cv::Rect boundingBox = cv::boundingRect(contour);
                cv::rectangle(detectionFrame, boundingBox, cv::Scalar(0, 255, 0), 2);
                cv::Moments moments = cv::moments(contour);
                cv::circle(detectionFrame, cv::Point(static_cast<int>(moments.m10 / moments.m00), static_cast<int>(moments.m01 / moments.m00)), 5, cv::Scalar(0, 0, 255), -1);
            }



            // Use the foreground mask to extract the detected objects from the original frame
            cv::Mat detectedImage;           
            cv::bitwise_and(frame, frame, detectedImage, fgMask);
            
            cv::imshow("Original Video", frame);
            cv::imshow("Foreground Mask", fgMask);
            cv::imshow("Detected Image", detectedImage);
            cv::imshow("Detection Frame", detectionFrame);

            cv::namedWindow("VarBars", cv::WINDOW_AUTOSIZE);
            cv::createTrackbar("Kernel Size", "VarBars", &kernelSize, 20);
            cv::createTrackbar("Var Threshold", "VarBars", &varThreshold, 100);
            cv::createTrackbar("Size Threshold", "VarBars", &sizeTreshold, 500);
            //std::cout << "Processing frame: " << filename << std::endl;



            if (cv::waitKey(30) == 'q'){
                closeFlag = true;
                break;
            }
        }
        if (closeFlag) break;
    }


    cv::destroyAllWindows();
    return 0;
}