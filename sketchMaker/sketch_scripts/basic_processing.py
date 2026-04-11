import cv2
import numpy as np

def sobel_edge_detection(bgrImage, blur_ksize=5):
    """Apply Sobel edge detection to a BGR image
    How it works:
    1. Convert the image to grayscale
    2. Apply Gaussian blur to reduce noise
    3. Apply Sobel operator in both x and y directions
    4. Combine the results to get the edge magnitude
    4. Convert back to BGR for consistency with other effects
    """
    gray = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    sobelxy = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize = 5)
    sobel = np.uint8(np.clip(sobelxy, 0, 255))
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

def canny_edge_detection(bgrImage, blur_ksize=5, low_threshold=50, high_threshold=150):
    """Apply Canny edge detection to a BGR image
    How it works:
    1. Convert the image to grayscale
    2. Apply Gaussian blur to reduce noise
    3. Apply Canny edge detection with specified thresholds
    4. Convert back to BGR for consistency with other effects
    """
    gray = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)