import numpy as np
import cv2

def sepia_filter(bgrImage):
    """Apply a sepia filter to a BGR image
    How it works:
    1. Create a sepia filter matrix that transforms the colors to a warm, brownish tone
    2. Apply the filter to the image using matrix multiplication
    3. Clip the resulting pixel values to the valid range [0, 255] and convert back to uint8
    """
    gray = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2GRAY)
    normalized_gray = np.array(gray, dtype=np.float32) / 255.0
    sepia = np.ones(bgrImage.shape)
    sepia[:,:,0] *= 153
    sepia[:,:,1] *= 204
    sepia[:,:,2] *= 255
    sepia[:,:,0] *= normalized_gray
    sepia[:,:,1] *= normalized_gray
    sepia[:,:,2] *= normalized_gray
    return np.array(sepia, dtype=np.uint8)
    
    return sepia_image