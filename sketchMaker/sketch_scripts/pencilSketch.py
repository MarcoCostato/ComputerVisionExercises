import cv2

def pencil_sketch(bgrImage):
    """Convert a BGR image to a pencil sketch effect
    How it works:
    1. Convert the image to grayscale
    2. Invert the grayscale image
    3. Apply Gaussian blur to the inverted image
    4. Blend the original grayscale image with the blurred inverted image
    """
    gray = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    
    return sketch

def pencil_sketch(bgrImage, blur_ksize=21):
    """Convert a BGR image to a pencil sketch effect
    How it works:
    1. Convert the image to grayscale
    2. Invert the grayscale image
    3. Apply Gaussian blur to the inverted image
    4. Blend the original grayscale image with the blurred inverted image
    """
    gray = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (blur_ksize, blur_ksize), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    
    return sketch

def color_pencil_sketch(bgrImage, blur_ksize=21):
    """Convert a BGR image to a color pencil sketch effect
    How it works:
    1. Convert the image to grayscale
    2. Invert the grayscale image
    3. Apply Gaussian blur to the inverted image
    4. Blend the original grayscale image with the blurred inverted image to get the sketch
    5. Convert the sketch back to BGR and blend with the original color image to retain colors
    """
    gray = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (blur_ksize, blur_ksize), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    
    # Convert sketch back to BGR and blend with original color image
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    color_sketch = cv2.multiply(sketch_bgr, bgrImage, scale=1/255)
    
    return color_sketch