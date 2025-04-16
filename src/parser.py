import cv2
import numpy as np

def load_binary_image(path):
    """
    Loads a grayscale image from the given path and converts it into a binary image.
    Args:
        path (str): File path to the grayscale image.

    Returns:
        np.ndarray: Binary image with pixel values 0 or 1.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    
    #Convert to binary image
    binary_img = (img > 127).astype(np.uint8)
    return binary_img

def load_color_image(path):
    """
    Loads a color image and converts it from BGR to RGB format.

    Args:
        path (str): File path to the color image.

    Returns:
        np.ndarray: RGB image.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb