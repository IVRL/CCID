import cv2
import numpy as np


def gaussian_blur_filter(kernel_size, sigma=None):
    """Creates a gaussian blur filter."""
    return lambda input_image: cv2.GaussianBlur(input_image, ksize=(kernel_size, kernel_size), sigmaX=sigma)


def median_filter(kernel_size):
    """Creates a median filter (pixels are replaced by the median in the kernel window)."""
    return lambda input_image: cv2.medianBlur(input_image, kernel_size)


def bilateral_filter(diameter=9, sigma_color=75, sigma_space=75):
    """Creates a bilateral filter (a advanced gaussian filter method - can be slow)."""
    return lambda input_image: cv2.bilateralFilter(input_image, diameter, sigma_color, sigma_space)


def non_local_means_filter(h=30.0, template_window=7, search_window=21):
    """Create a NL means filter (weighted average of pixels, preserves sharpness)."""
    def non_local_means(input_image):
        uint8_input = (input_image * 255.0).astype(np.uint8)
        uint8_output = cv2.fastNlMeansDenoising(uint8_input, h=h, templateWindowSize=template_window, searchWindowSize=search_window)
        return uint8_output.astype(np.float32) / 255.0
    return non_local_means
