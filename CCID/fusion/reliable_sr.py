import cv2
import numpy as np


def downscale(img, sr):
    """Downscale an image."""
    dr = 1.0 / sr
    #assert img.shape[0] % sr == 0 and img.shape[1] % sr == 0
    return cv2.resize(img, None, fx=dr, fy=dr, interpolation=cv2.INTER_NEAREST)


def upscale(img, sr):
    """Upscale an image (with nearest neighbour filling)."""
    return cv2.resize(img, None, fx=sr, fy=sr, interpolation=cv2.INTER_NEAREST)


def _interpolation_filter(sr, interpolation):
    return lambda img: cv2.resize(downscale(img, sr), None, fx=sr, fy=sr, interpolation=interpolation)


def bilinear_interpolation_filter(sr):
    """Creates a bilinear interpolation filter."""
    return _interpolation_filter(sr, cv2.INTER_LINEAR)


def bicubic_interpolation_filter(sr):
    """Creates a bicubic interpolation filter."""
    return _interpolation_filter(sr, cv2.INTER_CUBIC)
