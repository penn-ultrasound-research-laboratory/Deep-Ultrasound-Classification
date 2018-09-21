import argparse
import cv2
import logging
import uuid
import numpy as np
from constants.ultrasoundConstants import IMAGE_TYPE
from matplotlib import pyplot

def get_ROI(image):

    GAUSSIAN_KERNEL_SIZE_PAPER = 301
    CUTOFF_FREQ_PAPER = 30
    min_dim = np.min(image.shape[:2]) 
    
    if min_dim >= GAUSSIAN_KERNEL_SIZE_PAPER:
        ks = GAUSSIAN_KERNEL_SIZE_PAPER
    else:
        ks = min_dim

    sigma = 1 / (2 * np.pi * CUTOFF_FREQ_PAPER)

    blur = cv2.GaussianBlur(image, (ks, ks), sigma)

    return blur

if __name__ == "__main__":

    elephant = cv2.imread("../TestImages/poorlyFocused.png", cv2.IMREAD_COLOR)
    blurred = get_ROI(elephant)
    cv2.imshow("elephant", elephant)
    cv2.waitKey(0)

    cv2.imshow("blurred", blurred)
    cv2.waitKey(0)

