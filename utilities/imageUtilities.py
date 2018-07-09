import argparse, cv2, uuid
import numpy as np
from constants.ultrasoundConstants import IMAGE_TYPE

def determine_image_type(bgr_image):
    b, g, r = cv2.split(bgr_image)
    equality_check = np.logical_and(np.logical_and(b==r, b==g), r==g)

    if 1.0 - (np.count_nonzero(equality_check) / equality_check.size) < 0.015:
        return IMAGE_TYPE.GRAYSCALE
    else:
        return IMAGE_TYPE.COLOR