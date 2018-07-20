import argparse
import cv2
import uuid
import numpy as np
from constants.ultrasoundConstants import IMAGE_TYPE
from matplotlib import pyplot


def determine_image_type(bgr_image):
    b, g, r = cv2.split(bgr_image)
    equality_check = np.logical_and(np.logical_and(b == r, b == g), r == g)

    if 1.0 - (np.count_nonzero(equality_check) / equality_check.size) < 0.015:
        return IMAGE_TYPE.GRAYSCALE
    else:
        return IMAGE_TYPE.COLOR


def center_crop(image, target_shape, origin=None):
    """Crop the center portion of a color image

    Arguments:
        image: a color image encoded as colors_last (e.g. [224, 224, 3])
        target_shape: 
        origin: (optional) tuple containing origin (row_offset, column_offset)

    Returns:
        A cropped color image if both the width and height are
        less than the actual width and height. Else, returns the original 
        image without cropping.

    https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    """
    number_rows, number_cols = image.shape[:2]
    height, width = target_shape[:2]

    if width > number_cols or height > number_rows:
        return image

    column_offset = number_cols // 2 - \
        (width // 2) if origin is None else origin[1]
        
    row_offset = number_rows // 2 - \
        (height // 2) if origin is None else origin[0]

    return image[
        row_offset: row_offset + height,
        column_offset: column_offset + width]


def image_random_sampling_batch(image, target_shape, batch_size=16):
    """Randomly sample an image to produce sample batch 

    Arguments:
        image: image to sample
        target_shape: np.array containing image shape
        batch_size: (optional) number of sample to generate in image batch
    Returns:
        4D array containing sampled images in axis=0. 

    Raises: 
        ValueError: the target_shape is greater than the actual image shape in at least one dimension
    """

    try:       
        if np.max(target_shape) > np.max(image.shape):
            raise ValueError("Target shape exceeds input image by at least one dimension")

        # Compute valid origin range
        row_origin_max = image.shape[0] - target_shape[0]
        column_origin_max = image.shape[1] - target_shape[1]

        # Sample random origins from origin range
        row_origins = np.random.randint(0, row_origin_max, batch_size) if row_origin_max > 0 else [0] * batch_size
        column_origins = np.random.randint(0, column_origin_max, batch_size) if column_origin_max > 0 else [0] * batch_size

        return np.stack(map(lambda sample_index: center_crop(
            image,
            target_shape,
            origin=(row_origins[sample_index], column_origins[sample_index])),
            range(batch_size)), axis=0)
    
    except ValueError as e:
        raise ValueError(e)


if __name__ == "__main__":

    batch_size=10
    a = cv2.imread("models/elephant.jpg", cv2.IMREAD_COLOR)

    min_dim = np.min(a.shape[:2])
    print(min_dim)
    b = image_random_sampling_batch(a, [400, 400, 3], batch_size=batch_size)
    print(b.shape)

    for i in range(batch_size):
        cv2.imshow("mini", b[i])
        cv2.waitKey(0)
