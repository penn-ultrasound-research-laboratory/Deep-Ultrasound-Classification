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
    """

    Arguments:
        image:
        target_shape: np.array containing image shape

    Returns:

    Raises: 
        ValueError: the target_shape is greater than the actual image shape in at least one dimension

    """

    image_batch = np.empty(([batch_size] + target_shape))
    print(image_batch.shape)
    
    # Compute valid origin range
    row_origin_max = image.shape[0] - target_shape[0]
    column_origin_max = image.shape[1] - target_shape[1]
    print("Row max: {} | Column max: {}".format(row_origin_max, column_origin_max))

    # Sample random origins from origin range
    row_origins = np.random.randint(0, row_origin_max, batch_size)
    column_origins = np.random.randint(0, column_origin_max, batch_size)
    print("Row origins: {} | Column origins: {}".format(row_origins, column_origins))


    for sample_index in range(batch_size):
        image_batch[sample_index] = center_crop(
            image,
            target_shape,
            origin=(row_origins[sample_index], column_origins[sample_index]))
            
    return image_batch

if __name__ == '__main__':

    batch_size=10
    a = cv2.imread('models/elephant.jpg', cv2.IMREAD_COLOR)
    b = image_random_sampling_batch(a, [10, 10, 3], batch_size=batch_size)
    print(b.shape)

    cv2.imshow('mini', b[batch_size-1])
    cv2.waitKey(0)