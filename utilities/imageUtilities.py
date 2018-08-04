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

    column_offset = (
        (number_cols // 2) - (width // 2) if origin is None
        else origin[1])
        
    row_offset = (
        (number_rows // 2) - (height // 2) if origin is None 
        else origin[0])
   
    return image[
        row_offset: row_offset + height,
        column_offset: column_offset + width]


def image_random_sampling_batch(
    image, 
    target_shape = None, 
    batch_size = 16,
    use_min_dimension = False,
    upscale_to_target = False,
    upscale_method = cv2.INTER_CUBIC):
    """Randomly sample an image to produce sample batch 

    Arguments:
        image: image to sample in channels_last format
        target_shape: (optional) np.array containing image shape. Must be square - e.g. [200, 200]
        batch_size: (optional) number of sample to generate in image batch
        use_min_dimension: (optional) boolean indicating to use the minimum shape dimension 
            as the cropping dimension. Must be True if target_shape is None. Will override 
            target_shape regardless of shape value.
        upscale_to_target: (optional) Upscale the image so that image dimensions >= target_shape before sampling
            target_shape must be defined to use upscale_to_target
        upscale_method: (optional) Interpolation method to used. Default cv2.INTER_CUBIC
    Returns:
        4D array containing sampled images in axis=0. 

    Raises: 
        ValueError: the target_shape is greater than the actual image shape in at least one dimension
    """
    try:

        if target_shape is not None and target_shape[0] != target_shape[1]:
            raise ValueError("Target Shape must be a square. E.g. [200, 200]")

        if target_shape is None and use_min_dimension is False:
            raise ValueError("Use minimum dimension must be True with no target shape specified")

        if target_shape is None and upscale_to_target:
            raise TypeError("If upscale_to_target is True, target_shape must be defined")

        if upscale_to_target:
            minimum_dimension = np.min(image.shape[:2])
            # Increase upscale ratio marginally to 
            upscale_ratio = (target_shape[0] / minimum_dimension) * 1.02
            image = cv2.resize(
                image, 
                None, 
                fx=upscale_ratio, 
                fy=upscale_ratio, 
                interpolation=upscale_method)
        else:
            # Use the minimum dimension if any dimension of image shape is less than the target shape
            if use_min_dimension is True or (
                target_shape is not None and np.min(target_shape) > np.min(image.shape[:2])):
                minimum_dimension = np.min(image.shape[:2])
                target_shape = np.array([minimum_dimension, minimum_dimension])

        # Compute valid origin range
        row_origin_max = image.shape[0] - target_shape[0]
        column_origin_max = image.shape[1] - target_shape[1]

        # Sample random origins from origin range
        row_origins = (
            np.random.randint(0, row_origin_max, batch_size) if row_origin_max > 0
            else [0] * batch_size)

        column_origins = (
            np.random.randint(0, column_origin_max, batch_size) if column_origin_max > 0
            else [0] * batch_size)

        return np.stack(map(lambda sample_index: center_crop(
            image,
            target_shape,
            origin=(row_origins[sample_index], column_origins[sample_index])),
            range(batch_size)), axis=0)
    
    except ValueError as e:
        raise ValueError(e)

    except TypeError as e:
        raise TypeError(e)

if __name__ == "__main__":

    batch_size=10
    elephant = cv2.imread("models/elephant.jpg", cv2.IMREAD_COLOR)

    random_batch = image_random_sampling_batch(
        elephant, 
        target_shape=[220, 220],
        upscale_to_target=True,
        batch_size=batch_size)
    
    for i in range(batch_size):
        cv2.imshow("mini", random_batch[i])
        cv2.waitKey(0)

    random_batch_max = image_random_sampling_batch(
        elephant, 
        use_min_dimension=True,
        batch_size=batch_size)

    for i in range(batch_size):
        cv2.imshow("mini", random_batch_max[i])
        cv2.waitKey(0)
