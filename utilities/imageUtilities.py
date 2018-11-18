import argparse
import cv2
import logging
import uuid
import numpy as np
from constants.ultrasoundConstants import IMAGE_TYPE
from matplotlib import pyplot


def determine_image_type(bgr_image):
    """Determines image type (Grayscale/Color) of image

    Arguments:
        bgr_image                            Image loaded w/ BGR channels (IMREAD.COLOR)

    Returns:
        IMAGE_TYPE Enum object. Specifies either IMAGE_TYPE.GRAYSCALE of IMAGE_TYPE.COLOR

    Note:
        0.04 is an empirically determined constant. When we convert MOV -> MP4 --> PNG frames, 
        there is a small probability that a grayscale frame has some color bleeding. That is, a tiny segment of the 
        pixels will take on a grayish/brown tint. Empirically, this gave ~0.02-0.04 color percentage to the full image.
        We were basically getting false attribution of GRAYSCALE images to the COLOR image type enum because the
        threshold of 0.015 was too low. Increased to 0.04. Hopefully shouldn't create false attribution. 
        The color scale bar in true COLOR scans all but guarantees a percentage greater than 10%. 
    """
    b, g, r = cv2.split(bgr_image)
    equality_check = np.logical_and(np.logical_and(b == r, b == g), r == g)

    if 1.0 - (np.count_nonzero(equality_check) / equality_check.size) < 0.04:
        return IMAGE_TYPE.GRAYSCALE
    else:
        return IMAGE_TYPE.COLOR


def center_crop_to_target_shape(image, target_shape, origin=None):
    """Crop the center portion of an image to a target shape

    Arguments:
        image                               An image. Either single channel (grayscale) or multi-channel (color)
        target_shape                        Target shape of the image section to crop

    Optional:
        origin                              Tuple containing hardcoded origin (row_offset, column_offset). The "origin"
                                                is the upper-left corner of the cropped image relative to 
                                                the top left at (0,0).
    Returns:
        A cropped image if both the target width and target height are
        less than the actual width and actual height. 
        Else, returns the original image without cropping.

    https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    """
    is_multi_channel = len(image.shape) == 3

    original_height, original_width = image.shape[:2] if is_multi_channel else image.shape
    target_height, target_width = target_shape[:2] if is_multi_channel else target_shape

    if target_width > original_width or target_height > original_height:
        return image

    column_offset = (
        (original_width // 2) - (target_width // 2) if origin is None
        else origin[1])
        
    row_offset = (
        (original_height // 2) - (target_height // 2) if origin is None 
        else origin[0])
   
    return image[
        row_offset: row_offset + target_height,
        column_offset: column_offset + target_width]


def center_crop_to_target_percentage(image, height_fraction, width_fraction):
    """Crop the center portion of an image to a target shape

    Arguments:
        image                               An image. Either single channel (grayscale) or multi-channel (color)
        height_fraction                     Target height fraction of the image to crop. 0 > arg >= 1 
                                                (e.g. 0.95 for 95%)
        width_fraction                      Target width fraction of the image to crop. 0 > arg >= 1
                                                (e.g. 0.95 for 95%)
    Returns:
        A cropped image if both the target width and target height are
        less than or equal to 1. Else, returns the original image without cropping.
        Additionally, returns the cropping bounds as a tuple. 
    """
    if height_fraction <= 0 or height_fraction > 1 or width_fraction <= 0 or width_fraction > 1:
        return image

    is_multi_channel = len(image.shape) == 3

    original_height, original_width = image.shape[:2] if is_multi_channel else image.shape

    height_divisor = 1 / height_fraction
    width_divisor = 1 / width_fraction

    height_remainder = original_height - (original_height // height_divisor)
    width_remainder = original_width - (original_width // width_divisor)

    top_crop = int(height_remainder // 2)
    bottom_crop = int(height_divisor - top_crop)
    left_crop = int(width_remainder // 2)
    right_crop = int(width_remainder - left_crop)

    cropped_slice = image[top_crop: original_height - bottom_crop, left_crop: original_width - right_crop]
    cropping_bounds = (top_crop, bottom_crop, left_crop, right_crop)

    return cropped_slice, cropping_bounds


def center_crop_to_target_pixel_boundary(image, height_pixel_boundary, width_pixel_boundary):
    """Crop the center portion of an image to a target shape

    Arguments:
        image                               An image. Either single channel (grayscale) or multi-channel (color)
        height_pixel_boundary               Target height pixel boundary of the image to crop. arg > 0 
                                                (e.g. 3 for 3px)
        width_pixel_boundary                Target width pixel boundary of the image to crop. arg > 0
                                                (e.g. 3 for 3px)
    Returns:
        A cropped image if both the target width and target height are
        greater than 0. Else, returns the original image without cropping.
        Additionally, returns the cropping bounds as a tuple. 
    """
    if height_pixel_boundary < 0 or width_pixel_boundary< 0:
        return image

    is_multi_channel = len(image.shape) == 3

    original_height, original_width = image.shape[:2] if is_multi_channel else image.shape

    cropped_slice = image[
        height_pixel_boundary: original_height - height_pixel_boundary,
        width_pixel_boundary: original_width - width_pixel_boundary
    ]

    cropping_bounds = (
        height_pixel_boundary,
        height_pixel_boundary,
        width_pixel_boundary,
        width_pixel_boundary)

    return cropped_slice, cropping_bounds


def image_random_sampling_batch(
    image, 
    target_shape = None, 
    batch_size = 16,
    use_min_dimension = False,
    upscale_to_target = False,
    upscale_method = cv2.INTER_CUBIC,
    always_sample_center=False):
    """Randomly sample an image to produce sample batch 

    Arguments:
        image                                Image to sample in channels_last format

    Optional:
        target_shape                         np.array containing output shape of each image sample. Must be square.
        batch_size                           Number of sample to generate in image batch

        use_min_dimension                    Boolean indicating to use the minimum shape dimension as the cropping
                                                 dimension. Must be True if target_shape is None. Will override 
                                                 target_shape regardless of shape value.
        
        upscale_to_target                    Upscale the image so that image dimensions >= target_shape before sampling
                                                target_shape must be defined to use upscale_to_target
        
        upscale_method                       Interpolation method to used. Default cv2.INTER_CUBIC

    Returns:
        4D array containing sampled images in axis=0. 

    Raises: 
        ValueError: the target_shape is greater than the actual image shape in at least one dimension
    """
    try:
        
        if target_shape is None:
            if use_min_dimension is False:
                raise ValueError("Use minimum dimension must be True with no target shape specified")
            if upscale_to_target:
                raise TypeError("If upscale_to_target is True, target_shape must be defined")
        else:
            if target_shape[0] != target_shape[1]:
                raise ValueError("Target Shape must be a square. E.g. [200, 200]")

            # TODO: should raise an error if upscale_to_target is False and the passed in image is smaller
            # in any dimension than the target shape

        if upscale_to_target:
            minimum_dimension = np.min(image.shape[:2])
            # Increase upscale ratio marginally to 
            # Assumption is to do a square upscale (fx=fy). The behavior of non-square interpolation is odd. 
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

        # If always_sample_center - always pull the same center cropped image to form the batch. 
        # Option likely used in conjunction with image data generator that will randomly transform 
        # samples of the image bath. Otherwise, randomly sample origins from origin range
        
        if always_sample_center:
            row_origins = [row_origin_max // 2] * batch_size
            column_origins = [column_origin_max // 2] * batch_size

        else:
            row_origins = (
                np.random.randint(0, row_origin_max, batch_size) if row_origin_max > 0
                else [0] * batch_size)

            column_origins = (
                np.random.randint(0, column_origin_max, batch_size) if column_origin_max > 0
                else [0] * batch_size)

        return np.stack(map(lambda sample_index: center_crop_to_target_shape(
            image,
            target_shape,
            origin=(row_origins[sample_index], column_origins[sample_index])),
            range(batch_size)), axis=0)
    
    except ValueError as e:
        raise ValueError(e)

    except TypeError as e:
        raise TypeError(e)


if __name__ == "__main__":

    batch_size=5
    elephant = cv2.imread("../TestImages/poorlyFocused.png", cv2.IMREAD_COLOR)

    random_batch = image_random_sampling_batch(
        elephant, 
        target_shape=[220, 220],
        upscale_to_target=True,
        batch_size=batch_size,
        always_sample_center=True)
    
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
