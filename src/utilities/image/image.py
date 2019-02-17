import argparse
import logging
import uuid
import numpy as np

import tensorflow as tf

from constants.ultrasound import IMAGE_TYPE
from tensorflow.image import ResizeMethod


def extract_height_width(image_shape):
    return tuple(image_shape[:2])


def crop_in_bounds(native_shape, target_shape, target_offset=(0, 0)):
    return all(np.add(target_shape, target_offset) <= native_shape)


def apply_single_crop(image, crop_description):
    return image[
        crop_description[0]: crop_description[0] + crop_description[2],
        crop_description[1]: crop_description[1] + crop_description[3]
    ]


def apply_multiple_crops(image, crop_descriptions):
    return np.stack(map(lambda crop: apply_single_crop(image, crop), crop_descriptions), axis=0)


def determine_image_type(bgr_image, color_percentage_threshold=0.04):
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
    
    b = bgr_image[0]
    g = bgr_image[1]
    r = bgr_image[2]

    equality_check = np.logical_and(np.logical_and(b == r, b == g), r == g)

    if 1.0 - (np.count_nonzero(equality_check) / equality_check.size) < color_percentage_threshold:
        return IMAGE_TYPE.GRAYSCALE
    else:
        return IMAGE_TYPE.COLOR


def origin_crop_to_target_shape(image, target_shape, origin):
    """Best effort to crop an image to a target shape from fixed origin

    Arguments:
        image                               An image. Either single channel (grayscale) or multi-channel (color)
        target_shape                        Target shape of the image section to crop (height, width)

        origin                              Tuple containing hardcoded origin (row_offset, column_offset). The "origin"
                                                is the upper-left corner of the cropped image relative to
                                                the top left at (0,0).
    Returns:
        A crop description
    """
    native_shape = extract_height_width(image.shape)
    target_shape = extract_height_width(target_shape)

    if not crop_in_bounds(native_shape, target_shape, origin):
        return ((0, 0) + native_shape)

    return (origin + target_shape)


def center_crop_to_target_shape(image, target_shape):
    """Best effort to crop an image to a target shape from center origin

    Arguments:
        image                               An image. Either single channel (grayscale) or multi-channel (color)
        target_shape                        Target shape of the image section to crop (height, width)

    Returns:
        A crop description
    """
    native_shape = extract_height_width(image.shape)
    target_shape = extract_height_width(target_shape)

    offset = tuple(np.subtract(native_shape, target_shape) // 2)

    if not crop_in_bounds(native_shape, target_shape, offset):
        return ((0, 0) + native_shape)

    return (offset + target_shape)


def center_crop_to_target_percentage(image, height_fraction, width_fraction):
    """Crop the center portion of an image to a target percentage

    Arguments:
        image                               An image. Either single channel (grayscale) or multi-channel (color)
        height_fraction                     Target height fraction of the image to crop. 0 > arg >= 1
                                                (e.g. 0.95 for 95%)
        width_fraction                      Target width fraction of the image to crop. 0 > arg >= 1
                                                (e.g. 0.95 for 95%)
    Returns:
        A crop description
    """
    native_shape = extract_height_width(image.shape)

    if height_fraction <= 0 or height_fraction > 1 or width_fraction <= 0 or width_fraction > 1:
        return ((0, 0) + native_shape)

    divisors = np.reciprocal((height_fraction, width_fraction))
    target_shape = tuple(np.floor_divide(native_shape, divisors))
    offset = tuple(np.subtract(native_shape, target_shape) // 2)

    return (offset + target_shape)


def center_crop_to_target_padding(image, height_padding, width_padding):
    """Crop the center portion of an image to a target boundary padding

    Arguments:
        image                               An image. Either single channel (grayscale) or multi-channel (color)
        height_padding                      Target height padding of the image to crop. (e.g. 3 for 3px)
        width_padding                       Target width pixel boundary of the image to crop. (e.g. 3 for 3px)

    Returns:
        A crop description
    """
    native_shape = extract_height_width(image.shape)
    offset = (height_padding, width_padding)

    if height_padding < 0 or width_padding < 0 or any(offset >= np.floor_divide(native_shape, 2)):
        return ((0, 0) + native_shape)

    target_shape = tuple(np.subtract(native_shape, np.multiply(offset, 2)))

    return (offset + target_shape)


def sample_to_batch_center_origin(image, target_shape, batch_size):
    crop_descriptions = [center_crop_to_target_shape(image, target_shape)] * batch_size
    return apply_multiple_crops(image, crop_descriptions)


def sample_to_batch_random_origin(image, target_shape, batch_size):
    # Compute valid origin range. Fallback is "1" to support exclusive randint
    row_origin_max = max(image.shape[0] - target_shape[0], 1)
    column_origin_max = max(image.shape[1] - target_shape[1], 1)

    # Generate list of origins for image samples
    row_origins = np.random.randint(0, row_origin_max, batch_size)
    column_origins = np.random.randint(0, column_origin_max, batch_size)
    origins = zip(row_origins, column_origins)

    # Generate crop descriptions from list of origins
    crop_descriptions = [origin_crop_to_target_shape(image, target_shape, ox) for ox in origins]

    return apply_multiple_crops(image, crop_descriptions)


def sample_to_batch(
        image,
        batch_size=16,
        target_shape=None,
        use_min_dimension=False,
        upscale_to_target=False,
        interpolation_method=ResizeMethod.BICUBIC,
        always_sample_center=False):
    """Randomly sample an image to produce sample batch

    Arguments:
        image                               Image to sample in channels_last format

    Optional:
        target_shape                        np.array containing output shape of each image sample. Must be square.
        batch_size                          Number of sample to generate in image batch

        use_min_dimension                   Boolean indicating to use the minimum shape dimension as the cropping
                                                dimension. Must be True if target_shape is None. Will override
                                                target_shape regardless of shape value.

        upscale_to_target                   Upscale the image so that image dimensions >= target_shape before sampling
                                                target_shape must be defined to use upscale_to_target
                                                
        interpolation_method                Interpolation method to used. Default ResizeMethod.BICUBIC

    Returns:
        4D array containing sampled images in axis=0.

    Raises:
        ValueError: the target_shape is greater than the actual image shape in at least one dimension
    """

    native_shape = extract_height_width(tf.shape(image))
    native_min_dim = np.min(native_shape)
    target_shape = extract_height_width(target_shape) if (target_shape is not None) else None
    
    print("Native shape: {0}".format(native_shape))
    print("Native min dim: {0}".format(native_min_dim))
    print("target shape: {0}".format(target_shape))

    if target_shape is None:
        if not use_min_dimension:
            return ValueError(
                "Use minimum dimension must be True with no target shape specified")
        elif upscale_to_target:
            return TypeError(
                "If upscale_to_target is True, target_shape must be defined")
    else:
        if (not upscale_to_target and any(np.subtract(native_shape, target_shape) < 0)):
            raise ValueError(
                "If upscale_to_target is False, every dimension in native shape must be greater than target shape")

    if upscale_to_target:
        image = tf.image.resize_images(
            image,
            target_shape,
            method=interpolation_method
        )
        print(image)

    if use_min_dimension:
        target_shape = np.array([native_min_dim, native_min_dim])

    if (target_shape is not None and (np.min(target_shape) > native_min_dim)):
        target_shape = np.array([native_min_dim, native_min_dim])
        
    if always_sample_center:
        return sample_to_batch_center_origin(image, target_shape, batch_size)
    else:
        return sample_to_batch_random_origin(image, target_shape, batch_size)
