import argparse
import cv2
import logging
import uuid
import numpy as np
from src.constants.ultrasoundConstants import IMAGE_TYPE
from matplotlib import pyplot


def extract_height_width(image_shape):
    return image_shape[:2]


def crop_in_bounds(native_shape, target_shape, target_offset):
    return all(np.add(target_shape, target_offset) <= native_shape)


def apply_crop(image, crop_description):
    return image[
        crop_description[0]: crop_description[0] + crop_description[2],
        crop_description[1]: crop_description[1] + crop_description[3]
    ]


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
    b, g, r = cv2.split(bgr_image)
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


def image_random_sampling_batch(
        image,
        target_shape=None,
        batch_size=16,
        use_min_dimension=False,
        upscale_to_target=False,
        upscale_method=cv2.INTER_CUBIC,
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
                raise ValueError(
                    "Use minimum dimension must be True with no target shape specified")
            if upscale_to_target:
                raise TypeError(
                    "If upscale_to_target is True, target_shape must be defined")
        else:
            if target_shape[0] != target_shape[1]:
                raise ValueError(
                    "Target Shape must be a square. E.g. [200, 200]")

            # TODO: should raise an error if upscale_to_target is False and the passed in image is smaller
            # in any dimension than the target shape

        native_shape = extract_height_width(image.shape)
        native_min_dim = np.min(native_shape)

        if upscale_to_target:
            # Increase upscale ratio marginally to
            # Assumption is to do a square upscale (fx=fy). The behavior of non-square interpolation is odd.
            upscale_ratio = (target_shape[0] / native_min_dim) * 1.02
            image = cv2.resize(
                image,
                None,
                fx=upscale_ratio,
                fy=upscale_ratio,
                interpolation=upscale_method)
        else:
            # Use the minimum dimension if any dimension of image shape is less than the target shape
            if use_min_dimension is True or (
                    target_shape is not None and np.min(target_shape) > native_min_dim):
                target_shape = np.array([native_min_dim, native_min_dim])

        # Compute valid origin range. Fallback is "1" to support exclusive randint
        row_origin_max = max(image.shape[0] - target_shape[0], 1)
        column_origin_max = max(image.shape[1] - target_shape[1], 1)

        # If always_sample_center - always pull the same center cropped image to form the batch.
        # Option likely used in conjunction with image data generator that will randomly transform
        # samples of the image bath. Otherwise, randomly sample origins from origin range

        if always_sample_center:
            # Generate crop descriptions for center crop
            crop_descriptions = [center_crop_to_target_shape(image, target_shape)] * batch_size
        else:
            # Generate list of origins for image samples
            row_origins = np.random.randint(0, row_origin_max, batch_size)
            column_origins = np.random.randint(0, column_origin_max, batch_size)
            origins = zip(row_origins, column_origins)
            # Generate crop descriptions from list of origins
            crop_descriptions = [origin_crop_to_target_shape(image, target_shape, o) for o in origins]

        return np.stack(
            map(lambda idx: apply_crop(image, crop_descriptions[idx])),
            axis=0)

    except ValueError as e:
        raise ValueError(e)

    except TypeError as e:
        raise TypeError(e)


if __name__ == "__main__":

    batch_size = 5
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
