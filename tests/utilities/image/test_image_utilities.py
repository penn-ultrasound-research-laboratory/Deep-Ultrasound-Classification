import unittest

import src.utilities.image.image as util
import numpy as np

from unittest.mock import MagicMock, ANY
from src.constants.ultrasound import IMAGE_TYPE


class Test_TestDetermineImageType(unittest.TestCase):
    def test_below_percentage_threshold_grayscale(self):
        image = np.ones((4, 4, 3))
        color_threshold = 0.05
        self.assertEqual(util.determine_image_type(
            image, color_threshold), IMAGE_TYPE.GRAYSCALE)


class Test_TestExtractHeightWidth(unittest.TestCase):
    def test_two_dimensions(self):
        HEIGHT, WIDTH = (10, 15)
        image_shape = (HEIGHT, WIDTH)
        self.assertEqual(
            util.extract_height_width(image_shape),
            (HEIGHT, WIDTH)
        )

    def test_three_dimensions(self):
        HEIGHT, WIDTH, DEPTH = (10, 15, 3)
        image_shape = (HEIGHT, WIDTH, DEPTH)
        self.assertEqual(
            util.extract_height_width(image_shape),
            (HEIGHT, WIDTH)
        )


class Test_TestCropInBounds(unittest.TestCase):
    def test_single_dim_exceed(self):
        NATIVE_SHAPE = (100, 200)
        TARGET_SHAPE = (80, 201)
        OFFSET = (0, 0)
        self.assertFalse(util.crop_in_bounds(
            NATIVE_SHAPE, TARGET_SHAPE, OFFSET))

    def test_multiple_dim_exceed(self):
        NATIVE_SHAPE = (100, 200)
        TARGET_SHAPE = (80, 201)
        OFFSET = (25, 0)
        self.assertFalse(util.crop_in_bounds(
            NATIVE_SHAPE, TARGET_SHAPE, OFFSET))

    def test_no_dim_exceed(self):
        NATIVE_SHAPE = (100, 200)
        TARGET_SHAPE = (80, 160)
        OFFSET = (0, 0)
        self.assertTrue(util.crop_in_bounds(
            NATIVE_SHAPE, TARGET_SHAPE, OFFSET))


class Test_TestApplySingleCrop(unittest.TestCase):
    def test_single_channel(self):
        CROP = (10, 10, 50, 40)
        _, _, CROP_HEIGHT, CROP_WIDTH = CROP
        mock_image = np.zeros((500, 400))
        cropped_image = util.apply_single_crop(mock_image, CROP)
        self.assertEqual(cropped_image.shape, (CROP_HEIGHT, CROP_WIDTH))

    def test_three_channels(self):
        CROP = (10, 10, 50, 40)
        Y_OFFSET, X_OFFSET, CROP_HEIGHT, CROP_WIDTH = CROP
        mock_image = np.zeros((500, 400, 3))
        cropped_image = util.apply_single_crop(mock_image, CROP)
        self.assertEqual(cropped_image.shape, (CROP_HEIGHT, CROP_WIDTH, 3))


class Test_TestOriginCropToTargetShape(unittest.TestCase):
    def test_successful_crop_origin(self):
        IMAGE_SHAPE = (500, 400, 3)
        ORIGIN = (200, 250)
        TARGET_SHAPE = (100, 100)
        mock_image = np.zeros(IMAGE_SHAPE)
        self.assertEqual(
            util.origin_crop_to_target_shape(mock_image, TARGET_SHAPE, ORIGIN),
            (ORIGIN + TARGET_SHAPE)
        )

    def test_reject_crop_origin(self):
        IMAGE_SHAPE = (500, 400)
        ORIGIN = (410, 250)
        TARGET_SHAPE = (100, 100)
        mock_image = np.zeros(IMAGE_SHAPE)
        self.assertEqual(
            util.origin_crop_to_target_shape(mock_image, TARGET_SHAPE, ORIGIN),
            ((0, 0) + IMAGE_SHAPE)
        )


class Test_TestCenterCropToTargetShape(unittest.TestCase):
    def test_successful_crop_center_shape(self):
        IMAGE_SHAPE = (500, 400, 3)
        TARGET_SHAPE = (100, 100)
        mock_image = np.zeros(IMAGE_SHAPE)
        self.assertEqual(
            util.center_crop_to_target_shape(mock_image, TARGET_SHAPE),
            ((200, 150) + TARGET_SHAPE)
        )

    def test_reject_crop_center_shape(self):
        IMAGE_SHAPE = (500, 400)
        TARGET_SHAPE = (600, 100)
        mock_image = np.zeros(IMAGE_SHAPE)
        self.assertEqual(
            util.center_crop_to_target_shape(mock_image, TARGET_SHAPE),
            ((0, 0) + IMAGE_SHAPE)
        )


class Test_TestCenterCropToTargetPercentage(unittest.TestCase):
    def test_successful_crop_center_percentage(self):
        IMAGE_SHAPE = (500, 400, 3)
        HEIGHT_FRACTION = 0.90
        WIDTH_FRACTION = 0.80
        mock_image = np.zeros(IMAGE_SHAPE)
        self.assertEqual(
            util.center_crop_to_target_percentage(
                mock_image, HEIGHT_FRACTION, WIDTH_FRACTION),
            ((25, 40) + (449, 320))
        )

    def test_reject_crop_greater_than_one_fraction(self):
        IMAGE_SHAPE = (500, 400)
        HEIGHT_FRACTION = 1.01
        WIDTH_FRACTION = 0.80
        mock_image = np.zeros(IMAGE_SHAPE)
        self.assertEqual(
            util.center_crop_to_target_percentage(
                mock_image, HEIGHT_FRACTION, WIDTH_FRACTION),
            ((0, 0) + IMAGE_SHAPE)
        )

    def test_reject_crop_greater_leq_zero_fraction(self):
        IMAGE_SHAPE = (500, 400)
        HEIGHT_FRACTION = 0.90
        WIDTH_FRACTION = 0.0
        mock_image = np.zeros(IMAGE_SHAPE)
        self.assertEqual(
            util.center_crop_to_target_percentage(
                mock_image, HEIGHT_FRACTION, WIDTH_FRACTION),
            ((0, 0) + IMAGE_SHAPE)
        )


class Test_TestCenterCropToTargetPadding(unittest.TestCase):
    def test_successful_crop_center_padding(self):
        IMAGE_SHAPE = (500, 400, 3)
        HEIGHT_PADDING = 10
        WIDTH_PADDING = 50
        mock_image = np.zeros(IMAGE_SHAPE)
        self.assertEqual(
            util.center_crop_to_target_padding(
                mock_image, HEIGHT_PADDING, WIDTH_PADDING),
            ((HEIGHT_PADDING, WIDTH_PADDING) + (480, 300))
        )

    def test_reject_crop_greater_leq_zero_padding(self):
        IMAGE_SHAPE = (500, 400)
        HEIGHT_PADDING = 3
        WIDTH_PADDING = -1
        mock_image = np.zeros(IMAGE_SHAPE)
        self.assertEqual(
            util.center_crop_to_target_padding(
                mock_image, HEIGHT_PADDING, WIDTH_PADDING),
            ((0, 0) + IMAGE_SHAPE)
        )

    def test_reject_crop_greater_than_half_boundary(self):
        IMAGE_SHAPE = (500, 400)
        HEIGHT_PADDING = 250
        WIDTH_PADDING = 10
        mock_image = np.zeros(IMAGE_SHAPE)
        self.assertEqual(
            util.center_crop_to_target_padding(
                mock_image, HEIGHT_PADDING, WIDTH_PADDING),
            ((0, 0) + IMAGE_SHAPE)
        )


class Test_TestUniformUpscaleToTargetShape(unittest.TestCase):

    def test_standard_arguments(self):
        # Setup mock on function
        applyImageUpscaleMock = MagicMock()
        util.apply_image_upscale = applyImageUpscaleMock
        # Setup mock image and args
        IMAGE_SHAPE = (94, 83)
        TARGET_SHAPE = (113, 90)
        SAFE_UPSCALE_RATIO = 1.02
        mock_image = np.zeros(IMAGE_SHAPE)

        expectedUpscaleFactor = max(np.divide(TARGET_SHAPE, IMAGE_SHAPE))

        util.uniform_upscale_to_target_shape(
            mock_image,
            TARGET_SHAPE,
            safe_upscale_ratio=SAFE_UPSCALE_RATIO)

        applyImageUpscaleMock.assert_called_once_with(
            mock_image,
            expectedUpscaleFactor * SAFE_UPSCALE_RATIO,
            expectedUpscaleFactor * SAFE_UPSCALE_RATIO
        )

class Test_TestSampleToBatchCenterOrigin(unittest.TestCase):

    def test_standard_arguments(self):
        centerCropToTargetShapeMock = MagicMock()
        applyMultipleCropsMock = MagicMock()

        util.center_crop_to_target_shape = centerCropToTargetShapeMock
        util.apply_multiple_crops = applyMultipleCropsMock

        # Setup mock image and args
        IMAGE_SHAPE = (94, 83)
        TARGET_SHAPE = (113, 90)
        BATCH_SIZE = 5
        mock_image = np.zeros(IMAGE_SHAPE)

        util.sample_to_batch_center_origin(mock_image, TARGET_SHAPE, BATCH_SIZE)

        # Should get the single crop description
        self.assertEqual(centerCropToTargetShapeMock.call_count, 1)
        # Apply multiple called once with repeated crop description
        applyMultipleCropsMock.assert_called_once_with(mock_image, ANY)
        args, _ = applyMultipleCropsMock.call_args_list[0]
        self.assertEqual(len(args[1]), BATCH_SIZE)

if __name__ == '__main__':
    unittest.main()