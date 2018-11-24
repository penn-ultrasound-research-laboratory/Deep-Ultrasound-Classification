import unittest

import src.utilities.imageUtilities as util
import numpy as np

from src.constants.ultrasoundConstants import IMAGE_TYPE

class Test_TestDetermineImageType(unittest.TestCase):
    def test_below_percentage_threshold_grayscale(self):
        image = np.ones((4,4,3))
        color_threshold = 0.05
        self.assertEqual(util.determine_image_type(image, color_threshold), IMAGE_TYPE.GRAYSCALE)


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
        self.assertFalse(util.crop_in_bounds(NATIVE_SHAPE, TARGET_SHAPE, OFFSET))

    def test_multiple_dim_exceed(self):
        NATIVE_SHAPE = (100, 200)
        TARGET_SHAPE = (80, 201)
        OFFSET = (25, 0)
        self.assertFalse(util.crop_in_bounds(NATIVE_SHAPE, TARGET_SHAPE, OFFSET))

    def test_no_dim_exceed(self):
        NATIVE_SHAPE = (100, 200)
        TARGET_SHAPE = (80, 160)
        OFFSET = (0, 0)
        self.assertTrue(util.crop_in_bounds(NATIVE_SHAPE, TARGET_SHAPE, OFFSET))


class Test_TestApplyCrop(unittest.TestCase):
    def test_single_channel(self):
        CROP = (10, 10, 50, 40)
        _, _, CROP_HEIGHT, CROP_WIDTH = CROP
        mock_image = np.zeros((500, 400))
        cropped_image = util.apply_crop(mock_image, CROP)
        self.assertEqual(cropped_image.shape, (CROP_HEIGHT, CROP_WIDTH))

    def test_three_channels(self):
        CROP = (10, 10, 50, 40)
        Y_OFFSET, X_OFFSET, CROP_HEIGHT, CROP_WIDTH = CROP
        mock_image = np.zeros((500, 400, 3))
        cropped_image = util.apply_crop(mock_image, CROP)
        self.assertEqual(cropped_image.shape, (CROP_HEIGHT, CROP_WIDTH, 3))


if __name__ == '__main__':
    unittest.main()