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
    def test_extract_height_width_from_two_dimensions(self):
        HEIGHT, WIDTH = (10, 15)
        image_shape = (HEIGHT, WIDTH)
        self.assertEqual(
            util.extract_height_width(image_shape),
            (HEIGHT, WIDTH)
        )

    def test_extract_height_width_from_three_dimensions(self):
        HEIGHT, WIDTH, DEPTH = (10, 15, 3)
        image_shape = (HEIGHT, WIDTH, DEPTH)
        self.assertEqual(
            util.extract_height_width(image_shape),
            (HEIGHT, WIDTH)
        )


if __name__ == '__main__':
    unittest.main()