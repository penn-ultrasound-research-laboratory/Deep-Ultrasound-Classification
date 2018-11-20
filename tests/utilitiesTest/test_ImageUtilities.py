import unittest

import src.utilities.imageUtilities as util
import numpy as np

from src.constants.ultrasoundConstants import IMAGE_TYPE

class Test_TestIncrementDecrement(unittest.TestCase):
    def test_below_percentage_threshold_grayscale(self):
        
        image = np.ones((4,4,3))
        color_threshold = 0.05
        self.assertEqual(util.determine_image_type(image, color_threshold), IMAGE_TYPE.GRAYSCALE)

if __name__ == '__main__':
    unittest.main()