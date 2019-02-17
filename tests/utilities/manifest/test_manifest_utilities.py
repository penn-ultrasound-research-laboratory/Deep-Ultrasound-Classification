import unittest

import src.utilities.manifest.manifest as util

from src.constants.ultrasound import (
    FOCUS_HASH_LABEL,
    IMAGE_TYPE,
    IMAGE_TYPE_LABEL
)

from unittest.mock import MagicMock, ANY

class Test_FrameImageTypeMatch(unittest.TestCase):
    def test_image_all_true(self):
        frame = {}
        self.assertTrue(util.frame_image_type_match(frame, IMAGE_TYPE.ALL))

    def test_image_not_all_match_true(self):
        frame = { IMAGE_TYPE_LABEL: IMAGE_TYPE.GRAYSCALE.value }
        self.assertTrue(util.frame_image_type_match(frame, IMAGE_TYPE.GRAYSCALE))

    def test_image_not_al_match_false(self):
        frame = { IMAGE_TYPE_LABEL: IMAGE_TYPE.GRAYSCALE.value }
        self.assertFalse(util.frame_image_type_match(frame, IMAGE_TYPE.COLOR))


class Test_FrameContainsSegment(unittest.TestCase):
    def test_contains_test_case(self):
        frame = { FOCUS_HASH_LABEL: "anything" }
        self.assertTrue(util.frame_contains_segment(frame))

    def test_not_contains_test_case(self):
        frame = {}
        self.assertFalse(util.frame_contains_segment(frame))


class Test_PassValidSampleCriteria(unittest.TestCase):
    def test_all_criteria_pass_true(self):
        frame = {}
        frameImageTypeMatchMock = MagicMock(return_value=True)
        frameContainsSegmentMock = MagicMock(return_value=True)
        util.frame_image_type_match = frameImageTypeMatchMock
        util.frame_contains_segment = frameContainsSegmentMock
        self.assertTrue(util.frame_pass_valid_sample_criteria(frame, IMAGE_TYPE.ALL))

    def test_any_criteria_fail_false(self):        
        frame = {}
        frameImageTypeMatchMock = MagicMock(return_value=True)
        frameContainsSegmentMock = MagicMock(return_value=False)
        util.frame_image_type_match = frameImageTypeMatchMock
        util.frame_contains_segment = frameContainsSegmentMock
        self.assertFalse(util.frame_pass_valid_sample_criteria(frame, IMAGE_TYPE.ALL))

        