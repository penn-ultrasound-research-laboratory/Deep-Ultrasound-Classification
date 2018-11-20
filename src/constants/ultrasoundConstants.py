import numpy as np
from enum import Enum

FOCUS_HASH_LABEL = 'FOCUS'
FRAME_LABEL = 'FRAME'

FRAME_DEFAULT_ROW_CROP_FOR_SCAN_SELECTION = 70
FRAME_DEFAULT_COL_CROP_FOR_SCAN_SELECTION = 90

class HSV_COLOR_THRESHOLD(Enum):
    LOWER = [60, 50, 50]
    UPPER = [100, 255, 255]

class HSV_GRAYSCALE_THRESHOLD(Enum):
    LOWER = [1, 1, 1]
    UPPER = [255, 255, 255]

class IMAGE_TYPE(Enum):
    GRAYSCALE = 'GRAYSCALE'
    COLOR = 'COLOR'
    ALL = 'ALL'

IMAGE_TYPE_LABEL = 'IMAGE_TYPE'
INTERPOLATION_FACTOR_LABEL = 'INTERPOLATION_FACTOR'

NUMBER_CHANNELS_COLOR = 3
NUMBER_CHANNELS_GRAYSCALE = 1

# Inspiration: https://stackoverflow.com/questions/1363839/python-singleton-object-instantiation
class ReadoutAbbrevs(object):
    __instance = None

    def __new__(cls):
        if cls.__instance == None:
            cls.__instance = object.__new__(cls)
            cls.__instance.COLOR_MODE = 'COLOR_MODE'
            cls.__instance.COLOR_TYPE = 'COLOR_TYPE'
            cls.__instance.RADIALITY = 'RADIALITY'
            cls.__instance.RAD = 'RAD'
            cls.__instance.ARAD = 'ARAD'
            cls.__instance.COLOR_LEVEL = 'COL'
            cls.__instance.CPA = 'CPA'
            cls.__instance.WALL_FILTER = 'WF'
            cls.__instance.PULSE_REPITITION_FREQUENCY = 'PRF'
            cls.__instance.SCALE = "SCALE"
            cls.__instance.SIZE = 'SIZE'
        return cls.__instance

READOUT_ABBREVS = ReadoutAbbrevs()

SCALE_LABEL = "SCALE"

TIMESTAMP_LABEL = "TIMESTAMP"

TUMOR_UNSPECIFIED = 'UNSPEC'
TUMOR_BENIGN = 'BENIGN'
TUMOR_MALIGNANT = 'MALIGNANT'
TUMOR_TYPES = [TUMOR_BENIGN, TUMOR_MALIGNANT]

def tumor_integer_label(tumor_type):
    if tumor_type == TUMOR_BENIGN:
        return 0
    elif tumor_type == TUMOR_MALIGNANT:
        return 1
    else:
        return 2

TUMOR_TYPE_LABEL = 'TUMOR_TYPE'

WALL_FILTER_MODES = ['LOW', 'MED', 'HIGH']