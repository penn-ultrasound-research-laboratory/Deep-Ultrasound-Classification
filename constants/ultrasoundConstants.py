import numpy as np
from enum import Enum

class HSV_COLOR_THRESHOLD(Enum):
    LOWER = [60, 50, 50]
    UPPER = [100, 255, 255]