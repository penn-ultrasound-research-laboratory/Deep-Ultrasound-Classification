import argparse
import cv2

from src.constants.ultrasound import IMAGE_TYPE
from src.utilities.ocr.ocr import isolate_text

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()

parser.add_argument("-i",
                "--image",
                required=True,
                help="path to input image to be OCR'd")

parser.add_argument("-t",
                    "--image_type",
                    type=str,
                    default=IMAGE_TYPE.GRAYSCALE,
                    help="Image type (COLOR/GRAYSCALE)")

args = parser.parse_args()

# load the example image and convert it to grayscale
image = cv2.imread(args.image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

print(isolate_text(
    gray, 
    args.image_type))