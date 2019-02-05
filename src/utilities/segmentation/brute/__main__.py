import argparse
import cv2
import os

from src.constants.ultrasound import (
    HSV_COLOR_THRESHOLD,
    FRAME_DEFAULT_ROW_CROP_FOR_SCAN_SELECTION,
    FRAME_DEFAULT_COL_CROP_FOR_SCAN_SELECTION)

from src.utilities.segmentation.brute.grayscale import select_scan_window_from_frame

# TODO: [#48] Add argument to support grayscale vs. color
# Only supports grayscale a.t.m

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image",
    help="path to input image to run through focus routine")

ap.add_argument("-f", "--folder",
    help="path to folder of input images to run through focus routine")
    
args = ap.parse_args()

if args.folder:
    for filename in os.listdir(args.folder):
        image = cv2.imread(args.folder + "/" + filename, cv2.IMREAD_GRAYSCALE)
        N, M = image.shape

        scan_window, scan_bounds = select_scan_window_from_frame(
            image, 
            5, 255, 
            select_bounds = (slice(70, N), slice(90, M)))

        x, y, w, h = scan_bounds

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            245,
            2)

        cv2.imshow("image", image)
        cv2.waitKey(0)

elif args.image:
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    N, M = image.shape

    scan_window, scan_bounds = select_scan_window_from_frame(
        image, 
        5, 255, 
        select_bounds = (
            slice(FRAME_DEFAULT_ROW_CROP_FOR_SCAN_SELECTION, N), 
            slice(FRAME_DEFAULT_COL_CROP_FOR_SCAN_SELECTION, M)))

    x, y, w, h = scan_bounds

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        245,
        2)

    cv2.imshow("image", image)
    cv2.waitKey(0)

else:
    print("Wrong input arguments")
