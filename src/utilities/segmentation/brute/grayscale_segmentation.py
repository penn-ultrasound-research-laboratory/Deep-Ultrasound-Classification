import argparse
import uuid
import os
import cv2
import numpy as np

from src.constants.ultrasoundConstants import (
    HSV_COLOR_THRESHOLD,
    FRAME_DEFAULT_ROW_CROP_FOR_SCAN_SELECTION,
    FRAME_DEFAULT_COL_CROP_FOR_SCAN_SELECTION)

def select_out_curvature_line(
    scan_window,
    target_slice_percentage=0.025,
    target_slice_erosion_kernel_size=(6,6),
    remain_slice_dilation_kernel_size=(8,8)):
    """
    Selects out the curvature line that sometimes overlaps with the scan window in an ultrasound frame.
    
    The curvature line is consistently thin (1-2px). A parallel set of morphological operations is used to "erase"
    the curvature line. We erode the region known to contain the curvature and dilate the remaining area to 
    all but guarantee that the largest contour covers the entire horizontal space.
 
    Arguments:
        scan_window                         scan window
        
    Optional:
        target_slice_percentage             Percentage taken from the righthand side of the image to erase a curvature                                      line. Default 0.025 or 2.5%

        target_slice_erosion_kernel_size    Size of the erosion kernel used to "erase" the curvature line

        remain_slice_dilation_kernel_size   Size of the dilation kernel used to fill-in the remainder of the image not                                      included in the target slice
    Returns:
        scan_bounds                         The rectangular bounds of largest contour in the scan window after 
                                                morphological operations
    """
    # Apply Otsu thresholding to the scan window
    mask = cv2.threshold(scan_window, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    r_s = int(scan_window.shape[1] // (1 / target_slice_percentage))
    
    # The target (right) slice may contain a curvature line
    target_slice = mask[:, mask.shape[1] - r_s:]

    # The remainder (left) slice is the remainder of the image outside the target slice
    remainder_slice = mask[:, :mask.shape[1] - r_s]

    # Erode the target slice
    target_slice[:] = cv2.erode(
        target_slice, 
        np.ones(target_slice_erosion_kernel_size, np.uint8))

    # Dilate the remainder slice
    remainder_slice[:] = cv2.dilate(
        remainder_slice, 
        np.ones(remain_slice_dilation_kernel_size, np.uint8))

    # Determine mask contours
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    if len(contours) == 0:
        raise Exception("Unable to find any matching contours")

    scan_contour = max(contours, key = cv2.contourArea)

    return cv2.boundingRect(scan_contour)


def select_scan_window_from_frame(
    image, 
    mask_lower_bound,
    mask_upper_bound,
    select_bounds=None,
    center_slice_percentage=0.966667,
    center_slice_dilation_kernel_size=(8,8)):
    """
    Selects the scan window of a raw ultrasound frame 

    Ultrasound frames contains a significant amount of diagnostic information about the patient and 
    ongoing scan. The frame boundary regions of the frame will list scan strength, frame scale, etc.
    This function selects the region of the frame that contains the scan image.

    Arguments:
        image                               raw ultrasound frame (GRAYSCALE)
        mask_lower_bound                    lower bound for mask
        mask_upper_bound                    upper bound for mask
    
    Optional:
        select_bounds                       Slice of the raw frame searched for scan window. Default to full-frame
                                                passed-in as (row_slice, column_slice)

        center_slice_percentage             Percentage of the image taken from the center that will be dilated.                                             Dilating the center portion of the image supports better contour search

        center_slice_dilation_kernel_size   Size of the dilation kernel used to fill-in the center region of the image

    Returns:
        scan_window                         Slice of the raw frame containing the scan window
        scan_bounds                         The rectangular bounds of the scan window (x, y, w, h) w.r.t to the                                             original frame. Not in the coordinate system of slice.
    """
    # Optionally slice the input frame
    if select_bounds is not None:
        row_slice, column_slice = select_bounds
        image = image[row_slice, column_slice]

    N, M = image.shape

    # Otsu thresholding on the image to remove background
    mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Run morphological closing on the target center percentage of the mask
    y_s = int(N // (1 / center_slice_percentage))
    x_s = int(M // (1 / center_slice_percentage))

    center_region = mask[slice(y_s, N - y_s), slice(x_s, M - x_s)] 
    
    # Dilate the center slice of the mask to make contour search more effective
    center_region[:] = cv2.dilate(
        center_region, 
        np.ones(center_slice_dilation_kernel_size, np.uint8))

    # Determine mask contours
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    if len(contours) == 0:
        raise Exception("Unable to find any matching contours")

    # Contour with maximum enclosed area corresponds to scan window
    scan_contour = max(contours, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(scan_contour)

    scan_window = image[y: y + h, x: x + w]

    # Remove the curvature line from the scan window
    x_s, y_s, w_s, _ = select_out_curvature_line(scan_window)

    # Only keep the horizontal information of the found contour. Leverage
    # the linearity of the tissue w.r.t to ultrasound scan (tissue stacks like layers).
    # We found a contour that has the correct width when righthand curvature line removed, no
    # guarantee on height so forget information
    scan_window_removed_line = scan_window[:, :x_s + w_s]

    # Return the scan window slice of the image
    if select_bounds is None:
        return (scan_window_removed_line, scan_contour)
    else:
        # Contour is with respect to the original image, 
        scan_contour = (x + column_slice.start, y + row_slice.start, w_s, h)
        return (scan_window_removed_line, scan_contour)


def load_select_scan_window(path_to_image, cm=5):
    """
    Load and Select the scan window of an ultrasound frame 

    Arguments:
        path_to_image                       path to input image file

    Optional: 
        cm                                  Global inwards crop to create a margin ("crop margin"). Default 5px

    Returns:
        scan_window                         The found scan window of the frame

    Raises:
        IOError: in case of any errors with OpenCV or file operations 
    """
    try:

        image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)

        scan_window, scan_bounds = select_scan_window_from_frame(
            image, 
            5, 255, 
            select_bounds = (
                slice(FRAME_DEFAULT_ROW_CROP_FOR_SCAN_SELECTION, N), 
                slice(FRAME_DEFAULT_COL_CROP_FOR_SCAN_SELECTION, M)))

        # Crop the image to the bounding rectangle
        # As conservative measure crop inwards 3 pixels to guarantee no boundary
        scan_window = scan_window[y + cm: y + h - cm,   x + cm: x + w - cm] 

        return scan_window

    except Exception as exception:
        raise IOError("Error isolating and saving image focus")


def load_select_save_scan_window(
    path_to_image, 
    path_to_output_directory, 
    cm=0):
    """
    Load, Select, and Save the scan window of a grayscale ultrasound frame

    Arguments:
        path_to_image                       Path to input image file
        path_to_output_directory            Path to output directory 

    Optional: 
        cm                                  Global inwards crop to create a margin ("crop margin"). Default 0px

    Returns:
        output_path                         Path to saved scan window

    Raises:
        IOError: in case of any errors with OpenCV or file operations 
    """
    try:
        scan_window = load_select_scan_window(path_to_image, cm)

        output_path = "{0}/{1}.png".format(path_to_output_directory, uuid.uuid4())

        cv2.imwrite(output_path, scan_window)

        return output_path

    except Exception as exception:
        raise IOError("Error isolating and saving image focus")
