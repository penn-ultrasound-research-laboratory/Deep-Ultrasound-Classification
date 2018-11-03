import argparse, uuid
import cv2
import numpy as np
from constants.ultrasoundConstants import HSV_COLOR_THRESHOLD

def select_focus_from_scan_window():
    """
    Selects the focus from a scan window

    The scan window of an ultrasound scan contains the tumor and surrounding tissue. While a radiologist can
    ignore the tissue on the periphery of a tumor scan, models are sensitive to noise. This function is a simple
    (naive) method to return the region containing just the tumor (tumor ROI) in the scan window. 

    Arguments:
        image                               scan window
    
    Optional:
        select_bounds                       Slice of the scan window searched for tumor ROI. Default to full-frame
                                                passed-in as (row_indices, col_indices)

    Returns:
        scan_window                         Slice of the scan window containing the tumor ROI
        scan_bounds                         The rectangular bounds of the tumor ROI (x, y, w, h)
    """
    pass


def select_scan_window_from_frame(
    image, 
    mask_lower_bound,
    mask_upper_bound,
    select_bounds=None):
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

    Returns:
        scan_window                         Slice of the raw frame containing the scan window
        scan_bounds                         The rectangular bounds of the scan window (x, y, w, h) w.r.t to the                                             original frame. Not in the coordinate system of slice.
    """

    if select_bounds is not None:
        row_slice, column_slice = select_bounds
        image = image[row_slice, column_slice]

    mask = cv2.inRange(
        image, 
        mask_lower_bound, 
        mask_upper_bound)

    # Determine contours of the masked image
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    if len(contours) == 0:
        raise Exception("Unable to find any matching contours")

    # Contour with maximum enclosed area corresponds to highlight rectangle
    
    max_contour = max(contours, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    # Crop the image to the bounding rectangle
    focus_image = image[y:y+h, x:x+w]

    if select_bounds is None:
        return (focus_image, max_contour)
    else:
        ret_contour = (x + column_slice.stop, y + row_slice.stop, w, h)
        return (focus_image, ret_contour)


def get_grayscale_image_focus(
    path_to_image, 
    path_to_output_directory, 
    HSV_lower_bound, 
    HSV_upper_bound,
    interpolation_factor=None,
    interpolation_method=cv2.INTER_CUBIC):
    """
    Determines the "focus" of an ultrasound frame in Color/CPA. 

    Ultrasound frames in Color/CPA mode highlight the tumor under examination to 
    focus the direction of the scan. This function extracts the highlighted region, which
    is surrounded by a bright rectangle and saves it to file. 

    Arguments:
        path_to_image: path to input image file
        path_to_output_directory: path to output directory 
        HSV_lower_bound: np.array([1, 3], uint8) lower HSV threshold to find highlight box
        HSV_upper_bound: np.array([1, 3], uint8) upper HSV threshold to find highlight box

    Returns:
        path_to_image_focus: path to saved image focus with has as filename

    Raises:
        IOError: in case of any errors with OpenCV or file operations 

    """
    try:
        

  
        # The bounding box includes the border. Remove the border by masking on the same 
        # thresholds as the initial mask, then flip the mask and draw a bounding box. 

        focus_hsv = cv2.cvtColor(focus_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            focus_hsv, 
            HSV_lower_bound, 
            HSV_upper_bound)
            
        mask = cv2.bitwise_not(mask)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

        if len(contours) == 0:
            raise Exception("Unable to find any matching contours")

        #find the biggest area
        max_contour = max(contours, key = cv2.contourArea)

        x, y, w, h = cv2.boundingRect(max_contour)

        # Crop the image to the bounding rectangle
        # As conservative measure crop inwards 3 pixels to guarantee no boundary

        cropped_image = focus_image[y+3:y+h-3, x+3:x+w-3]

        # Interpolate (upscale/downscale) the found segment if an interpolation factor is passed
        if interpolation_factor is not None:
            cropped_image = cv2.resize(
                cropped_image, 
                None, 
                fx=interpolation_factor, 
                fy=interpolation_factor, 
                interpolation=interpolation_method)

        output_path = "{0}/{1}.png".format(path_to_output_directory, uuid.uuid4())

        cv2.imwrite(output_path, cropped_image)

        return output_path

    except Exception as exception:
        raise IOError("Error isolating and saving image focus")


if __name__ == "__main__":

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--image", required=True,
        help="path to input image target of OCR subroutine")

    ap.add_argument("-p", "--preprocess", type=str, default="thresh",
        help="type of preprocessing to be done")
        
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
    N, M = image.shape

    scan_window, scan_bounds = select_scan_window_from_frame(
        image, 
        1, 255, 
        select_bounds = (slice(70, N), slice(90, M)))

    cv2.imshow("window", image)

    cv2.draw
    cv2.waitKey(0)