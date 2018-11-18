import argparse
import cv2
import uuid
import numpy as np

from constants.ultrasoundConstants import HSV_COLOR_THRESHOLD
from utilities.imageUtilities import center_crop_to_target_pixel_boundary


def get_color_image_focus(
    image,
    HSV_lower_bound, 
    HSV_upper_bound,
    crop_inside_boundary_radius=3):
    """
    Determines the "focus" of an ultrasound frame in Color/CPA. 

    Ultrasound frames in Color/CPA mode highlight the tumor under examination to 
    focus the direction of the scan. This function extracts the highlighted region, which
    is surrounded by a bright rectangle and saves it to file. 

    Arguments:
        image                               The input image in BGR format    
        HSV_lower_bound                     np.array([1, 3], uint8) lower HSV threshold to find highlight box
        HSV_upper_bound                     np.array([1, 3], uint8) upper HSV threshold to find highlight box

    Optional:
        crop_inside_boundary_radius         Crop center of found image focus creating boundary of radius pixels.
                                                Default is 2px boundary radius.
    Returns:
        image_focus                         The found color image focus
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, HSV_lower_bound, HSV_upper_bound)

    # Determine contours of the masked image

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    if len(contours) == 0:
            raise Exception("Unable to find any matching contours.")

    # Contour with maximum enclosed area corresponds to highlight rectangle

    max_contour = max(contours, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    # Crop the image to the bounding rectangle

    focus_image = image[y:y+h, x:x+w]

    # The bounding box includes the border. Remove the border by masking on the same 
    # thresholds as the initial mask, then flip the mask and draw a bounding box. 

    focus_hsv = cv2.cvtColor(focus_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(focus_hsv, HSV_lower_bound, HSV_upper_bound)
    mask = cv2.bitwise_not(mask)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    if len(contours) == 0:
        raise Exception("Unable to find any matching contours.")

    #find the biggest area
    max_contour = max(contours, key = cv2.contourArea)

    x, y, w, h = cv2.boundingRect(max_contour)

    # Crop the image to the bounding rectangle

    image_focus = focus_image[y: y + h, x: x + w]

    # As conservative measure, crop inwards by small radius to guarantee no boundary
    image_focus, _ = center_crop_to_target_pixel_boundary(
        image_focus,
        crop_inside_boundary_radius,
        crop_inside_boundary_radius
    )

    return image_focus


def load_select_color_image_focus(
    path_to_image, 
    HSV_lower_bound, 
    HSV_upper_bound,
    crop_inside_boundary_radius=3):
    """
    Load and Select color image focus of an input frame

    Arguments:
        path_to_image                       Path to input image file
        HSV_lower_bound                     np.array([1, 3], uint8) lower HSV threshold to find highlight box
        HSV_upper_bound                     np.array([1, 3], uint8) upper HSV threshold to find highlight box

    Optional:
        crop_inside_boundary_radius         Crop center of found image focus creating boundary of radius pixels.
                                                Default is 2px boundary radius.
    Returns:
        image_focus                         The found color image focus

    Raises:
        IOError: in case of any errors with OpenCV or file operations 
    """
    try:
        bgr_image = cv2.imread(path_to_image, cv2.IMREAD_COLOR)

        image_focus = get_color_image_focus(
            bgr_image,
            HSV_lower_bound,
            HSV_upper_bound,
            crop_inside_boundary_radius
        )

        return image_focus

    except Exception as e:
        raise IOError("Error isolating and saving image focus. " + str(e))


def load_select_save_color_image_focus(
    path_to_image, 
    path_to_output_directory, 
    HSV_lower_bound, 
    HSV_upper_bound,
    crop_inside_boundary_radius=3):
    """
    Load, Select, and Save color image focus of an input frame

    Arguments:
        path_to_image                       Path to input image file
        path_to_output_directory            Path to output directory 
        HSV_lower_bound                     np.array([1, 3], uint8) lower HSV threshold to find highlight box
        HSV_upper_bound                     np.array([1, 3], uint8) upper HSV threshold to find highlight box

    Optional:
        crop_inside_boundary_radius         Crop center of found image focus creating boundary of radius pixels.
                                                Default is 2px boundary radius.
    Returns:
        path_to_image_focus                 Path to saved image focus with has as filename

    Raises:
        IOError: in case of any errors with OpenCV or file operations 
    """
    try:
        image_focus = load_select_color_image_focus(
            path_to_image,
            HSV_lower_bound,
            HSV_upper_bound,
            crop_inside_boundary_radius
        )

        output_path = "{0}/{1}.png".format(path_to_output_directory, uuid.uuid4())

        cv2.imwrite(output_path, image_focus)

        return output_path

    except Exception as e:
        raise IOError("Error isolating and saving image focus. " + str(e))


if __name__ == "__main__":

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--image", required=True,
        help="path to input image target of OCR subroutine")

    ap.add_argument("-p", "--preprocess", type=str, default="thresh",
        help="type of preprocessing to be done")
        
    args = vars(ap.parse_args())

    image_focus = load_select_color_image_focus(
        args["image"],
        np.array(HSV_COLOR_THRESHOLD.LOWER.value, np.uint8), 
        np.array(HSV_COLOR_THRESHOLD.UPPER.value, np.uint8))

    cv2.imshow("focus", image_focus)
    cv2.waitKey(0)