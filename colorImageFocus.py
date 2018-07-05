import argparse
import cv2
import numpy as np
from constants.ultrasoundConstants import HSV_COLOR_THRESHOLD


def get_color_image_focus(path_to_image, path_to_output_directory, HSV_lower_bound, HSV_upper_bound):
    '''
    Determines the "focus" of an ultrasound frame in Color/CPA. 

    Ultrasound frames in Color/CPA mode highlight the tumor under examination to 
    focus the direction of the scan. This function extracts the highlighted region, which
    is surrounded by a bright rectangle and saves it to file. 

    Arguments:
        path_to_image:
        path_to_output_directory: 
        HSV_lower_bound: np.array([1, 3], uint8) lower HSV threshold to find highlight box
        HSV_upper_bound: np.array([1, 3], uint8) upper HSV threshold to find highlight box

    Returns:
        path_to_image_focus: path to saved image focus with has as filename

    Raises:
        IOError: in case of any errors with OpenCV or file operations 


    '''
    try:
        
        # load the example image and convert it to grayscale
        bgr_image = cv2.imread(path_to_image, cv2.IMREAD_COLOR)

        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, HSV_lower_bound, HSV_upper_bound)

        # Bitwise-AND mask and original image
        masked_output = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)

        ret, thresh = cv2.threshold(mask, 40, 255, 0)

        im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:

            # draw in blue the contours that were founded
            cv2.drawContours(masked_output, contours, -1, 255, 3)

            #find the biggest area
            c = max(contours, key = cv2.contourArea)

            x,y,w,h = cv2.boundingRect(c)

            # draw the book contour (in green)
            cv2.rectangle(masked_output,(x,y),(x+w,y+h),(0,255,0),2)

            # Crop the image to the bounding rectangle
            focus_bgr_image = bgr_image[y:y+h, x:x+w]

            # The bounding box includes the border. Remove the border by masking on the same 
            # thresholds as the initial mask, then flip the mask and draw a bounding box. 

            focus_hsv = cv2.cvtColor(focus_bgr_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(focus_hsv, HSV_lower_bound, HSV_upper_bound)
            mask = cv2.bitwise_not(mask)
        
            # Bitwise-AND mask and original image
            masked_output = cv2.bitwise_and(focus_bgr_image, focus_bgr_image, mask=mask)

            ret, thresh = cv2.threshold(mask, 40, 255, 0)


            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0:

                # draw in blue the contours that were founded
                cv2.drawContours(masked_output, contours, -1, 255, 3)

                #find the biggest area
                c = max(contours, key = cv2.contourArea)

                x,y,w,h = cv2.boundingRect(c)

                # draw the book contour (in green)
                cv2.rectangle(masked_output,(x,y),(x+w,y+h),(0,255,0), 2)

                # Crop the image to the bounding rectangle
                # As conservative measure crop inwards 3 pixels to guarantee no boundary

                cropped_image = focus_bgr_image[y+3:y+h-3, x+3:x+w-3]

                cv2.imshow('Original', bgr_image)
                cv2.imshow("Result", cropped_image)
                cv2.waitKey(0)

        else:
            raise Exception('Unable to find any matching contours')

    except Exception as exception:
        raise IOError('Error isolating and saving image focus')


if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--image", required=True,
        help="path to input image to be OCR'd")

    ap.add_argument("-p", "--preprocess", type=str, default="thresh",
        help="type of preprocessing to be done")
        
    args = vars(ap.parse_args())

    get_color_image_focus(
        args['image'],
        '.', 
        np.array(HSV_COLOR_THRESHOLD.LOWER.value, np.uint8), 
        np.array(HSV_COLOR_THRESHOLD.UPPER.value, np.uint8))