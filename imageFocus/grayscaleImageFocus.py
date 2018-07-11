import argparse, uuid
import cv2
import numpy as np


def get_grayscale_image_focus(path_to_image, path_to_output_directory, HSV_lower_bound, HSV_upper_bound):
    '''
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

    '''
    try:
        
        # Load the image and convert it to HSV from BGR
        # Then, threshold the HSV image to get only target border color
        bgr_image = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
        bgr_image = bgr_image[70:, 90:]
        mask = cv2.inRange(
            bgr_image, 
            HSV_lower_bound, 
            HSV_upper_bound)

        # Determine contours of the masked image

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

        if len(contours) == 0:
             raise Exception('Unable to find any matching contours')

        # Contour with maximum enclosed area corresponds to highlight rectangle
        
        max_contour = max(contours, key = cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # Crop the image to the bounding rectangle

        focus_image = bgr_image[y:y+h, x:x+w]

  
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
            raise Exception('Unable to find any matching contours')

        #find the biggest area
        max_contour = max(contours, key = cv2.contourArea)

        x, y, w, h = cv2.boundingRect(max_contour)

        # Crop the image to the bounding rectangle
        # As conservative measure crop inwards 3 pixels to guarantee no boundary

        cropped_image = focus_image[y+3:y+h-3, x+3:x+w-3]

        output_path = '{0}/{1}.png'.format(path_to_output_directory, uuid.uuid4())

        cv2.imwrite(output_path, cropped_image)

        return output_path

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

    get_grayscale_image_focus(
        args['image'],
        '.', 
        np.array(HSV_GRAYSCALE_THRESHOLD.LOWER.value, np.uint8), 
        HSV_upper_bound)