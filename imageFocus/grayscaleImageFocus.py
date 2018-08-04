import argparse, uuid
import cv2
import numpy as np
from matplotlib import pyplot as plt

from constants.ultrasoundConstants import (
    HSV_GRAYSCALE_THRESHOLD
)

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
        histogram_image = np.copy(bgr_image)
        bgr_image = bgr_image[70:, 90:]
        histogram_image[70:, 90:] = 0

        # cv2.imshow('histogram_image', histogram_image)
        # cv2.waitKey(0)
        
        hist, bins = np.histogram(histogram_image.ravel(), bins=256)
        
        artefact_focal_points = []
        # Always append the global maximum in the artefact list
        artefact_focal_points.append(100 + np.argmax(hist[100:]))

        # Optionally include another focal point if bright values show up as artefacts
        if hist[230 + np.argmax(hist[230:])] > 30:
            artefact_focal_points.append(230 + np.argmax(hist[230:]))

        print("Artefacts: {}".format(artefact_focal_points))

        # Two situations: The global max greater than 100 is the only color iff 
        # there is no value about 235 where the histgram count is above 30.

        # max_1_100 = 0 + np.argmax(hist[1:101])
        # max_101_200 = 101 + np.argmax(hist[101:201])
        # max_201_255 = 201 + np.argmax(hist[201:256]) 
        # print("global: {} | 200+: {}".format(global_max, max_200_plus))
        # print("Count: {} | Count: {}".format(hist[global_max], hist[max_200_plus]))

        # plt.plot(hist)
        # plt.show()

        # return

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

        mask_global = cv2.inRange(
            cropped_image,
            np.array([artefact_focal_points[0]-20]* 3, np.uint8),
            np.array([artefact_focal_points[0]+20]* 3, np.uint8)
        )

        if len(artefact_focal_points) > 1:

            mask_white = cv2.inRange(
                cropped_image,
                np.array([artefact_focal_points[1]-20] * 3, np.uint8),
                np.array([artefact_focal_points[1]+20] * 3, np.uint8)
            )

        # Mask to get rid of black is just adding noise. Consider removing and just going with high values 120, 240, etc
        
        if len(artefact_focal_points) == 1:
            composite_mask = mask_global
        else:
            composite_mask = cv2.bitwise_or(mask_global, mask_white)

        # Not terrible. Kernel size will need to be adjusted ad-hoc
        kernel = np.ones((2,2), np.uint8)
        dilation = cv2.morphologyEx(composite_mask, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(cropped_image,100,200, 3)

        plt.plot(hist)
        plt.show()

        cv2.imshow('composite', dilation)
        cv2.imshow('cropped', cropped_image)
        cv2.imshow('composite_mask', composite_mask)
        cv2.imshow('canny detection', edges)
        cv2.waitKey(0)
        


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
        
    args = ap.parse_args()

    get_grayscale_image_focus(
        args.image,
        '.', 
        np.array(HSV_GRAYSCALE_THRESHOLD.LOWER.value, np.uint8), 
        np.array(HSV_GRAYSCALE_THRESHOLD.UPPER.value, np.uint8))


    # get_region_of_interest(args.image)
