# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
 
def isolate_text(grayscale_image):
	

	# Attempt to crop the top section 
	grayscale_image = grayscale_image[250:265, :100]

	# write the grayscale image to disk as a temporary file so we can
	# apply OCR to it
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, grayscale_image)

	# load the image as a PIL/Pillow image, apply OCR, and then delete
	# the temporary file
	text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)

	## Assume the image is grayscale
	print(text)

	# show the output images
	cv2.imshow("Output", grayscale_image)
	cv2.waitKey(0)

if __name__ == '__main__':


	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()

	ap.add_argument("-i", "--image", required=True,
		help="path to input image to be OCR'd")

	args = vars(ap.parse_args())

	# load the example image and convert it to grayscale
	image = cv2.imread(args["image"])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	isolate_text(gray)

