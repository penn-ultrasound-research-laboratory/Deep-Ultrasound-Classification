# import the necessary packages
from constants.ultrasoundConstants import IMAGE_TYPE, READOUT_ABBREVS, WALL_FILTER_MODES
from PIL import Image
import os, re, cv2, argparse, pytesseract

RADIALITY = READOUT_ABBREVS.RADIALITY.value
COLOR_TYPE = READOUT_ABBREVS.COLOR_TYPE.value
COLOR_MODE = READOUT_ABBREVS.COLOR_MODE.value
ARAD = READOUT_ABBREVS.ARAD.value
RAD = READOUT_ABBREVS.RAD.value
CPA = READOUT_ABBREVS.CPA.value
COLOR_LEVEL = READOUT_ABBREVS.COLOR_LEVEL.value
WALL_FILTER = READOUT_ABBREVS.WALL_FILTER.value
PRF = READOUT_ABBREVS.PULSE_REPITITION_FREQUENCY.value

def isolate_text(grayscale_image, image_type):
	
	FOUND_TEXT = {}

	# Attempt to crop the top section 
	grayscale_image = grayscale_image[50:, :100]

	# write the grayscale image to disk as a temporary file so we can
	# apply OCR to it
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, grayscale_image)

	# load the image as a PIL/Pillow image, apply OCR, and then delete
	# the temporary file. Tesseract segmentation mode 11 is critical for this to work
	# None of the other automatic segmentation modes correctly read the text

	raw_text = pytesseract.image_to_string(Image.open(filename),
		lang='eng',
		boxes=False,  
		config='--psm 11') 

	os.remove(filename)

	text_segments = raw_text.splitlines()
	text_segments = [segment.upper().strip() for segment in text_segments if segment is not '']

	# Determine radiality

	FOUND_TEXT[RADIALITY] = ARAD if ARAD in text_segments else RAD

	if image_type is IMAGE_TYPE.GRAYSCALE:
		print('grayscale')

	else:

		FOUND_TEXT[COLOR_MODE] = IMAGE_TYPE.COLOR

		CPA_check = len([ segment for segment in text_segments if CPA in segment]) == 1
		COL_check = len([ segment for segment in text_segments if COLOR_LEVEL in segment]) == 1

		if not CPA_check ^ COL_check:
			raise IOError('Color/CPA text check did not pass for image')

		# Find the color mode

		found_mode = CPA if CPA_check else COLOR_LEVEL
		FOUND_TEXT[COLOR_MODE] = found_mode

		# Find the segment with CPA or COLOR
		
		mode_segment = [ segment for segment in text_segments if found_mode in segment][0]

		# Grab longest integer as color level percentage
		
		numerical_strings = [num_string for num_string in re.findall(r'[\d.]+', mode_segment)]
		numerical_strings = [ns.replace('.', '') for ns in numerical_strings]
		FOUND_TEXT[COLOR_LEVEL] = int(max(numerical_strings, key=len))

		# Find the segment with Wall Filter (WF)

		WF_check = len([ segment for segment in text_segments if WALL_FILTER in segment]) == 1

		if not WF_check:
			raise IOError('Wall filter check did not pass for image. Incorrect matching segments')

		wf_segment = [ segment for segment in text_segments if WALL_FILTER in segment][0]
		wf_segment = wf_segment.replace(WALL_FILTER, '').strip()

		if wf_segment not in WALL_FILTER_MODES:
			# Can always improve to levenshtein if poor matching
			raise IOError('Wall filter check did not pass for image. Incorrect mode: ${0}'.format(wf_segment))

		FOUND_TEXT[WALL_FILTER] = wf_segment

		# Find the segment with Pulse Repitition Frequency (PRF)

		PRF_check = len([ segment for segment in text_segments if PRF in segment]) == 1

		if not PRF_check:
			raise IOError('PRF check did not pass for image. Incorrect matching segments')

		prf_segment = [ segment for segment in text_segments if PRF in segment][0]
		
		numerical_strings = [num_string for num_string in re.findall(r'[\d.]+', prf_segment)]
		numerical_strings = [ns.replace('.', '') for ns in numerical_strings]
		FOUND_TEXT[PRF] = int(max(numerical_strings, key=len))

		print(FOUND_TEXT)

	print(text_segments)
	print(FOUND_TEXT)

	return {

	}


if __name__ == '__main__':


	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()

	ap.add_argument("-i", "--image", required=True,
		help="path to input image to be OCR'd")

	args = vars(ap.parse_args())

	# load the example image and convert it to grayscale
	image = cv2.imread(args["image"])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		
	isolate_text(gray, IMAGE_TYPE.GRAYSCALE)

