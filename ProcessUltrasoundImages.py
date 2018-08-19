from datetime import datetime
from utilities.imageUtilities import determine_image_type
from utilities.focus.colorImageFocus import get_color_image_focus
from utilities.focus.grayscaleImageFocus import get_grayscale_image_focus
from utilities.ocrUtilities import isolate_text

from constants.ultrasoundConstants import (
    FOCUS_HASH_LABEL,
    FRAME_LABEL,
    HSV_COLOR_THRESHOLD,
    HSV_GRAYSCALE_THRESHOLD,
    IMAGE_TYPE,
    IMAGE_TYPE_LABEL,
    TUMOR_BENIGN,
    TUMOR_MALIGNANT,
    TUMOR_TYPES,
    TUMOR_TYPE_LABEL,
    TUMOR_UNSPECIFIED
)

import numpy as np

import argparse
import cv2
import json
import logging
import os
import uuid

logger = logging.getLogger('processing')

def process_patient_ocr(
    absolute_path_to_patient_folder, 
    relative_path_to_frames_folder, 
    relative_path_to_focus_output_folder,
    manifest_file_pointer,
    timestamp,
    patient_type_label=None):
    """Process an individual patient

    Each patient"s ultrasound lives in a unique folder. Expectation is that a consistently named folder exists for each patient containing the individual frames of the patient"s ultrasound. For each patient, go frame by frame and process the frf = open("workfile", "w")ame for type (color/grayscale), information (WF, RAD/ARAD, etc.), and a frame focus. The frame focus is clearly delineated for color frames and must be inferred for grayscale frames. 

    Arguments:
        absolute_path_to_patient_folder: absolute path to patient folder
        relative_path_to_frames_folder: relative path from the patient folder to patient frames folder
        relative_path_to_focus_output_folder: relative path from the patient folder to frame focus output folder 
        manifest_file_pointer: file pointer to the manifest
        timestamp: timestamp to postfix to focus output directory

    Optional:
        patient_type_label: type of patient. Prefix in filename and present in all records

    Returns:
        A list of records for all cleanly processed patient frames
    """

    patient_label = os.path.basename(absolute_path_to_patient_folder)

    absolute_path_to_frame_directory = "{}/{}".format(
        absolute_path_to_patient_folder.rstrip("/"),
        relative_path_to_frames_folder)

    absolute_path_to_focus_output_directory = "{}/{}_{}".format(
        absolute_path_to_patient_folder.rstrip("/"),
        relative_path_to_focus_output_folder,
        timestamp)

    # Create the focus output directory if it does not exist

    if not os.path.isdir(absolute_path_to_focus_output_directory):
        os.mkdir(absolute_path_to_focus_output_directory)

    # Set up frame objects
    all_frames = [name for name in os.listdir(absolute_path_to_frame_directory)]
    ocr_clear_frames = []

    # Array storing all found & cleared text records
    found_text_records = []

    #
    # First pass over all data frames: 
    # OCR to get frame scale, RAD/ARAD, etc 
    #

    for frame in all_frames:
        logger.debug("Attempting text OCR for frame: {}".format(frame))
        try: 
            path_to_frame = "{}/{}".format(absolute_path_to_frame_directory, frame)
            bgr_image = cv2.imread(path_to_frame, cv2.IMREAD_COLOR)

            # Determine whether the frame is color or grayscale
            image_type = determine_image_type(bgr_image)

            # Always use the grayscale converted image for text isolation
            grayscale_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            grayscale_image = cv2.threshold(grayscale_image, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            found_text = isolate_text(grayscale_image, image_type)

            found_text[FRAME_LABEL] = os.path.basename(path_to_frame)
            found_text[TUMOR_TYPE_LABEL] = patient_type_label
            
            found_text[IMAGE_TYPE_LABEL] = IMAGE_TYPE.COLOR.value if image_type is IMAGE_TYPE.COLOR else IMAGE_TYPE.GRAYSCALE.value

            ocr_clear_frames.append(frame)

        except Exception as e:
            # Write the specific error in the failure log. Progress to the next frame
            logger.error("{}\n".format(e))
            continue

    if self.auto_resize_to_manifest_scale_max:
        try:
            frame_scale = self.patient_frames[self.frame_index][SCALE_LABEL]
            upscale_ratio = self.manifest_scale_max / frame_scale

            logging.debug("Upscale Ratio: {}".format(upscale_ratio))

            loaded_image = cv2.resize(
                loaded_image, 
                None, 
                fx=upscale_ratio, 
                fy=upscale_ratio, 
                interpolation=cv2.INTER_CUBIC)

        except:
            logger.error("Unable to auto-resize. Frame {} does not have scale label".format(
                self.patient_frames[self.frame_index][FOCUS_HASH_LABEL]
            ))

    #
    # Second pass over frames that cleared OCR: 
    # Segmentation to crop the frame's "focus"
    # Ideally, the focus is a close crop of the tumor
    #

    for frame in ocr_clear_frames:    
        try:
            if image_type is IMAGE_TYPE.COLOR:

                hash_path = get_color_image_focus(
                    path_to_frame, 
                    absolute_path_to_focus_output_directory, 
                    np.array(HSV_COLOR_THRESHOLD.LOWER.value, np.uint8), 
                    np.array(HSV_COLOR_THRESHOLD.UPPER.value, np.uint8))

            else: 
                hash_path = get_grayscale_image_focus(
                    path_to_frame, 
                    absolute_path_to_focus_output_directory, 
                    np.array(HSV_GRAYSCALE_THRESHOLD.LOWER.value, np.uint8), 
                    np.array(HSV_GRAYSCALE_THRESHOLD.UPPER.value, np.uint8))

            found_text[FOCUS_HASH_LABEL] = hash_path
            found_text_records.append(found_text)

        except Exception as e:
            # Write the specific error in the failure log. Progress to the next frame
            logger.error("[{}, {}, {}] | {}".format(patient_label, frame, image_type, e))
            continue

    return found_text_records
            


def process_patient_set(
    path_to_benign_top_level_directory,
    path_to_malignant_top_level_directory,
    relative_path_to_frames_folder,
    relative_path_to_focus_output_folder,
    path_to_manifest_output_directory, 
    timestamp=None,
    upscale_to_maximum=False):

    """Processes a set of patients from a top level directory.

    A set of patient folders lives in a top level directory (benign/malignant). Each patient is in
    its own folder. This script will generate two files: a manifest (manifest.json) containing 
    all of the parsed information from patient ultrasounds and a failure log (failure.txt) that 
    details any issues parsing. 

    Arguments:

        path_to_benign_top_level_directory: absolute path to top level directory 
            containing benign patient folders

        path_to_malignant_top_level_directory: absolute path to top level directory 
            containing malignant patient folders

        relative_path_to_frames_folder: relative path from the patient folder to patient frames folder
        relative_path_to_focus_output_folder: relative path from the patient folder to frame focus output folder 
        path_to_manifest_output_directory: absolute path to manifest output directory

    Optional:
        timestamp: String timestamp to use instead of generating using the current time")
        upscale_to_maximum: second path over all generated focuses to resize them to the maximum value in the manifest. E.g. say the largest scale on a frame is 4.8 cm. Say the scale on a new frame is 3.0 cm. We then upscale the focus by a factor of 4.8 / 3.0. If the scale has no frame, the scale factor will be 4.8 / mean(all_frames)


    Returns:
        A tuple containing: (Integer: #sucesses, Integer: #failures)
    """

    timestamp =  timestamp if timestamp is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    manifest_absolute_path = "{}/manifest_{}.json".format(
        path_to_manifest_output_directory.rstrip("/"), 
        timestamp)

    manifest_file = open(manifest_absolute_path, "a")

    # Process each individual patient

    all_patients = [
        (path_to_benign_top_level_directory, TUMOR_BENIGN),
        (path_to_malignant_top_level_directory, TUMOR_MALIGNANT)]

    patient_records = {}

    for path, patient_type_label in all_patients:

        patient_subdirectories = [name for name in os.listdir(path) 
            if os.path.isdir(os.path.join(path, name))]

        for patient in patient_subdirectories:

            logger.info("Processing patient: {}".format(patient))

            patient_directory_absolute_path = "{}/{}".format(
                path.rstrip("/"),
                patient)

            acquired_records = process_patient_ocr(
                patient_directory_absolute_path, 
                relative_path_to_frames_folder, 
                relative_path_to_focus_output_folder,
                manifest_file,
                timestamp,
                patient_type_label)

            patient_records[patient] = acquired_records

    patient_records["TIMESTAMP"] = timestamp

    # Dump the patient records to file
    json.dump(patient_records, manifest_file)

    # Cleanup

    manifest_file.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("path_to_benign_top_level_directory",
        help="absolute path to top level directory containing benign patient folders")

    parser.add_argument("path_to_malignant_top_level_directory",
        help="absolute path to top level directory containing malignant patient folders")
    
    parser.add_argument("-frames", "--relative_path_to_frames_directory",
                        type=str,
                        default="frames",
                        help="relative path from the patient folder to patient frames folder")

    parser.add_argument("-focus", "--relative_path_to_focus_output_folder",
                        type=str,
                        default="focus",
                        help="relative path from the patient folder to frame focus output folder ")

    parser.add_argument("-out", "--path_to_manifest_output_directory",
                        type=str,
                        default=".",
                        help="absolute path to manifest/failure log output directory")

    parser.add_argument("-time",
                        "--timestamp",
                        type=str,
                        default=None,
                        help="timestamp to use instead of generating one using the current time")

    parser.add_argument("-up",
                        "--upscale",
                        type=int,
                        default=0,
                        help="Boolean indicating whether to upscale frame focuses to the maximum value in the manifest")

    ## Missing functionality to wipe out old folders, manifests, error logs

    args = parser.parse_args()

    logging.basicConfig(level = logging.INFO, filename = "{}/failures_{}_{}.log".format(
        args.path_to_manifest_output_directory,
        args.timestamp,
        uuid.uuid4()
    ))

    process_patient_set(
        args.path_to_benign_top_level_directory,
        args.path_to_malignant_top_level_directory,
        args.relative_path_to_frames_directory,
        args.relative_path_to_focus_output_folder,
        args.path_to_manifest_output_directory, 
        timestamp=args.timestamp,
        upscale_to_maximum=bool(args.upscale))

