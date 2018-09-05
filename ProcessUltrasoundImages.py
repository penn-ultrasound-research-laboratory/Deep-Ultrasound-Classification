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
    INTERPOLATION_FACTOR_LABEL,
    READOUT_ABBREVS as RA,
    SCALE_LABEL,
    TIMESTAMP_LABEL,
    TUMOR_BENIGN,
    TUMOR_MALIGNANT,
    TUMOR_TYPES,
    TUMOR_TYPE_LABEL,
    TUMOR_UNSPECIFIED
)

from tqdm import tqdm

import numpy as np

import argparse
import cv2
import json
import logging
import os
import uuid
import sys

logger = logging.getLogger('processing')

def frame_segmentation(
    path_to_frame,
    absolute_path_to_focus_output_directory,
    image_type,
    interpolation_factor=None):
    '''
            upscale_to_maximum: (float, float) containing the maximum scale to be used and the average in case of missing scale for a frame
    '''
    if image_type is IMAGE_TYPE.COLOR:
        hash_path = get_color_image_focus(
            path_to_frame, 
            absolute_path_to_focus_output_directory, 
            np.array(HSV_COLOR_THRESHOLD.LOWER.value, np.uint8), 
            np.array(HSV_COLOR_THRESHOLD.UPPER.value, np.uint8),
            interpolation_factor=interpolation_factor)
    else: 
        hash_path = get_grayscale_image_focus(
            path_to_frame, 
            absolute_path_to_focus_output_directory, 
            np.array(HSV_GRAYSCALE_THRESHOLD.LOWER.value, np.uint8), 
            np.array(HSV_GRAYSCALE_THRESHOLD.UPPER.value, np.uint8),
            interpolation_factor=interpolation_factor)

    return hash_path


def frame_ocr(color_frame, image_type):
    # Always use the grayscale converted image for text isolation
    grayscale_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    
    # Some weird thresholding code. Not sure where it came from
    grayscale_frame = cv2.threshold(grayscale_frame, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return isolate_text(grayscale_frame, image_type)


            
def patient_ocr(
    absolute_path_to_patient_folder, 
    patient,
    relative_path_to_frames_folder, 
    relative_path_to_focus_output_folder,
    timestamp,
    composite_records=None,
    patient_type_label=None):
    """Run OCR subroutine for an individual patient

    Each patient"s ultrasound lives in a unique folder. Expectation is that a consistently named folder exists for each patient 
    containing the individual frames of the patient"s ultrasound. For this patient, go frame by frame and process use OCR to 
    extract textual information (WF, RAD/ARAD, etc.), and a the tumor segmentation ("frame focus"). The frame focus is clearly
    delineated for color frames and must be inferred for grayscale frames. 

    Arguments:
        absolute_path_to_patient_folder: absolute path to patient folder
        patient: the basename of the patient folder. e.g. (00BER90238)
        relative_path_to_frames_folder: relative path from the patient folder to patient frames folder
        relative_path_to_focus_output_folder: relative path from the patient folder to frame focus output folder 
        timestamp: timestamp to postfix to focus output directory

    Optional:
        composite_records: reference to manifest with keys patient ids (e.g. 00BER90238) 
            and values list of all frame records for the patient
        patient_type_label: type of patient. Prefix in filename and present in all records

    Returns:
       If composite_records is passed in, returns reference to composite_records. Else returns an array of patient records
    """

    BUILD_NEW_RECORDS = composite_records is None

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

    individual_patient_frames = [name for name in os.listdir(absolute_path_to_frame_directory)]
    
    # Create an array to store all found & cleared text patient records if building the records from scratch
    if BUILD_NEW_RECORDS:
        compiled_patient_records = []

    ############################################################
    # OCR to get frame scale, RAD/ARAD, etc 
    ############################################################

    for frame_label in individual_patient_frames:

        path_to_frame = "{}/{}".format(absolute_path_to_frame_directory, frame_label)
        color_frame = cv2.imread(path_to_frame, cv2.IMREAD_COLOR)

        # Run the OCR subroutine
        try:
            # Determine whether the frame is color or grayscale
            image_type = determine_image_type(color_frame)

            logger.info("Attempting text OCR for frame: {}".format(frame_label))
            found_text = frame_ocr(color_frame, image_type)
        except Exception as e:
            logger.error("Failed text OCR for frame: {}. {}".format(frame_label, str(e)))
            continue
        
        if BUILD_NEW_RECORDS:
            # Use the found text as a basis for a new frame record
            found_text[FRAME_LABEL] = frame_label
            found_text[TUMOR_TYPE_LABEL] = patient_type_label
            found_text[IMAGE_TYPE_LABEL] = image_type.value

            compiled_patient_records.append(found_text)
            
        else:
            # Get the reference to the current frame record
            frame_record = [rec for rec in composite_records[patient] if rec[FRAME_LABEL] == frame_label]
            
            if len(frame_record) == 0:
                logger.error("Frame record not in composite records: {}".format(frame_label))
                continue

            frame_record = frame_record[0]

            # Augment the existing record with new frame information    
            for key, value in found_text.items():
                frame_record[key] = value        

    # Either return the new records or reference to the composite records
    return compiled_patient_records if BUILD_NEW_RECORDS else composite_records


def patient_segmentation(
    absolute_path_to_patient_folder, 
    patient,
    relative_path_to_frames_folder, 
    relative_path_to_focus_output_folder,
    timestamp,
    composite_records=None,
    patient_type_label=None,
    interpolation_context=None):

    '''
        ############################################################
        # Segmentation to crop the frame's "focus"
        # Ideally, the focus is a close crop of the tumor
        ############################################################

            interpolation_context: (float, float) containing the maximum scale to be used and the average in case of missing scale for a frame
    '''

    BUILD_NEW_RECORDS = composite_records is None

    # patient_label = os.path.basename(absolute_path_to_patient_folder)
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
    individual_patient_frames = [name for name in os.listdir(absolute_path_to_frame_directory)]
    
    # Create an array to store all found & cleared text patient records if building the records from scratch
    if BUILD_NEW_RECORDS:
        compiled_patient_records = []

    for frame_label in individual_patient_frames:

        path_to_frame = "{}/{}".format(absolute_path_to_frame_directory, frame_label)
        color_frame = cv2.imread(path_to_frame, cv2.IMREAD_COLOR)

        try:
            # Determine whether the frame is color or grayscale
            image_type = determine_image_type(color_frame)
            
            logger.info("Attempting tumor segmentation for frame: {}".format(frame_label))

            if interpolation_context is not None and not BUILD_NEW_RECORDS:
                logger.info("Segmentation | Upscaling frame: {}".format(frame_label))
                # Get the reference to the current frame record
                frame_record = [rec for rec in composite_records[patient] if rec[FRAME_LABEL] == frame_label]
                
                if len(frame_record) == 0:
                    logger.error("Segmentation | Frame record not in composite records: {}".format(frame_label))
                    continue

                frame_record = frame_record[0]

                # If the scale is undefined, use the global average frame scale
                found_scale = frame_record.get(RA.SCALE, interpolation_context[1])
                found_scale = found_scale if found_scale is not None else interpolation_context[1]

                interpolation_factor = found_scale / interpolation_context[0]

                # Sanity check
                if interpolation_factor > 5:
                    logger.warning("Segmentation | Factor: {} exceeds limit for frame: {}".format(interpolation_factor, frame_label))

                logger.info("Segmentation | Interpolation factor: {} | frame: {}".format(interpolation_factor, frame_label))

                # Get the tumor segmentation from the patient frame            
                hash_path = frame_segmentation(
                    path_to_frame,
                    absolute_path_to_focus_output_directory,
                    image_type,
                    interpolation_factor=interpolation_factor)
            else:
                # Get the tumor segmentation from the patient frame            
                hash_path = frame_segmentation(
                    path_to_frame,
                    absolute_path_to_focus_output_directory,
                    image_type)

        except Exception as e:
            logger.error("Failed tumor segmentation for frame: {}. {}".format(frame_label, str(e)))
            continue

        if BUILD_NEW_RECORDS:

            new_record = {}

            new_record[FRAME_LABEL] = frame_label
            new_record[TUMOR_TYPE_LABEL] = patient_type_label
            new_record[IMAGE_TYPE_LABEL] = image_type.value

            # Use the found text as a basis for a new frame record
            new_record[FOCUS_HASH_LABEL] = hash_path

            if interpolation_context is not None:
                new_record[INTERPOLATION_FACTOR_LABEL] = interpolation_factor
            
            compiled_patient_records.append(new_record)
            
        else:
            # Get the reference to the current frame record
            frame_record = [rec for rec in composite_records[patient] if rec[FRAME_LABEL] == frame_label]
            
            if len(frame_record) == 0:
                logger.error("Frame record not in composite records: {}".format(frame_label))
                continue

            frame_record = frame_record[0]

            if interpolation_context is not None:
                frame_record[INTERPOLATION_FACTOR_LABEL] = interpolation_factor

            # Augment the existing record with new frame information            
            frame_record[FOCUS_HASH_LABEL] = hash_path

    # Either return the new records or reference to the composite records
    return compiled_patient_records if BUILD_NEW_RECORDS else composite_records



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
        upscale_to_maximum: second path over all generated focuses to resize them to the maximum value in the manifest. 
            E.g. say the largest scale on a frame is 4.8 cm. Say the scale on a new frame is 3.0 cm. We then upscale 
            the focus by a factor of 4.8 / 3.0. If the scale has no frame, the scale factor will be 4.8 / avg(all_frames)
    """

    '''
                                    ██████╗  ██████╗██████╗ 
                                    ██╔═══██╗██╔════╝██╔══██╗
                                    ██║   ██║██║     ██████╔╝
                                    ██║   ██║██║     ██╔══██╗
                                    ╚██████╔╝╚██████╗██║  ██║
                                    ╚═════╝  ╚═════╝╚═╝  ╚═╝
    '''

    ALL_PATIENTS = [
        (path_to_benign_top_level_directory, TUMOR_BENIGN),
        (path_to_malignant_top_level_directory, TUMOR_MALIGNANT)]

    patient_records = {}

    # Process each individual patient
    for path, patient_type_label in tqdm(ALL_PATIENTS, desc="OCR"):

        patient_subdirectories = [name for name in os.listdir(path) 
            if os.path.isdir(os.path.join(path, name))]

        for patient_label in tqdm(patient_subdirectories, desc=patient_type_label, leave=False):

            logger.info("OCR | Processing patient: {}".format(patient_label))

            patient_directory_absolute_path = "{}/{}".format(
                path.rstrip("/"),
                patient_label)

            acquired_records = patient_ocr(
                patient_directory_absolute_path, 
                patient_label,
                relative_path_to_frames_folder, 
                relative_path_to_focus_output_folder,
                timestamp,
                composite_records=None,
                patient_type_label=patient_type_label)

            patient_records[patient_label] = acquired_records

    '''
    █████╗ ██╗   ██╗████████╗ ██████╗       ███████╗ ██████╗ █████╗ ██╗     ██╗███╗   ██╗ ██████╗ 
    ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗      ██╔════╝██╔════╝██╔══██╗██║     ██║████╗  ██║██╔════╝ 
    ███████║██║   ██║   ██║   ██║   ██║█████╗███████╗██║     ███████║██║     ██║██╔██╗ ██║██║  ███╗
    ██╔══██║██║   ██║   ██║   ██║   ██║╚════╝╚════██║██║     ██╔══██║██║     ██║██║╚██╗██║██║   ██║
    ██║  ██║╚██████╔╝   ██║   ╚██████╔╝      ███████║╚██████╗██║  ██║███████╗██║██║ ╚████║╚██████╔╝
    ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝       ╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝                                                                                          
    '''
    # Handle upscale to the maximum scale found in the corpus
    if upscale_to_maximum:

        logger.info("Scale to maximum flag TRUE. Attempting to get scale information.")
        scale_minimum = 1000
        scale_total = 0 
        scale_count = 0
        scale_average = 0

        # Process each individual patient
        for path, patient_type_label in tqdm(ALL_PATIENTS, desc="Auto-scale"):

            patient_subdirectories = [name for name in os.listdir(path) 
                if os.path.isdir(os.path.join(path, name))]

            for patient_label in tqdm(patient_subdirectories, desc=patient_type_label, leave=False):

                frames_with_none = []
                scale_histogram = {}

                for frame in patient_records[patient_label]:

                    found_scale = frame.get(RA.SCALE, None)

                    if found_scale is not None:
                        scale_minimum = min(found_scale, scale_minimum) 
                        scale_total += found_scale
                        scale_count += 1
                        
                        if found_scale in scale_histogram:
                            scale_histogram[found_scale] += 1
                        else:
                            scale_histogram[found_scale] = 1
                    else:
                        # Frames with no specified scale are not included in the average
                        frames_with_none.append(frame)

                most_common_scale = max(scale_histogram, key=scale_histogram.get)

                for frame_with_no_scale in frames_with_none:
                    logger.info("Frame: {} missing scale. Updating with scale: {}".format(frame_with_no_scale[FRAME_LABEL], most_common_scale))

                    frame_with_no_scale[SCALE_LABEL] = most_common_scale
              
        scale_average = scale_total / scale_count
        
        print("Scale to minimum. minimum: {}. Average: {}".format(scale_minimum, scale_average))
        logger.info("Scale to minimum. minimum: {}. Average: {}".format(scale_minimum, scale_average))

    '''
        ███████╗███████╗ ██████╗ ███╗   ███╗███████╗███╗   ██╗████████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
        ██╔════╝██╔════╝██╔════╝ ████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
        ███████╗█████╗  ██║  ███╗██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ███████║   ██║   ██║██║   ██║██╔██╗ ██║
        ╚════██║██╔══╝  ██║   ██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
        ███████║███████╗╚██████╔╝██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
        ╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
                                                                                                            
    '''

    for path, patient_type_label in tqdm(ALL_PATIENTS, desc="Segmentation"):

        patient_subdirectories = [name for name in os.listdir(path) 
            if os.path.isdir(os.path.join(path, name))]

        for patient_label in tqdm(patient_subdirectories, desc=patient_type_label, leave=False):

            logger.info("SEGMENTATION | Processing patient: {}".format(patient_label))

            patient_directory_absolute_path = "{}/{}".format(
                path.rstrip("/"),
                patient_label)

            interpolation_context = (scale_minimum, scale_average) if upscale_to_maximum else None

            # Pass in complete records to update records in-place
            patient_segmentation(
                patient_directory_absolute_path, 
                patient_label,
                relative_path_to_frames_folder, 
                relative_path_to_focus_output_folder,
                timestamp,
                composite_records=patient_records,
                patient_type_label=patient_type_label,
                interpolation_context=interpolation_context)

    # Dump the patient records to file
    manifest_absolute_path = "{}/manifest_{}.json".format(
        path_to_manifest_output_directory.rstrip("/"), 
        timestamp)

    manifest_file = open(manifest_absolute_path, "a")

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

    timestamp =  args.timestamp if args.timestamp is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logging.basicConfig(level = logging.INFO, filename = "{}/preprocess_{}.log".format(
        args.path_to_manifest_output_directory,
        timestamp
    ))

    process_patient_set(
        args.path_to_benign_top_level_directory,
        args.path_to_malignant_top_level_directory,
        args.relative_path_to_frames_directory,
        args.relative_path_to_focus_output_folder,
        args.path_to_manifest_output_directory, 
        timestamp=timestamp,
        upscale_to_maximum=bool(args.upscale))
