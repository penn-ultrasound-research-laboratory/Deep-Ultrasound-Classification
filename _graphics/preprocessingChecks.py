import argparse
import cv2
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utilities.AutomaticSegmentation import get_ROI_debug
from constants.ultrasoundConstants import HSV_COLOR_THRESHOLD
from constants.ultrasoundConstants import (
    IMAGE_TYPE,
    IMAGE_TYPE_LABEL,
    FRAME_LABEL)

from utilities.focus.grayscaleImageFocus import get_scan_area, get_tumor_focus

from pipeline.PatientSampleGenerator import PatientSampleGenerator


def __select_random_grayscale_frame(patient_frames_path, manifest, patient):
    gf = [fr[FRAME_LABEL] for fr in manifest[patient]
          if fr[IMAGE_TYPE_LABEL] == IMAGE_TYPE.GRAYSCALE.value]
    return os.path.join(patient_frames_path, np.random.choice(gf))


def __is_frame_clear(self, frame):
    return frame[IMAGE_TYPE_LABEL] == self.image_type.value


def __grayscale_region_of_interest_graphic(
        benign_top_level_path,
        malignant_top_level_path,
        manifest_path,
        frame_folder,
        rows=1,
        cols=3):

    dirname = os.path.dirname(__file__)

    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)

    K = rows * cols

    mpat = [name for name in os.listdir(malignant_top_level_path)
            if os.path.isdir(os.path.join(malignant_top_level_path, name))]

    bpat = [name for name in os.listdir(benign_top_level_path)
            if os.path.isdir(os.path.join(benign_top_level_path, name))]

    num_malignant = K // 2
    num_benign = K - num_malignant

    # Sample the patients 
    mpat_bar = [(p, "MALIGNANT") for p in np.random.choice(mpat, num_malignant)]
    bpat_bar = [(p, "BENIGN") for p in np.random.choice(bpat, num_benign)]

    # From each patient, randomly grab one grayscale frame
    mpat_frames = [
        (p, __select_random_grayscale_frame(os.path.join(
            malignant_top_level_path, p, frame_folder), manifest, p), tag)
        for (p, tag) in mpat_bar]

    bpat_frames = [
        (p, __select_random_grayscale_frame(os.path.join(
            benign_top_level_path, p, frame_folder), manifest, p), tag)
        for (p, tag) in bpat_bar]

    # Display the randomly chosen frame
    for p, f, label in (mpat_frames + bpat_frames):
        print(f)
        frame = cv2.imread(f, cv2.IMREAD_COLOR)
        
        x_s, y_s, w_s, h_s = get_scan_area(
            frame,
            np.array(HSV_COLOR_THRESHOLD.LOWER.value, np.uint8),
            np.array(HSV_COLOR_THRESHOLD.UPPER.value, np.uint8))
        
        cv2.rectangle(
            frame,
            (x_s, y_s),
            (x_s + w_s, y_s + h_s),
            255,
            2)

        scan_frame = frame[y_s:y_s+h_s, x_s:x_s+w_s]
        print(scan_frame.shape)

        roi_rect, seed_pt = get_ROI_debug(scan_frame)
        x_r, y_r, w_r, h_r = roi_rect

        cv2.rectangle(
            frame,
            (x_r, y_r),
            (x_r + w_r, y_r + h_r),
            255,
            2)
        
        # plt.figure()
        # plt.scatter(x=seed_pt[1], y=seed_pt[0], s=40, c='r')
        # plt.imshow(frame)
        # plt.xticks([])
        # plt.yticks([])



        cv2.imshow("frame", frame)
        cv2.waitKey(0)



if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        "benign_top_level_path",
        type=str,
        help="Path to benign top level directory")

    PARSER.add_argument(
        "malignant_top_level_path",
        type=str,
        help="Path to malignant top level directory")

    PARSER.add_argument(
        "manifest_path",
        type=str,
        help="Absolute path to manifest file"
    )

    PARSER.add_argument(
        "frame_folder",
        type=str,
        help="Folder name (a timestamp by default) containing patient frames for a single run"
    )

    ARGS = PARSER.parse_args()

    # python3 -m _graphics.preprocessingChecks ../100_Cases/ComprehensiveMaBenign/Benign ../100_Cases/ComprehensiveMaBenign/Malignant ../ProcessedDatasets/2018-08-25/manifest_2018-08-25_18-52-25.json 2018-08-25_18-52-25

    __grayscale_region_of_interest_graphic(
        ARGS.benign_top_level_path,
        ARGS.malignant_top_level_path,
        ARGS.manifest_path,
        "frames"
    )
