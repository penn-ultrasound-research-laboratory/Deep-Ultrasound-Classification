import argparse
import logging

from datetime import datetime

from src.process_ultrasound import process_patient_set

PARSER = argparse.ArgumentParser()

PARSER.add_argument("path_to_benign_dir",
                    help="absolute path to top level directory containing benign patient folders")

PARSER.add_argument("path_to_malignant_dir",
                    help="absolute path to top level directory containing malignant patient folders")

PARSER.add_argument("-frames", "--relative_path_to_frames_directory",
                    type=str,
                    default="frames",
                    help="relative path from the patient folder to patient frames folder")

PARSER.add_argument("-focus", "--rel_path_to_focus_output_folder",
                    type=str,
                    default="focus",
                    help="relative path from the patient folder to frame focus output folder ")

PARSER.add_argument("-out", "--path_to_manifest_output_dir",
                    type=str,
                    default=".",
                    help="absolute path to manifest/failure log output directory")

PARSER.add_argument("-time",
                    "--timestamp",
                    type=str,
                    default=None,
                    help="timestamp to use instead of generating one using the current time")

PARSER.add_argument("-up",
                    "--upscale",
                    type=int,
                    default=0,
                    help="Boolean indicating whether to upscale frame focuses to the maximum value in the manifest")

## Missing functionality to wipe out old folders, manifests, error logs

ARGS = PARSER.parse_args()

TIMESTAMP = ARGS.timestamp if ARGS.timestamp is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(level=logging.INFO, filename="{}/preprocess_{}.log".format(
    ARGS.path_to_manifest_output_dir,
    TIMESTAMP
))

process_patient_set(
    ARGS.path_to_benign_dir,
    ARGS.path_to_malignant_dir,
    ARGS.relative_path_to_frames_directory,
    ARGS.rel_path_to_focus_output_folder,
    ARGS.path_to_manifest_output_dir,
    timestamp=TIMESTAMP,
    upscale_to_maximum=bool(ARGS.upscale))
