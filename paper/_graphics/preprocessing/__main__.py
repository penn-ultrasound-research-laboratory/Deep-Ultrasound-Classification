import argparse

from _graphics.preprocessing.preprocessing_checks import __grayscale_region_of_interest_graphic

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
    "frames",
    rows = 1,
    cols= 5
)
