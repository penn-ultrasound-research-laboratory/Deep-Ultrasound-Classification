import argparse

from utilities.partition.patient_partition import patient_train_test_validation_split

PARSER = argparse.ArgumentParser()

PARSER.add_argument(
    "benign_top_level_path",
    type=str,
    help="Path to benign top level directory")

PARSER.add_argument(
    "malignant_top_level_path",
    type=str,
    help="Path to malignant top level directory")

ARGS = PARSER.parse_args()

INDEX_DICTIONARY = patient_train_test_validation_split(
    ARGS.benign_top_level_path,
    ARGS.malignant_top_level_path,
    include_validation=False)

ALL_PATIENTS = set()
for key in INDEX_DICTIONARY.keys():
    for patient in INDEX_DICTIONARY[key]:
        ALL_PATIENTS.add(patient[0])

# Set size should be the length of all patients (each patient used exactly once)
print(len(ALL_PATIENTS))