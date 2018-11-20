import os
import argparse

from math import floor

from constants.modelConstants import TRAIN_TEST_VALIDATION_SPLIT
from constants.ultrasoundConstants import (
    TUMOR_BENIGN,
    TUMOR_MALIGNANT)

import numpy as np


def train_test_split_indices(train_split_percentage, test_split_percentage, number_samples):
    """Return train/test/validate indices for a given number of samples

    Given percentages to use for train/test/(validate) split, return indices that can be used to index
    into identifier arrays at a higher level.

    Note: The sum of train and test does not need to equal 1.0. Any remaining percentage will be used for
    validation indices

    Arguments:
        train_split_percentage               The percentage to use for the train split (0 < x < 1.0)
        test_split_percentage                The percentage to use for the test split (0 < x < 1.0)
        number_samples                       Integer number of samples to split

    Returns:
        Length 3 tuple where each value is a list of indices:
        (
            train_indices,
            test_indices,
            validation_indices
        )
    """

    # Training indices
    train_indices = np.random.choice(
        number_samples,
        floor(train_split_percentage * number_samples),
        replace=False).tolist()

    # Non-training indices
    non_train_indices = [index for index in np.arange(number_samples) if index not in train_indices]

    # Test indices
    test_indices = np.random.choice(
        non_train_indices,
        floor(test_split_percentage * number_samples),
        replace=False).tolist()

    # Validation indices (optional)
    validation_indices = [index for index in np.arange(number_samples) if index not in train_indices + test_indices]

    return (train_indices, test_indices, validation_indices)


def patient_train_test_validation_split(
        benign_top_level_path,
        malignant_top_level_path,
        include_validation=False):
    """Allocate patients to training, test, validation sets.

    The directory structure of the ultrasound data is split into Malignant and Benign folders at the top level. This
    function compiles lists of all the patients and randomly assigns them to training, test, or validation sets using
    the ratios specified in a constants file.

    Arguments:
        benign_top_level_path                absolute path to benign top level folder
        malignant_top_level_path             absolute path to malignant top level folder

    Optional:
        ignore_validation                    Only split the dataset into training/test. No validation partition.

    Returns:
        Dictionary containing arrays: benign_train, benign_test, benign_cval,
            malignant_train, malignant_test, malignant_cval
    """

    malignant_patients = [name for name in os.listdir(malignant_top_level_path)
                          if os.path.isdir(os.path.join(malignant_top_level_path, name))]

    benign_patients = [name for name in os.listdir(benign_top_level_path)
                       if os.path.isdir(os.path.join(benign_top_level_path, name))]

    num_malignant = len(malignant_patients)
    num_benign = len(benign_patients)

    train_indices, test_indices, validation_indices = train_test_split_indices(
        TRAIN_TEST_VALIDATION_SPLIT["TRAIN"],
        TRAIN_TEST_VALIDATION_SPLIT["TEST"],
        num_malignant)

    # Partition always contains train/test
    patient_dataset = {
        "malignant_train": [(malignant_patients[index], TUMOR_MALIGNANT) for index in train_indices],
        "malignant_test": [(malignant_patients[index], TUMOR_MALIGNANT) for index in test_indices]
    }

    # Optionally define validation as a separate partition or append to test partition
    if include_validation:
        patient_dataset["malignant_cval"] = [
            (malignant_patients[index], TUMOR_MALIGNANT) for index in validation_indices]
    else:
        patient_dataset["malignant_test"] += [
            (malignant_patients[index], TUMOR_MALIGNANT) for index in validation_indices]

    # TODO: Code reusability here obviously poor. Split into generic helper method.

    # Number of benign patients differs from number malignant. Generate correctly size index set.
    if num_benign != num_malignant:

        train_indices, test_indices, validation_indices = train_test_split_indices(
            TRAIN_TEST_VALIDATION_SPLIT["TRAIN"],
            TRAIN_TEST_VALIDATION_SPLIT["TEST"],
            num_benign)

    patient_dataset.update({
        "benign_train": [(benign_patients[index], TUMOR_BENIGN) for index in train_indices],
        "benign_test": [(benign_patients[index], TUMOR_BENIGN) for index in test_indices]
    })

    if include_validation:
        patient_dataset["benign_cval"] = [
            (benign_patients[index], TUMOR_BENIGN) for index in validation_indices]
    else:
        patient_dataset["benign_test"] += [
            (benign_patients[index], TUMOR_BENIGN) for index in validation_indices]

    return patient_dataset


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
