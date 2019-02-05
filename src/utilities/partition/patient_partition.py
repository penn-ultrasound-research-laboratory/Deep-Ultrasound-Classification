import os
import argparse

from math import floor

from src.constants.model import TRAIN_TEST_VALIDATION_SPLIT
from src.constants.ultrasound import (
    TUMOR_BENIGN,
    TUMOR_MALIGNANT)

import numpy as np


def train_test_split_indices(
    train_split, 
    number_samples,
    random_seed=None):
    """Return train/test/validate indices for a given number of samples

    Given percentages to use for train/test/(validate) split, return indices that can be used to index
    into identifier arrays at a higher level.

    Note: The sum of train and test does not need to equal 1.0. Any remaining percentage will be used for
    validation indices

    Arguments:
        train_split                         The percentage to use for the train split (0 < x < 1.0)
        number_samples                      Integer number of samples to split

    Returns:
        Length 3 tuple where each value is a list of indices:
        (
            train_indices,
            test_indices,
            validation_indices
        )
    """

    # Optionally set random seed
    if random_seed is not None:
         np.random.seed(random_seed)

    # Training indices
    train_indices = np.random.choice(
        number_samples,
        floor(train_split * number_samples),
        replace=False).tolist()

    # Test indices
    test_indices = [index for index in np.arange(number_samples) if index not in train_indices]

    return (train_indices, test_indices)


def patient_train_test_split(
        benign_path,
        malignant_path,
        train_split,
        random_seed=None):
    """Allocate patients to training, test, validation sets.

    The directory structure of the ultrasound data is split into Malignant and Benign folders at the top level. This
    function compiles lists of all the patients and randomly assigns them to training, test, or validation sets using
    the ratios specified in a constants file.

    Arguments:
        benign_path                absolute path to benign top level folder
        malignant_path             absolute path to malignant top level folder

    Returns:
        Dictionary containing arrays: benign_train, benign_test, benign_cval,
            malignant_train, malignant_test, malignant_cval
    """

    malignant_patients = [_ for _ in os.listdir(malignant_path) if os.path.isdir(os.path.join(malignant_path, _))]
    benign_patients = [_ for _ in os.listdir(benign_path) if os.path.isdir(os.path.join(benign_path, _))]

    num_malignant = len(malignant_patients)
    num_benign = len(benign_patients)

    train_indices_mal, test_indices_mal = train_test_split_indices(
        train_split,
        num_malignant,
        random_seed=random_seed)

    train_indices_ben, test_indices_ben = train_test_split_indices(
        train_split,
        num_benign,
        random_seed=random_seed)

    # Partition always contains train/test
    patient_split = {
        "benign_train": [(benign_patients[_], TUMOR_BENIGN) for _ in train_indices_ben],
        "benign_test": [(benign_patients[_], TUMOR_BENIGN) for _ in test_indices_ben],
        "malignant_train": [(malignant_patients[_], TUMOR_MALIGNANT) for _ in train_indices_mal],
        "malignant_test": [(malignant_patients[_], TUMOR_MALIGNANT) for _ in test_indices_mal]
    }

    patient_split.update({
        "train": patient_split["benign_train"] + patient_split["malignant_train"],
        "test": patient_split["benign_test"] + patient_split["malignant_test"]
    })

    return patient_split