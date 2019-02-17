import argparse

from math import floor

from tensorflow.random import shuffle

from constants.model import TRAIN_TEST_VALIDATION_SPLIT
from constants.ultrasound import (
    TUMOR_BENIGN,
    TUMOR_MALIGNANT)

import tensorflow as tf
import numpy as np


def train_test_validation_indices(train_split, validation_split, N, random_seed=None):

    splits = np.floor(np.array([train_split * N, validation_split * N]))
    print(splits)
    splits = np.append(splits, N - np.sum(splits)).astype(np.uint32)

    print(splits)
    train, validation, test = tf.split(
        shuffle(list(range(N)), seed=random_seed),
        splits
    )

    return (train, validation, test)


def train_test_split_indices(train_split, N, random_seed=None):

    splits = np.floor([train_split * N])
    np.append(splits, N - int(np.sum(splits)))

    print(splits)
    train, test = tf.split(
        shuffle(list(range(N)), seed=random_seed),
        splits
    )

    return (train, test)


def patient_train_test_split(
        benign_patients,
        malignant_patients,
        train_split,
        validation_split = None,
        random_seed=None):
    """Allocate patients to training, test, validation sets.

    The directory structure of the ultrasound data is split into Malignant and Benign folders at the top level. This
    function compiles lists of all the patients and randomly assigns them to training, test, or validation sets using
    the ratios specified in a constants file.

    Arguments:
        benign_patients                     list of benign patient ids
        malignant_patients                  list of malignant patient ids

    Returns:
        Dictionary containing arrays: benign_train, benign_test, benign_cval,
            malignant_train, malignant_test, malignant_cval
    """

    num_benign = len(benign_patients)
    num_malignant = len(malignant_patients)

    if validation_split:
        train_ben, val_ben, test_ben = train_test_validation_indices(
            train_split,
            validation_split,
            num_benign,
            random_seed=random_seed)
        
        train_mal, val_mal, test_mal = train_test_validation_indices(
            train_split,
            validation_split,
            num_malignant,
            random_seed=random_seed)

    else:
        train_ben, test_ben = train_test_split_indices(
            train_split,
            num_benign,
            random_seed=random_seed)
        
        train_mal, test_mal = train_test_split_indices(
            train_split,
            num_malignant,
            random_seed=random_seed)

    # Partition always contains train/test
    ps = {
        "benign_train": [(benign_patients[_], TUMOR_BENIGN) for _ in train_ben],
        "benign_test": [(benign_patients[_], TUMOR_BENIGN) for _ in test_ben],
        "malignant_train": [(malignant_patients[_], TUMOR_MALIGNANT) for _ in train_mal],
        "malignant_test": [(malignant_patients[_], TUMOR_MALIGNANT) for _ in test_mal]
    }

    if validation_split:
        ps.update({
            "benign_validation": [(benign_patients[_], TUMOR_BENIGN) for _ in val_ben],
            "malignant_validation": [(malignant_patients[_], TUMOR_MALIGNANT) for _ in val_mal],
        })

    ps.update({
        "train": ps["benign_train"] + ps["malignant_train"],
        "test": ps["benign_test"] + ps["malignant_test"]
    })

    if validation_split:
        ps.update({
            "validation": ps["benign_validation"] + ps["malignant_validation"]
        })

    return ps
