from constants.ultrasoundConstants import (
    IMAGE_TYPE, 
    TUMOR_BENIGN, 
    TUMOR_MALIGNANT)
from constants.modelConstants import TRAIN_TEST_VALIDATION_SPLIT
from math import floor
import numpy as np
import os

def patient_train_test_validation_split(
    benign_top_level_path, 
    malignant_top_level_path):
    """Allocate patients to training, test, validation sets. 

    The directory structure of the ultrasound data is split into Malignant and Benign
    folders at the top level. This function compiles lists of all the patients and 
    randomly assigns them to training, test, or validation sets using the ratios
    specified in a constants file.  

    Arguments:
        benign_top_level_path: absolute path to benign top level folder
        malignant_top_level_path: absolute path to malignant top level folder

    Returns:
        Dictionary containing arrays: benign_train, benign_test, benign_cval,
            malignant_train, malignant_test, malignant_cval
    """

    malignant_patients = [name for name in os.listdir(malignant_top_level_path) 
        if os.path.isdir(os.path.join(malignant_top_level_path, name))]

    benign_patients = [name for name in os.listdir(benign_top_level_path) 
        if os.path.isdir(os.path.join(benign_top_level_path, name))]

    M = len(malignant_patients)
    B = len(benign_patients)

    train_indices = np.random.choice(M, floor(TRAIN_TEST_VALIDATION_SPLIT["TRAIN"] * M), replace=False).tolist()
    non_train_indices = [ index for index in np.arange(M) if index not in train_indices ]
    test_indices = np.random.choice(non_train_indices, floor(TRAIN_TEST_VALIDATION_SPLIT["TEST"] * M), replace=False).tolist()
    validation_indices = [ index for index in np.arange(M) if index not in (train_indices + test_indices) ]

    patient_dataset = {
        "malignant_train": [ (malignant_patients[index], TUMOR_MALIGNANT) for index in train_indices ],
        "malignant_test": [ (malignant_patients[index], TUMOR_MALIGNANT) for index in test_indices ],
        "malignant_cval": [ (malignant_patients[index], TUMOR_MALIGNANT) for index in validation_indices ]
    }    

    # TODO: Code reusability here obviously poor. Split into generic helper method. 

    if B != M:
        train_indices = np.random.choice(B, floor(TRAIN_TEST_VALIDATION_SPLIT["TRAIN"] * B), replace=False).tolist()
        non_train_indices = [ index for index in np.arange(B) if index not in train_indices ]
        test_indices = np.randoB.choice(non_train_indices, floor(TRAIN_TEST_VALIDATION_SPLIT["TEST"] * B), replace=False).tolist()
        validation_indices = [ index for index in np.arange(B) if index not in (train_indices + test_indices) ]

    patient_dataset.update({
        "benign_train": [(benign_patients[index], TUMOR_BENIGN) for index in train_indices],
        "benign_test": [(benign_patients[index], TUMOR_BENIGN) for index in test_indices],
        "benign_cval": [(benign_patients[index], TUMOR_BENIGN) for index in validation_indices]
    })

    return patient_dataset
