import cv2
import json
import logging
import os
import uuid

import numpy as np
import matplotlib.pyplot as plt

import src.utilities.manifest.manifest_utilities as mu

from src.constants.ultrasoundConstants import (
    image_type_to_opencv_color_mode,
    IMAGE_TYPE,
    IMAGE_TYPE_LABEL,
    TUMOR_BENIGN,
    TUMOR_MALIGNANT,
    TUMOR_TYPE_LABEL,
    FOCUS_HASH_LABEL,
    FRAME_LABEL,
    SCALE_LABEL)

from src.constants.modelConstants import (
    DEFAULT_BATCH_SIZE,
    SAMPLE_WIDTH,
    SAMPLE_HEIGHT)

from src.constants.ultrasoundConstants import tumor_integer_label
from src.constants.exceptions.customExceptions import PatientSampleGeneratorException
from src.utilities.image.image_utilities import sample_to_batch

from keras.utils import to_categorical

LOGGER = logging.getLogger('research')

FRAME_START_INDEX = 0
PATIENT_START_INDEX = 0

class PatientSampleGenerator:
    """Generator that returns batches of samples for training and evaluation

    Arguments:
        patient_list                         list of patients. Patient is tuple (patientId, TUMOR_TYPE)
        benign_top_level_path                absolute path to benign directory
        malignant_top_level_path             absolute path to malignant directory

        manifest                             dictionary parsed from JSON containing all information from
                                                image OCR, tumor types, etc
    Optional:
        batch_size                           number of images to output in a batch
        image_data_generator                 preprocessing generator to run on input images
        image_type                           type of image frames to process (IMAGE_TYPE Enum). i.e. grayscale or color
        target_shape                         array containing target shape to use for output samples
        timestamp                            timestamp string to append in focus directory path

        kill_on_last_patient                 Cycle through all matching patients exactly once. Forces generator to act
                                                like single-shot iterator

        use_categorical                      Output class labels one-hot categorical matrix instead of dense numerical

    Returns:
        Tuple containing numpy arrays ((batch_size, (target_shape)), [labels]) 
            where the labels array is length batch_size 

    Raises:
        PatientSampleGeneratorException      for any error generating sample batches
    """

    def __init__(self,
                 patient_list,
                 benign_top_level_path,
                 malignant_top_level_path,
                 manifest,
                 batch_size=DEFAULT_BATCH_SIZE,
                 image_data_generator=None,
                 image_type=IMAGE_TYPE.ALL,
                 target_shape=None,
                 timestamp=None,
                 kill_on_last_patient=False,
                 use_categorical=False):

        self.raw_patient_list = patient_list
        self.manifest = manifest
        self.benign_top_level_path = benign_top_level_path
        self.malignant_top_level_path = malignant_top_level_path
        self.image_type = image_type
        self.timestamp = timestamp
        self.image_data_generator = image_data_generator
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.kill_on_last_patient = kill_on_last_patient
        self.use_categorical = use_categorical

        # Find all the patientIds with at least one frame in the specified IMAGE_TYPE
        # Patient list is unfiltered if IMAGE_TYPE.

        cleared_patients = [
            patient_id for patient_id, _ in patient_list if mu.patient_has_samples(manifest, patient_id, image_type)]

        if not cleared_patients:
            raise PatientSampleGeneratorException(
                "No patients found with focus in image type: {0}".format(self.image_type.value))

        # Determine the total number of cleared frames
        # External functions may need to preallocate memory. Helpful to maintain count of frames.

        patient_frame_counts = [mu.count_valid_frame_samples(manifest[p], image_type) for p in cleared_patients]
        total_num_cleared_frames = sum(patient_frame_counts)

        self.total_num_cleared_frames = total_num_cleared_frames
        self.cleared_patients = cleared_patients

        self.patient_index = PATIENT_START_INDEX
        self.frame_index = FRAME_START_INDEX

        self.__load_current_patient_frames_into_generator()


    def __load_current_patient_frames_into_generator(self):
        """Private method to update current patient information based on patient_index"""

        self.patient_id = self.cleared_patients[self.patient_index]
        self.patient_record = self.manifest[self.patient_id]
        self.patient_type = self.patient_record[0][TUMOR_TYPE_LABEL]

        uncleared_patient_frames = self.manifest[self.cleared_patients[self.patient_index]]
        
        self.patient_frames = mu.get_valid_frame_samples(uncleared_patient_frames, self.image_type)


    def __transition_to_next_patient_frame(self):
        """Move to next frame for patient"""
        self.frame_index += 1
    
        
    def __transition_to_next_patient(self):
        """Move to next patient"""
        self.patient_index += 1
        self.frame_index = FRAME_START_INDEX
        self.__load_current_patient_frames_into_generator()


    def __transition_to_first_patient_reset(self):
        """Move to first patient. Resets the generator"""
        self.patient_index = PATIENT_START_INDEX
        self.frame_index = FRAME_START_INDEX
        self.__load_current_patient_frames_into_generator()


    def __kill_generator(self):
        raise StopIteration("Reached the last patient. Terminating PatientSampleGenerator.")


    def __transition_to_next_patient_frame_state(self, current_is_last_frame, current_is_last_patient):
        if not current_is_last_frame:
            self.__transition_to_next_patient_frame()
        elif current_is_last_patient:
            if self.kill_on_last_patient:
                self.__kill_generator()
            else:
                self.__transition_to_first_patient_reset()
        else:
            self.__transition_to_next_patient()


    def __load_current_frame_image(self, current_frame_color):
        color_mode = image_type_to_opencv_color_mode(current_frame_color)

        loaded_image = cv2.imread(self.patient_frames[self.frame_index][FOCUS_HASH_LABEL], color_mode)

        # TODO: Verify that this makes sense with channel ordering. I thought OpenCV
        # typically used GBR and not RGB

        if current_frame_color == IMAGE_TYPE.GRAYSCALE.value:
            loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_GRAY2RGB)

        return loaded_image


    def __next__(self):

        while True:

            is_last_frame = self.frame_index == len(self.patient_frames) - 1
            is_last_patient = self.patient_index == len(self.cleared_patients) - 1

            current_frame_color = self.patient_frames[self.frame_index][IMAGE_TYPE_LABEL]
            loaded_image = self.__load_current_frame_image(current_frame_color)

            if loaded_image is None or len(loaded_image.shape) < 2:
                # Stored image is corrupted. Skip to next frame.
                LOGGER.info("Skipping due to corruption: %s | frame: %s",
                            self.patient_id,
                            self.patient_frames[self.frame_index][FOCUS_HASH_LABEL])

                self.__transition_to_next_patient_frame_state(is_last_frame, is_last_patient)
                continue

            raw_image_batch = sample_to_batch(
                loaded_image,
                target_shape=self.target_shape,
                upscale_to_target=True,
                batch_size=self.batch_size,
                always_sample_center=True)

            LOGGER.info("Raw Image Batch shape: %s", raw_image_batch.shape)

            # Convert the tumor string label to integer label
            frame_label = tumor_integer_label(self.patient_frames[self.frame_index][TUMOR_TYPE_LABEL])

            # Optional image preprocessing
            if self.image_data_generator is not None:

                # Input raw_image_batch is BGR in standard 0-255 range.
                self.image_data_generator.fit(
                    raw_image_batch,
                    augment=True,
                    rounds=10,
                    seed=None)

                gen = self.image_data_generator.flow(
                    raw_image_batch,
                    batch_size=self.batch_size,
                    shuffle=True)

                # Output of ImageDataGenerator assigned to raw_image_batch is now preprocessed. Values will be
                # negative in range spanning zero if mean normalization is included as part of preprocessing
                # functions.

                raw_image_batch = next(gen)

                LOGGER.debug("Used image data generator to transform input image to shape: {}".format(
                    raw_image_batch.shape))

            LOGGER.info("Training on patient: %s | color: %s | frame: %s",
                        self.patient_id,
                        current_frame_color,
                        self.patient_frames[self.frame_index][FOCUS_HASH_LABEL])

            if self.use_categorical:
                yield (
                    raw_image_batch,
                    to_categorical(np.repeat(frame_label, self.batch_size), num_classes=2))
            else:
                yield (
                    raw_image_batch,
                    np.repeat(frame_label, self.batch_size))

            self.__transition_to_next_patient_frame_state(
                is_last_frame, is_last_patient)

        return
