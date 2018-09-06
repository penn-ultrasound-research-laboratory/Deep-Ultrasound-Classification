import cv2
import json
import logging
import os
import uuid

import numpy as np
import matplotlib.pyplot as plt

from constants.ultrasoundConstants import (
    IMAGE_TYPE,
    IMAGE_TYPE_LABEL,
    TUMOR_BENIGN,
    TUMOR_MALIGNANT,
    TUMOR_TYPE_LABEL,
    FOCUS_HASH_LABEL,
    FRAME_LABEL,
    SCALE_LABEL)
from constants.modelConstants import (
    DEFAULT_BATCH_SIZE,
    SAMPLE_WIDTH,
    SAMPLE_HEIGHT)
from constants.ultrasoundConstants import tumor_integer_label
from constants.exceptions.customExceptions import PatientSampleGeneratorException
from utilities.imageUtilities import image_random_sampling_batch
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

LOGGER = logging.getLogger('research')

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

        kill_on_last_patient                 Cycle through all matching patients exactly once. Forces generator to act                                      like single-shot iterator

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
        
        if image_type is IMAGE_TYPE.ALL:
            cleared_patients = [patient[0] for patient in patient_list]
        else:
            cleared_patients = list(filter(
                lambda patient: any([self.__is_frame_clear(frame) for frame in manifest[patient]]), 
                [patient[0] for patient in patient_list]))

        if not cleared_patients:
            raise PatientSampleGeneratorException(
                "No patients found with focus in image type: {0}".format(self.image_type.value))

        # Determine the total number of cleared frames 
        # External functions may need to preallocate memory. Helpful to maintain count of frames.
        total_num_cleared_frames = 0
        for patient in cleared_patients:
            total_num_cleared_frames += len([frame for frame in manifest[patient] if self.__is_frame_clear(frame)])
        
        self.total_num_cleared_frames = total_num_cleared_frames
        self.cleared_patients = cleared_patients
        self.patient_index = self.frame_index = 0

        self.__load_current_patient_frames_into_generator()

    def __is_frame_clear(self, frame):
        return frame[IMAGE_TYPE_LABEL] == self.image_type.value and FOCUS_HASH_LABEL in frame

    def __load_current_patient_frames_into_generator(self):
        """Private method to update current patient information based on patient_index"""

        self.patient_id = self.cleared_patients[self.patient_index]
        self.patient_record = self.manifest[self.patient_id]
        self.patient_type = self.patient_record[0][TUMOR_TYPE_LABEL]

        all_patient_frames = self.manifest[self.cleared_patients[self.patient_index]]

        # Find all patient's frames matching the specified image type
        if self.image_type is IMAGE_TYPE.ALL:
            self.patient_frames = all_patient_frames
        else:
            self.patient_frames = [
                frame for frame in all_patient_frames if frame[IMAGE_TYPE_LABEL] == self.image_type.value]

        # Frames must contain an image focus
        self.patient_frames = [frame for frame in self.patient_frames if FOCUS_HASH_LABEL in frame]


    def __move_to_next_generator_patient_frame_state(self, current_is_last_frame, current_is_last_patient):
        if not current_is_last_frame:
            # Move to next frame for patient
            self.frame_index += 1
        elif current_is_last_patient:
            # Last patient frame. Loop back to first patient or terminate
            if self.kill_on_last_patient:
                raise StopIteration("Reached the last patient. Terminating PatientSampleGenerator.")

            self.patient_index = 0
            self.frame_index = 0
            self.__load_current_patient_frames_into_generator()
        else:
            # Move to next patient
            self.patient_index += 1
            self.frame_index = 0
            self.__load_current_patient_frames_into_generator()

    def __next__(self):

        while True:

            current_frame_color = self.patient_frames[self.frame_index][IMAGE_TYPE_LABEL]
            is_last_frame = self.frame_index == len(self.patient_frames) - 1
            is_last_patient = self.patient_index == len(self.cleared_patients) - 1
                
            color_mode = (
                cv2.IMREAD_COLOR if current_frame_color == IMAGE_TYPE.COLOR.value
                else cv2.IMREAD_GRAYSCALE)

            loaded_image = cv2.imread("{}".format(self.patient_frames[self.frame_index][FOCUS_HASH_LABEL]), color_mode)

            # All images are processed as RGB
            # TODO: Verify that this makes sense with channel ordering. I thought OpenCV 
            # typically used GBR and not RGB

            if current_frame_color == IMAGE_TYPE.GRAYSCALE.value:
                loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_GRAY2RGB)

            if loaded_image is None or len(loaded_image.shape) < 2:
                logging.info("Skipping due to corruption: {} | frame: {}".format(self.patient_id, self.patient_frames[self.frame_index][FOCUS_HASH_LABEL]))
                # Stored image is corrupted. Skip to next frame. 
                
                self.__move_to_next_generator_patient_frame_state(is_last_frame, is_last_patient)
                continue

            else:
                
                raw_image_batch = image_random_sampling_batch(
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

                    # Output of ImageDataGenerator assigned to raw_image_batch is now preprocessed. Values will be negative in range spanning zero if mean normalization is included as part of preprocessing functions. 

                    raw_image_batch = next(gen)

                    logging.debug("Used image data generator to transform input image to shape: {}".format(raw_image_batch.shape))

            logging.info("Training on patient: {} | color: {} | frame: {}".format(self.patient_id, current_frame_color, self.patient_frames[self.frame_index][FOCUS_HASH_LABEL]))

            if self.use_categorical:
                yield (
                    raw_image_batch, 
                    to_categorical(np.repeat(frame_label, self.batch_size), num_classes=2))
            else:
                yield (
                    raw_image_batch, 
                    np.repeat(frame_label, self.batch_size))

            self.__move_to_next_generator_patient_frame_state(is_last_frame, is_last_patient)

        return



if __name__ == "__main__":

    dirname = os.path.dirname(__file__)

    with open(os.path.abspath("../ProcessedDatasets/2018-08-25/manifest_2018-08-25_18-52-25.json"), "r") as fp:
        manifest = json.load(fp)

    logging.basicConfig(level = logging.INFO, filename = "./main_output.log")

    # W/ aim of generating graphics for paper / email. Featurwise normalize according to mean or std. That should be used only in image preprocessing pipeline. Issue is that negative scaled values are meaningless when saving to file or displaying as negative values are truncated to zero. 

    image_data_generator = ImageDataGenerator(
        horizontal_flip = True,
        vertical_flip = True)

    BATCH_SIZE = 5
    MAX_PLOTTING_ROWS = 10

    # Limit to small subset of patients

    patient_sample_generator = next(PatientSampleGenerator([
        ("01PER2043096", "BENIGN"),
        ("79BOY3049163", "MALIGNANT"),
        ("93KUD3041008", "MALIGNANT")],
    os.path.join(dirname, "../../100_Cases/ComprehensiveMaBenign/Benign"),
    os.path.join(
        dirname, "../../100_Cases/ComprehensiveMaBenign/Malignant"),
    manifest,
    target_shape=[220, 220],
    image_type=IMAGE_TYPE.ALL,
    image_data_generator=image_data_generator,
    timestamp="2018-08-25_18-52-25",
    batch_size=BATCH_SIZE,
    kill_on_last_patient=True
))

    count = 0
    plot_rows = []
    try:
        while count < MAX_PLOTTING_ROWS:
            raw_image_batch, labels = next(patient_sample_generator)
                
            split = np.split(raw_image_batch, raw_image_batch.shape[0], axis=0)
            split = [ np.squeeze(img, axis=0) for img in split]

            # cv2.imshow("sample", np.hstack(split))
            # cv2.waitKey(0)

            plot_rows.append(np.hstack(split))

            count += 1 

    except Exception as e:
        print(e)

    sample_thumbnails = np.vstack(plot_rows)

    # cv2.imshow("sample", sample_thumbnails)
    # cv2.waitKey(0)

    cv2.imwrite('thumbnails_{}.png'.format(uuid.uuid4()), sample_thumbnails)