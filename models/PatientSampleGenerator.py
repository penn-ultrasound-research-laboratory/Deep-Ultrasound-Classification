import numpy as np
import matplotlib.pyplot as plt

from constants.ultrasoundConstants import (
    IMAGE_TYPE,
    IMAGE_TYPE_LABEL,
    TUMOR_BENIGN,
    TUMOR_MALIGNANT,
    TUMOR_TYPE_LABEL,
    FOCUS_HASH_LABEL,
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

import cv2
import json
import logging
import os
import uuid

logger = logging.getLogger('research')

class PatientSampleGenerator:
    """Generator that returns batches of samples for training and evaluation
    
    Arguments:
        patient_list: list of patients. Patient is tuple (patientId, TUMOR_TYPE)
        benign_top_level_path: absolute path to benign directory
        malignant_top_level_path: absolute path to malignant directory
        manifest: dictionary parsed from JSON containing all information from image OCR, tumor types, etc

    Optional:
        batch_size: number of images to output in a batch
        image_data_generator: preprocessing generator to run on input images
        image_type: type of image frames to process (IMAGE_TYPE Enum). i.e. grayscale or color
        target_shape: array containing target shape to use for output samples
        timestamp: optional timestamp string to append in focus directory path. i.e. "*/focus_timestamp/*
        kill_on_last_patient: Cycle through all matching patients exactly once. Forces generator to act like single-shot iterator
        use_categorical: Output class labels as one-hot categorical matrix instead of numerical label 
        auto_resize_to_manifest_scale_max: use the maximum scale value in the manifest as a reference 

    Returns:
        Tuple containing numpy arrays ((batch_size, (target_shape)), [labels]) 
            where the labels array is length batch_size 

    Raises:
        PatientSampleGeneratorException for any error generating sample batches
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
        use_categorical=False,
        auto_resize_to_manifest_scale_max=False):
        
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
        self.auto_resize_to_manifest_scale_max = auto_resize_to_manifest_scale_max

        # Find all the patientIds with at least one frame in the specified IMAGE_TYPE
        # Patient list is unfiltered if IMAGE_TYPE.ALL
        if image_type is IMAGE_TYPE.ALL:
            cleared_patients = [patient[0] for patient in patient_list]
        else:
            cleared_patients = list(filter(
                lambda patient: any([frame[IMAGE_TYPE_LABEL] == image_type.value for frame in manifest[patient]]), 
                [patient[0] for patient in patient_list]))

        if len(cleared_patients) == 0:
            raise PatientSampleGeneratorException(
                "No patients found with focus in image type: {0}".format(self.image_type.value))


        if auto_resize_to_manifest_scale_max:
            manifest_scale_max = 0.0
            for patient in cleared_patients:            
                try:
                    candidate_max = max([frame[SCALE_LABEL] for frame in manifest[patient]])
                    if candidate_max > manifest_scale_max: 
                        manifest_scale_max = candidate_max
                except:
                    pass

            logging.info("Maximum scale for patient partition: {}".format(manifest_scale_max))
            self.manifest_scale_max = manifest_scale_max

        self.cleared_patients = cleared_patients
        self.patient_index = self.frame_index = 0

        self.__update_current_patient_information()


    def __update_current_patient_information(self):
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

    def __next__(self):

        while True:

            skip_flag = False

            current_frame_color = self.patient_frames[self.frame_index][IMAGE_TYPE_LABEL]
            is_last_frame = self.frame_index == len(self.patient_frames) - 1
            is_last_patient = self.patient_index == len(self.cleared_patients) - 1

            top_level_path = (
                self.benign_top_level_path if self.patient_type == TUMOR_BENIGN 
                else self.malignant_top_level_path)

            focus_directory = (
                "focus" if self.timestamp is None 
                else "focus_{}".format(self.timestamp))

            color_mode = (
                cv2.IMREAD_COLOR if current_frame_color == IMAGE_TYPE.COLOR.value
                else cv2.IMREAD_GRAYSCALE)

            loaded_image = cv2.imread("{}".format(
                self.patient_frames[self.frame_index][FOCUS_HASH_LABEL]),
                color_mode)

            # All images are processed as RGB
            # TODO: Verify that this makes sense with channel ordering. I thought OpenCV 
            # typically used GBR and not RGB

            if current_frame_color == IMAGE_TYPE.GRAYSCALE.value:
                loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_GRAY2RGB)


            if loaded_image is None or len(loaded_image.shape) < 2:
                logging.info("Skipping due to corruption: {} | frame: {}".format(self.patient_id, self.patient_frames[self.frame_index][FOCUS_HASH_LABEL]))
                # Stored image is corrupted. Skip to next frame. 
                skip_flag = True
            else:

                if self.auto_resize_to_manifest_scale_max:
                    try:
                        frame_scale = self.patient_frames[self.frame_index][SCALE_LABEL]
                        upscale_ratio = self.manifest_scale_max / frame_scale

                        logging.debug("Upscale Ratio: {}".format(upscale_ratio))

                        loaded_image = cv2.resize(
                            loaded_image, 
                            None, 
                            fx=upscale_ratio, 
                            fy=upscale_ratio, 
                            interpolation=cv2.INTER_CUBIC)

                    except:
                        logger.error("Unable to auto-resize. Frame {} does not have scale label".format(
                            self.patient_frames[self.frame_index][FOCUS_HASH_LABEL]
                        ))

                raw_image_batch = image_random_sampling_batch(
                    loaded_image, 
                    target_shape=self.target_shape,
                    upscale_to_target=True,
                    batch_size=self.batch_size,
                    always_sample_center=True)

                logger.info("Raw Image Batch shape: {}".format(raw_image_batch.shape))

                # Convert the tumor string label to integer label
                frame_label = tumor_integer_label(self.patient_frames[self.frame_index][TUMOR_TYPE_LABEL])

                # Optional image preprocessing
                if self.image_data_generator is not None:

                    self.image_data_generator.fit(
                        raw_image_batch,
                        augment=True,
                        rounds=10,
                        seed=None)

                    gen = self.image_data_generator.flow(
                        raw_image_batch,
                        batch_size=self.batch_size, 
                        shuffle=True)

                    raw_image_batch = next(gen)

                    logging.debug("Used image data generator to transform input image to shape: {}".format(raw_image_batch.shape))

                # Always augment by providing several common gradient transforms on the input
                # Randomly sample an images from the batch and generate gradients from the batch 

                # print(self)
                # cv2.imshow('img', raw_image_batch[np.random.randint(self.batch_size)])
                # cv2.waitKey(0)

                # gradient_batch = np.stack([
                #     cv2.Laplacian(raw_image_batch[np.random.randint(self.batch_size)], cv2.CV_64F),
                #     cv2.Sobel(raw_image_batch[np.random.randint(self.batch_size)], cv2.CV_64F, 1, 0, ksize=3),
                #     cv2.Sobel(raw_image_batch[np.random.randint(self.batch_size)], cv2.CV_64F, 0, 1, ksize=3)
                # ], 
                # axis=0)

                # raw_image_batch = np.concatenate((raw_image_batch, gradient_batch), axis=0)

            if not is_last_frame:

                self.frame_index += 1
            
            elif is_last_patient:

                if self.kill_on_last_patient:
                    raise StopIteration("Reached the last patient. Terminating PatientSampleGenerator.")

                self.patient_index = 0
                self.frame_index = 0
                self.__update_current_patient_information()
            else:
                self.patient_index += 1
                self.frame_index = 0
                self.__update_current_patient_information()
           
            # Class outputs must be categorical. 
            if not skip_flag:
                logging.info("Training on patient: {} | color: {} | frame: {}".format(self.patient_id, current_frame_color, self.patient_frames[self.frame_index][FOCUS_HASH_LABEL]))

                if self.use_categorical:
                    yield (raw_image_batch, to_categorical(np.repeat(frame_label, self.batch_size), num_classes=2))
                else:
                    yield (raw_image_batch, np.repeat(frame_label, self.batch_size))
            else:
                continue

        return



if __name__ == "__main__":

    dirname = os.path.dirname(__file__)

    with open(os.path.abspath("../ProcessedDatasets/2018-08-04_16-19-39/manifest_2018-08-04_16-19-39.json"), "r") as fp:
        manifest = json.load(fp)

    image_data_generator = ImageDataGenerator(
        featurewise_center = True,
        featurewise_std_normalization = True,
        horizontal_flip = True,
        vertical_flip = True)

    BATCH_SIZE = 5

    patient_sample_generator = next(PatientSampleGenerator(
        [("30BRO3007451", "BENIGN"),
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
        timestamp="2018-08-04_16-19-39",
        batch_size=BATCH_SIZE,
        auto_resize_to_manifest_scale_max=True,
        kill_on_last_patient=True
    ))

    MAX_PLOTTING_ROWS = 10
    

    count = 0
    plot_rows = []
    try:
        while count < MAX_PLOTTING_ROWS:
            raw_image_batch, labels = next(patient_sample_generator)
                
            split = np.split(raw_image_batch, raw_image_batch.shape[0], axis=0)
            split = [ np.squeeze(img, axis=0) for img in split]
            print(split[0].shape)

            # cv2.imshow("sample", np.hstack(split))
            # cv2.waitKey(0)

            plot_rows.append(np.hstack(split))

            count += 1 

    except Exception as e:
        print(e)

    sample_thumbnails = np.vstack(plot_rows)

    cv2.imshow("sample", sample_thumbnails)
    cv2.waitKey(0)

    cv2.imwrite('thumbnails_{}.png'.format(uuid.uuid4()), cv2.cvtColor(sample_thumbnails, COLOR_HSV2BGR, sample_thumbnails))