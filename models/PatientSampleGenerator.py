from constants.ultrasoundConstants import (
    IMAGE_TYPE,
    IMAGE_TYPE_LABEL,
    TUMOR_BENIGN,
    TUMOR_MALIGNANT,
    TUMOR_TYPE_LABEL,
    FOCUS_HASH_LABEL)
from constants.modelConstants import DEFAULT_BATCH_SIZE
from constants.ultrasoundConstants import tumor_integer_label
from constants.exceptions.customExceptions import PatientSampleGeneratorException
from utilities.imageUtilities import image_random_sampling_batch
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
import cv2, os, json

class PatientSampleGenerator:
    """Generator that returns batches of samples for training and evaluation
    
    Arguments:
        patient_list: list of patients. Patient is tuple (patientId, TUMOR_TYPE)
        benign_top_level_path: absolute path to benign directory
        malignant_top_level_path: absolute path to malignant directory
        manifest: dictionary parsed from JSON containing all information from image OCR, tumor types, etc
        batch_size: (optional) number of images to output in a batch
        image_data_generator: (optional) preprocessing generator to run on input images
        image_type: (optional) type of image frames to process (IMAGE_TYPE Enum). i.e. grayscale or color
        number_channels (optional) number of color channels. Should be 3 if color, 1 if grayscale. 
            3 by default to match image_type=COLOR default
        target_shape: (optional) array containing target shape to use for output samples
        timestamp: (optional) optional timestamp string to append in focus directory path. i.e. "*/focus_timestamp/*

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
        image_type=IMAGE_TYPE.COLOR,
        number_channels=3, 
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
        self.number_channels = number_channels
        self.kill_on_last_patient = kill_on_last_patient
        self.use_categorical = use_categorical

        # Find all the patientIds with at least one frame in the specified IMAGE_TYPE

        cleared_patients = list(filter(
            lambda patient: any([frame[IMAGE_TYPE_LABEL] == image_type.value for frame in manifest[patient]]), 
            [patient[0] for patient in patient_list]))

        if len(cleared_patients) == 0:
            raise PatientSampleGeneratorException(
                "No patients found with focus in image type: {0}".format(self.image_type.value))
        
        self.cleared_patients = cleared_patients
        self.patient_index = self.frame_index = 0

        self.__update_current_patient_information()


    def __update_current_patient_information(self):
        """Private method to update current patient information based on patient_index"""

        self.patient_id = self.cleared_patients[self.patient_index]
        self.patient_record = self.manifest[self.patient_id]
        self.patient_type = self.patient_record[0][TUMOR_TYPE_LABEL]

        # Find all patient's frames matching the specified image type
        self.patient_frames = list(filter(
            lambda frame: frame[IMAGE_TYPE_LABEL] == self.image_type.value,
            self.manifest[self.cleared_patients[self.patient_index]]
        ))


    def __next__(self):

        while True:

            skip_flag = False
            is_last_frame = self.frame_index == len(self.patient_frames) - 1
            is_last_patient = self.patient_index == len(self.cleared_patients) - 1

            loaded_image = cv2.imread("{}/{}/{}/{}".format(
                (self.benign_top_level_path if self.patient_type == TUMOR_BENIGN else self.malignant_top_level_path),
                self.patient_id,
                ("focus" if self.timestamp is None else "focus_{}".format(self.timestamp)),
                self.patient_frames[self.frame_index][FOCUS_HASH_LABEL]
            ),
                (cv2.IMREAD_COLOR if self.image_type.value == IMAGE_TYPE.COLOR.value else cv2.IMREAD_GRAYSCALE))

            if loaded_image is None or len(loaded_image.shape) < 2:
                # Stored image is corrupted. Skip to next frame. 
                skip_flag = True
            else:
                print("Training on patient: {} | frame: {}".format(self.patient_id, self.frame_index))

                raw_image_batch = image_random_sampling_batch(
                    loaded_image, 
                    target_shape=self.target_shape,
                    use_min_dimension=(self.target_shape is None),
                    batch_size=self.batch_size)

                print("Generated raw image batch size: {}".format(raw_image_batch.shape))

                # Weird indexing is due to shape mismatches between specified target shape 
                # and extra dimension of batch size. Determine row/column padding
                # Target shape is [rows, columns]
                # iImage batch shape is [batch_size, rows, columns]

                pad_rows = np.max(self.target_shape[0]-raw_image_batch.shape[1], 0)
                pad_cols = np.max(self.target_shape[1]-raw_image_batch.shape[2], 0)

                padding_tuple = (
                    (0, 0),
                    (pad_rows // 2, pad_rows // 2 + pad_rows % 2), 
                    (pad_cols // 2, pad_cols // 2 + pad_cols % 2))
                
                if self.image_type is IMAGE_TYPE.COLOR:
                    padding_tuple += ((0,0),)

                padded_image_batch = np.pad(
                    raw_image_batch,
                    padding_tuple,
                    "constant",
                    constant_values=0)

                # Convert the tumor string label to integer label
                frame_label = tumor_integer_label(self.patient_frames[self.frame_index][TUMOR_TYPE_LABEL])

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
                print("Patient: {} | Frame: {} | label: {}".format(self.patient_id, self.frame_index, frame_label))
                if self.use_categorical:
                    yield (padded_image_batch, to_categorical(np.repeat(frame_label, self.batch_size), num_classes=2))
                else:
                    yield (padded_image_batch, np.repeat(frame_label, self.batch_size))
            else:
                continue

        return



if __name__ == "__main__":

    dirname = os.path.dirname(__file__)

    with open(os.path.abspath("processedData//manifest_COMPLETE_2018-07-11_18-51-03.json"), "r") as fp:
        manifest = json.load(fp)

    image_data_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True)

    BATCH_SIZE = 5

    patient_sample_generator = next(PatientSampleGenerator(
        [("30BRO3007451", "BENIGN"), ("01PER2043096", "BENIGN")],
        os.path.join(dirname, "../../100_Cases/ComprehensiveMaBenign/Benign"),
        os.path.join(dirname, "../../100_Cases/ComprehensiveMaBenign/Malignant"),
        manifest,
        target_shape=[200, 200],
        number_channels=1,
        image_type=IMAGE_TYPE.GRAYSCALE,
        image_data_generator=image_data_generator,
        timestamp="2018-07-11_18-51-03",
        batch_size=BATCH_SIZE
    ))
    
    raw_image_batch, labels = next(patient_sample_generator)

    for i in range(BATCH_SIZE):
        cv2.imshow(str(labels[i]), raw_image_batch[i])
        cv2.waitKey(0)    

    raw_image_batch, labels = next(patient_sample_generator)

    for i in range(BATCH_SIZE):
        cv2.imshow(str(labels[i]), raw_image_batch[i])
        cv2.waitKey(0)    

    raw_image_batch, labels = next(patient_sample_generator)

    for i in range(BATCH_SIZE):
        cv2.imshow(str(labels[i]), raw_image_batch[i])
        cv2.waitKey(0)    

    raw_image_batch, labels = next(patient_sample_generator)

    for i in range(BATCH_SIZE):
        cv2.imshow(str(labels[i]), raw_image_batch[i])
        cv2.waitKey(0)    
