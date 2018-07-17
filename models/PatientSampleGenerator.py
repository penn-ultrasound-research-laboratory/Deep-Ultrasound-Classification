from constants.ultrasoundConstants import *
from keras.preprocessing.image import ImageDataGenerator
from constants.exceptions.customExceptions import PatientSampleGeneratorException
import numpy as np
import cv2

class PatientSampleGenerator:
    '''Generator that returns batches of samples for training and evaluation
    
    Arguments:
        patient_list: list of patients. Patient is tuple (patientId, TUMOR_TYPE)
        top_level_path:
        manifest:
        image_type:
        timestamp: (optional) optional timestamp string to append in focus directory path. i.e. "*/focus_timestamp/*

    Returns:

    '''

    def __init__(self, patient_list, benign_top_level_path, malignant_top_level_path, manifest, image_type, timestamp=None):
        self.image_type = image_type
        self.raw_patient_list = patient_list
        self.manifest = manifest
        self.benign_top_level_path = benign_top_level_path
        self.malignant_top_level_path = malignant_top_level_path
        self.timestamp = timestamp
        # self.grayscaleDataGenerator = ImageDataGenerator()
        # self.colorDataGenerator = ImageDataGenerator()

        # Find all the patientIds with at least one frame in the specified IMAGE_TYPE
        cleared_patients = [patient[0] for patient in self.raw_patient_list if
                            any([frame[IMAGE_TYPE_LABEL] == self.image_type.value
                                 for frame in self.manifest[patient[0]]])]

        if len(cleared_patients) == 0:
            raise PatientSampleGeneratorException('No patients found with focus in image type: {0}'.format(self.image_type.value))
        
        self.cleared_patients = cleared_patients

        self.patient_index = 0
        self.patient_frames = [frame for frame in self.manifest[self.cleared_patients[self.patient_index]] if frame[IMAGE_TYPE_LABEL] == self.image_type.value]
        self.frame_index = 0



    # def __iter__(self):

    def __next__(self):

        is_last_patient = self.patient_index == len(self.cleared_patients) - 1
        is_last_frame = self.frame_index == len(self.patient_frames) - 1

        while not (is_last_patient and is_last_frame):

            patient_name = self.cleared_patients[self.patient_index]
            patient_record = self.manifest[patient_name]
            patient_type = patient_record[0][TUMOR_TYPE_LABEL]

            # Only supporting pre-classified training samples. No support for unspecified patients. 

            print("{}/{}/{}/{}".format(
                (self.benign_top_level_path if patient_type == TUMOR_BENIGN else self.malignant_top_level_path),
                patient_name,
                ('focus' if self.timestamp is None else "focus_{}".format(self.timestamp)),
                self.patient_frames[self.frame_index][FOCUS_HASH_LABEL]
            ))

            loaded_image = cv2.imread("{}/{}/{}/{}".format(
                (self.benign_top_level_path if patient_type == TUMOR_BENIGN else self.malignant_top_level_path),
                patient_name,
                ('focus' if self.timestamp is None else "focus_{}".format(self.timestamp)),
                self.patient_frames[self.frame_index][FOCUS_HASH_LABEL]
            ),
                (cv2.IMREAD_COLOR if self.image_type.value == IMAGE_TYPE.COLOR.value else cv2.IMREAD_GRAYSCALE))

            loaded_image = np.expand_dims(loaded_image, axis=0)
            frame_label = self.patient_frames[self.frame_index][TUMOR_TYPE_LABEL]
            frame_label = 0 if frame_label == TUMOR_BENIGN else 1

            if not is_last_frame:
                self.frame_index += 1
            else:
                self.patient_index += 1
                self.patient_frames = [frame for frame in self.manifest[self.cleared_patients[self.patient_index]] if frame[IMAGE_TYPE_LABEL] == self.image_type.value]
                self.frame_index = 0

            is_last_patient = self.patient_index == len(self.cleared_patients) - 1
            is_last_frame = self.frame_index == len(self.patient_frames) - 1
            
            if len(loaded_image.shape) < 4 or loaded_image.shape[1] < 197 or loaded_image.shape[2] < 197:
                print("Skipping")
                print(loaded_image.shape)
                continue

            # print("{}/{}/{}/{}".format(
            #     (self.benign_top_level_path if patient_type == TUMOR_BENIGN else self.malignant_top_level_path),
            #     patient_name,
            #     ('focus' if self.timestamp is None else "focus_{}".format(self.timestamp)),
            #     self.patient_frames[self.frame_index][FOCUS_HASH_LABEL]
            # ))

            # print((loaded_image, frame_label) )
            print("Shape: {} | Label: {}".format(loaded_image.shape, frame_label))
            yield (loaded_image, np.array([frame_label]))

        return