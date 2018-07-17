from constants.ultrasoundConstants import *
from keras.preprocessing.image import ImageDataGenerator
from constants.exceptions.customExceptions import PatientSampleGeneratorException
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
        self.reset_to_initial_patient()


    # def __iter__(self):


    def reset_to_initial_patient(self):
        self.patient_index = 0
        self.patient_frames = [frame[IMAGE_TYPE_LABEL] == self.image_type.value
                               for frame in self.manifest[self.cleared_patients[self.patient_index]]]
        self.frame_index = 0


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
                patient_record[self.frame_index][FOCUS_HASH_LABEL]
            ))

            loaded_image = cv2.imread("{}/{}/{}/{}".format(
                (self.benign_top_level_path if patient_type == TUMOR_BENIGN else self.malignant_top_level_path),
                patient_name,
                ('focus' if self.timestamp is None else "focus_{}".format(self.timestamp)),
                patient_record[self.frame_index][FOCUS_HASH_LABEL]
            ),
                (cv2.IMREAD_COLOR if self.image_type.value == IMAGE_TYPE.COLOR.value else cv2.IMREAD_GRAYSCALE))

            print(loaded_image.shape)

            frame_label = patient_record[self.frame_index][TUMOR_TYPE_LABEL]

            if not is_last_frame:
                self.frame_index += 1
            else:
                self.patient_index += 1
                self.patient_frames = [frame[IMAGE_TYPE_LABEL] == self.image_type.value
                                    for frame in self.manifest[self.cleared_patients[self.patient_index]]]
                self.frame_index = 0

            yield (loaded_image, frame_label) 

        return