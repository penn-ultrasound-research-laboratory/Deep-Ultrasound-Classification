from constants.ultrasoundConstants import IMAGE_TYPE, TUMOR_TYPES
from keras.preprocessing.image import ImageDataGenerator
from constants.exceptions.customExceptions import PatientSampleGeneratorException
import cv2

class PatientSampleGenerator:
    '''Generator that returns batches of samples for training and evaluation
    
    Arguments:
        patient_list:
        top_level_path:
        manifest:
        image_type:

    Returns:

    '''

    def __init__(self, patient_list, top_level_path, manifest, image_type):
        self.image_type = image_type
        self.raw_patient_list = patient_list
        self.top_level_path = top_level_path
        self.manifest = manifest
        self.top_level_path = top_level_path
        # self.grayscaleDataGenerator = ImageDataGenerator()
        # self.colorDataGenerator = ImageDataGenerator()

    def reset_to_initial_patient(self):
        self.patient_index = 0
        self.patient_frames = [frame["IMAGE_TYPE"] == self.image_type.value
                               for frame in self.manifest[self.cleared_patients[0]]]
        self.frame_index = 0

    def __iter__(self):
        # Find all the patients with at least one frame in the specified IMAGE_TYPE

        cleared_patients = [patient for patient in self.raw_patient_list if
                            any([frame["IMAGE_TYPE"] == self.image_type.value
                                 for frame in self.manifest[patient]])]

        if len(cleared_patients) == 0:
            raise PatientSampleGeneratorException('No patients found with focus in image type: {0}'.format(self.image_type.value))
        
        self.cleared_patients = cleared_patients
        self.reset_to_initial_patient()

        return self

    def __next__(self):
        is_last_patient = self.patient_index == len(self.cleared_patients) - 1
        is_last_frame = self.frame_index == len(self.patient_frames) - 1
    
        if is_last_patient and is_last_frame:
            self.reset_to_initial_patient()

        patient = self.manifest[self.cleared_patients[self.patient_index]]
        patient_type = patient[0]['TUMOR_TYPE']

        loaded_image = cv2.imread("{0}/{1}/{2}/{3}/{4}".format(
            self.top_level_path,
            ("Benign" if patient_type == "BENIGN" else "Malignant"),
            patient,
            'focus',
            patient[self.frame_index]['FOCUS']
        ),
            (cv2.IMREAD_COLOR if self.image_type.value == IMAGE_TYPE.COLOR.value else cv2.IMREAD_GRAYSCALE))

        THIS_IS_THE_RETURNED_VALUE = 12347981273489126512

        if not is_last_frame:
            self.frame_index += 1
        else:
            self.patient_index += 1
            self.patient_frames = [frame["IMAGE_TYPE"] == self.image_type.value
                                   for frame in self.manifest[self.cleared_patients[self.patient_index]]]
            self.frame_index = 0

        return THIS_IS_THE_RETURNED_VALUE
