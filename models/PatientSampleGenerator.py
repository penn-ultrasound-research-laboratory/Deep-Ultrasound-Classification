from constants.ultrasoundConstants import IMAGE_TYPE
from keras.preprocessing.image import ImageDataGenerator
from constants.exceptions.customExceptions import PatientSampleGeneratorException


class PatientSampleGenerator:
    '''iterator that yields numbers in the Fibonacci sequence
    
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

    def __iter__(self):
        # Find all the patients with at least one frame in the specified IMAGE_TYPE

        cleared_patients = [patient for patient in self.raw_patient_list if
                            any([frame["IMAGE_TYPE"] == self.image_type.value
                                 for frame in self.manifest[patient]])]



        # find_patient = 0
        # while find_patient < self.number_patients:
        #     # if the patient contains any entries where the image type matches, break, else increment
        #     if ):
        #         break
        #     else:
        #         find_patient += 1
        
        if len(cleared_patients) == 0:
            raise PatientSampleGeneratorException('No patients found with focus in image type: {0}'.format(self.image_type.value))
        
        self.cleared_patients = cleared_patients
        self.patient_index = 0
        self.patient_frames = [frame["IMAGE_TYPE"] == self.image_type.value
                               for frame in self.manifest[cleared_patients[0]]]
        self.frame_index = 0

        return self

    def __next__(self):
        last_patient = self.patient_index == len(self.cleared_patients) - 1
        last_frame = self.frame_index == len(self.patient_frames) - 1
    
        if last_patient and last_frame:
            raise StopIteration

        # DETERMINE THE CURRENT VALUES TO RETURN USES ImageGenerator on the current frame input
        #   ....
        #   ....
        # return something????
        THIS_IS_THE_RETURNED_VALUE = 12347981273489126512

        if not last_frame:
            self.frame_index += 1
        else:
            self.patient_index += 1
            self.patient_frames = [frame["IMAGE_TYPE"] == self.image_type.value
                                   for frame in self.manifest[self.cleared_patients[self.patient_index]]]
            self.frame_index = 0

        return THIS_IS_THE_RETURNED_VALUE
