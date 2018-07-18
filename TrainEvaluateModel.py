import argparse, json
from models.resNet50 import ResNet50
from models.patientsPartition import patient_train_test_validation_split
from models.PatientSampleGenerator import PatientSampleGenerator
from constants.ultrasoundConstants import *
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('benign_top_level_path',
        help='absolute path to top level directory containing benign patient folders')
    
    parser.add_argument('malignant_top_level_path',
        help='absolute path to top level directory containing malignant patient folders')
    
    parser.add_argument('manifest_path',
        help="absolute path to complete manifest file (merged benign and malignant manifests")

    parser.add_argument('-T', '--timestamp', type=str,
        default=None,
        help="String timestamp to use as prefix to focus directory and manifest directory")

    arguments = vars(parser.parse_args())

    # Load the manifest
    with open(arguments["manifest_path"], 'r') as f:
        manifest = json.load(f) 

    # classes = 2

    # # Instantiate the resNet50 model
    # model = ResNet50(
    #     include_top=False,
    #     input_shape=(244, 244, 3),
    #     weights=None,
    #     pooling='max'
    # )

    # model = ResNet50(
    #     include_top=True,
    #     classes=2,
    #     weights=None
    # )

    patient_partition = patient_train_test_validation_split(
        arguments["benign_top_level_path"],
        arguments["malignant_top_level_path"])

    patient_sample_generator = PatientSampleGenerator(
        patient_partition["benign_train"] + patient_partition["malignant_train"],
        arguments["benign_top_level_path"],
        arguments["malignant_top_level_path"],
        manifest,
        IMAGE_TYPE.COLOR,
        timestamp=arguments["timestamp"])
        
    print(patient_sample_generator)