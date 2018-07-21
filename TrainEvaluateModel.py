import argparse, json
import numpy as np
from models.resNet50 import ResNet50
from models.patientsPartition import patient_train_test_validation_split
from models.PatientSampleGenerator import PatientSampleGenerator
from constants.ultrasoundConstants import IMAGE_TYPE
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

    partition = patient_train_test_validation_split(
        arguments["benign_top_level_path"],
        arguments["malignant_top_level_path"])

    training_partition = partition["benign_train"] + partition["malignant_train"]
    validation_partition = partition["benign_cval"] + partition["malignant_cval"]

    np.random.shuffle(training_partition)
    np.random.shuffle(validation_partition)

    training_sample_generator = PatientSampleGenerator(
        training_partition,
        arguments["benign_top_level_path"],
        arguments["malignant_top_level_path"],
        manifest,
        batch_size=16,
        image_type=IMAGE_TYPE.COLOR, 
        timestamp=arguments["timestamp"])

    validation_sample_generator = PatientSampleGenerator(
        validation_partition,
        arguments["benign_top_level_path"],
        arguments["malignant_top_level_path"],
        manifest,
        batch_size=16,
        image_type=IMAGE_TYPE.COLOR, 
        timestamp=arguments["timestamp"])


    # print(next(next(training_sample_generator))[0].shape)

    # classes = 2

    # Instantiate the resNet50 model
    # model = ResNet50(
    #     include_top=False,
    #     input_shape=(200, 200, 3),
    #     weights=None,
    #     pooling='max'
    # )

    model = ResNet50(
        include_top=True,
        classes=2,
        weights=None
    )


    # With a categorical crossentropy loss function, the network outputs must be categorical 
    model.compile(loss=categorical_crossentropy,
                optimizer=Adam(),
                metrics=['accuracy'])


    model.fit_generator(
        next(training_sample_generator), 
        steps_per_epoch=1, 
        validation_data=next(validation_sample_generator),
        validation_steps=1,
        epochs=100, 
        verbose=2,
        use_multiprocessing=True)

    # gen = next(patient_sample_generator)
    # print(next(gen)[0].shape)µ