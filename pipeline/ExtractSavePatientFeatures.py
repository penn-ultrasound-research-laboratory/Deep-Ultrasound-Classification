import argparse
import json
import logging
import uuid

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from utilities.patientsPartition import patient_train_test_validation_split
from pipeline.PatientSampleGenerator import PatientSampleGenerator

from constants.exceptions.customExceptions import ExtractSavePatientFeatureException

from constants.ultrasoundConstants import (
    IMAGE_TYPE,
    NUMBER_CHANNELS_COLOR,
    NUMBER_CHANNELS_GRAYSCALE)

from constants.modelConstants import (
    DEFAULT_BATCH_SIZE,
    RESNET50_REQUIRED_NUMBER_CHANNELS,
    SAMPLE_WIDTH,
    SAMPLE_HEIGHT,
    INCEPTION_RESNET_V2_WIDTH,
    INCEPTION_RESNET_V2_HEIGHT,
    RESNET_50_HEIGHT,
    RESNET_50_WIDTH)
    
from keras.models import Model
from keras.applications import inception_resnet_v2
from keras.applications import resnet50
from keras.preprocessing.image import ImageDataGenerator

logger = logging.getLogger('research')

def extract_save_patient_features(
    benign_top_level_path, 
    malignant_top_level_path, 
    manifest_path, 
    output_directory_path,
    batch_size=DEFAULT_BATCH_SIZE,
    image_data_generator=None,
    image_type=IMAGE_TYPE.ALL,
    target_shape=[RESNET_50_HEIGHT, RESNET_50_WIDTH],
    timestamp=None):
    """Builds and saves a Numpy dataset with Resnet extracted features

    Arguments:
        benign_top_level_path: absolute path to benign directory
        malignant_top_level_path: absolute path to malignant directory
        manifest_path: absolute path to JSON containing all information from image OCR, tumor types, etc
        output_directory_path: absolute path to output directory
        batch_size: (optional) number of samples to take from each input frame (default 16)
        image_data_generator: (optional) preprocessing generator to run on input images
        image_type: (optional) type of image frames to process (IMAGE_TYPE Enum). i.e. grayscale or color
        target_shape: (optional) array containing target shape to use for output samples [rows, columns]. 
            No channel dimension.
        timestamp: (optional) optional timestamp string to append in focus directory path. 
            i.e. ***/focus_{timestamp}/***

    Returns:
        Integer status code. Zero indicates clean processing and write to file. 
        Non-zero indicates error in feature generation script. 

    Raises:
        PatientSampleGeneratorException for any error generating sample batches
    """

    try:
        
        # When include_top=True input shape must be 224,224
        base_model = resnet50.ResNet50(
            include_top=True,
            classes=2,
            weights=None)

        # base_model = inception_resnet_v2.InceptionResNetV2(
        #     include_top=True,
        #     classes=2,
        #     weights=None)

        # Pre-softmax layer may be way too late
        model = Model(
            input=base_model.input,
            output=base_model.get_layer('avg_pool').output)

        # Load the patient manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f) 

        # Generate a random patient partition. Split the partition the data into train and test partitions.
        # Shuffle the patients as best practice to prevent even out training over classes

        partition = patient_train_test_validation_split(
            benign_top_level_path,
            malignant_top_level_path,
            include_validation = False)

        image_data_generator = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            horizontal_flip=True,
            vertical_flip=True)

        training_partition = partition["benign_train"] + partition["malignant_train"]
        test_partition = partition["benign_test"] + partition["malignant_test"]

        np.random.shuffle(training_partition)
        np.random.shuffle(test_partition)

        logging.info("Training Partition: {}".format(len(training_partition)))
        logging.info("Test Partition: {}".format(len(test_partition)))

        # Instantiate generators over training/test/(validation) partitions

        training_sample_generator = PatientSampleGenerator(
            training_partition,
            benign_top_level_path,
            malignant_top_level_path,
            manifest,
            target_shape = target_shape,
            batch_size = batch_size,
            image_type = image_type,
            image_data_generator = image_data_generator,
            timestamp = timestamp,
            kill_on_last_patient = True,
            auto_resize_to_manifest_scale_max=False)

        test_sample_generator = PatientSampleGenerator(
            test_partition,
            benign_top_level_path,
            malignant_top_level_path,
            manifest,
            target_shape = target_shape,
            batch_size = batch_size,
            image_type = image_type,
            image_data_generator = image_data_generator,
            timestamp = timestamp,
            kill_on_last_patient = True,
            auto_resize_to_manifest_scale_max=False)

        # Count the number of training and test samples - unknown at runtime due to randomized partition and
        # OCR/segmentation errors in preprocessing

        training_count = 0
        test_count = 0
        try:
            gen = next(training_sample_generator)
            while True:
                if training_count == 0:
                    current_batch = next(gen)
                    current_features = model.predict(current_batch[0])
                    output_feature_shape = np.squeeze(current_features).shape

                    print("Features shape determined to be: {}".format(current_features.shape))
                    print("Compressed feature shape: {}".format(np.squeeze(current_features).shape))
                    print("Sample classes shape: {}".format(current_batch[1].shape))
                else:
                    next(gen)
                
                training_count += 1
        except:
            logger.info("Training count number samples: {}".format(training_count))

        try:
            gen = next(test_sample_generator)
            while True:
                next(gen)
                test_count += 1
        except:
            logger.info("Test count number samples: {}".format(test_count))

        ## Preallocate output feature matrix

        X_training = np.empty((batch_size * training_count, output_feature_shape[1]))
        X_test = np.empty((batch_size * test_count, output_feature_shape[1]))
        print("Final training features output shape: {}".format(X_training.shape))
        print("Final test features output shape: {}".format(X_test.shape))
    
        y_training = np.empty(training_count * batch_size)
        y_test = np.empty(test_count * batch_size)
        print("Final training classes output shape: {}".format(y_training.shape))
        print("Final test classes output shape: {}".format(y_test.shape))

        # Insantiate fresh generators

        training_sample_generator = PatientSampleGenerator(
            training_partition,
            benign_top_level_path,
            malignant_top_level_path,
            manifest,
            target_shape = target_shape,
            batch_size = batch_size,
            image_type = image_type,
            image_data_generator = image_data_generator,
            timestamp = timestamp,
            kill_on_last_patient = True,
            auto_resize_to_manifest_scale_max=False)

        test_sample_generator = PatientSampleGenerator(
            test_partition,
            benign_top_level_path,
            malignant_top_level_path,
            manifest,
            target_shape = target_shape,
            batch_size = batch_size,
            image_type = image_type,
            image_data_generator = image_data_generator,
            timestamp = timestamp,
            kill_on_last_patient = True,
            auto_resize_to_manifest_scale_max=False)
        
        training_gen = next(training_sample_generator)
        test_gen = next(test_sample_generator)

        # Extract training features

        for bx in tqdm(range(training_count), desc="Training"):
            current_batch = next(training_gen)
            current_features = np.squeeze(model.predict(current_batch[0]))
            current_classes = current_batch[1]

            X_training[batch_size*bx:batch_size*(bx+1), :] = current_features
            y_training[batch_size*bx:batch_size*(bx+1)] = current_classes

        # Extract test features
        
        for bx in tqdm(range(test_count), desc="Test"):
            current_batch = next(test_gen)
            current_features = np.squeeze(model.predict(current_batch[0]))
            current_classes = current_batch[1]

            X_test[batch_size*bx:batch_size*(bx+1), :] = current_features
            y_test[batch_size*bx:batch_size*(bx+1)] = current_classes

        output_hash = uuid.uuid4()

        # TODO: Should probably add a hash to the output of this function
        with open("{}/features_{}_{}.npy".format(output_directory_path, timestamp, output_hash), "wb") as f:
            data = {
                "test_features": X_test, 
                "test_labels": y_test,
                "training_features": X_training,
                "training_labels": y_training,
                "training_partition": training_partition,
                "test_partition": test_partition
            }
            np.save(f, data)
            logging.info("Saved generated features to output directory. Output hash: {}".format(output_hash))

        return output_hash

    except Exception as e:
        logging.critical(e)
        return 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('benign_top_level_path',
                        help = 'absolute path to top level directory containing benign patient folders')

    parser.add_argument('malignant_top_level_path',
                        help = 'absolute path to top level directory containing malignant patient folders')

    parser.add_argument('manifest_path',
                        help = "absolute path to complete manifest file (merged benign and malignant manifests")

    parser.add_argument("output_directory_path",
                        help = "absolute path to complete output directory for generated features and labels")

    parser.add_argument("-bs",
                        "--batch_size",
                        type = int,
                        default = DEFAULT_BATCH_SIZE,
                        help = "Number of samples to extract from each input frame")

    parser.add_argument("-idg",
                        "--image_data_generator",
                        type = ImageDataGenerator,
                        default = None,
                        help = "Keras ImageDataGenerator to preprocess sample image data")

    parser.add_argument("-it",
                        "--image_type",
                        type = IMAGE_TYPE,
                        default = IMAGE_TYPE.ALL,
                        help = "Image class to consider in manifest")

    parser.add_argument("-ts",
                        "--target_shape",
                        type = list,
                        default = [RESNET_50_HEIGHT, RESNET_50_WIDTH],
                        help = "Size to use for image samples of taken frames. Used to pad frames that are smaller the target shape, and crop-sample images that are larger than the target shape")

    parser.add_argument('-T', '--timestamp',
                        type = str,
                        default = None,
                        help = "String timestamp to use as prefix to focus directory and manifest directory")

    args = parser.parse_args()

    try:
    
        logging.basicConfig(level = logging.INFO, filename = "{}/{}_{}.log".format(
            args.output_directory_path,
            args.timestamp,
            uuid.uuid4()
        ))

        exit_code = extract_save_patient_features(
            args.benign_top_level_path,
            args.malignant_top_level_path,
            args.manifest_path,
            args.output_directory_path,
            batch_size = args.batch_size,
            image_data_generator = args.image_data_generator,
            image_type = args.image_type,
            target_shape = args.target_shape,
            timestamp = args.timestamp)


    except Exception as e:
        raise(e)