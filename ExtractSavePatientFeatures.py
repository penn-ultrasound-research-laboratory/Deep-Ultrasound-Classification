import argparse, json, uuid
import numpy as np
import tensorflow as tf
from constants.exceptions.customExceptions import ExtractSavePatientFeatureException
from models.resNet50 import ResNet50
from utilities.patientsPartition import patient_train_test_validation_split
from models.PatientSampleGenerator import PatientSampleGenerator
from keras.preprocessing.image import ImageDataGenerator
from constants.ultrasoundConstants import (
    IMAGE_TYPE,
    NUMBER_CHANNELS_COLOR,
    NUMBER_CHANNELS_GRAYSCALE)
from constants.modelConstants import (
    DEFAULT_BATCH_SIZE,
    RESNET50_REQUIRED_NUMBER_CHANNELS,
    SAMPLE_WIDTH,
    SAMPLE_HEIGHT)
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Model

def extract_save_patient_features(
    benign_top_level_path, 
    malignant_top_level_path, 
    manifest_path, 
    output_directory_path,
    batch_size=DEFAULT_BATCH_SIZE,
    image_data_generator=None,
    image_type=IMAGE_TYPE.COLOR,
    number_channels=3,
    target_shape=[SAMPLE_HEIGHT, SAMPLE_WIDTH],
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
        number_channels (optional) number of color channels. Should be 3 if color, 1 if grayscale. 
            3 by default to match image_type=COLOR default
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
        
        base_model = ResNet50(
            include_top=False,
            input_shape=tuple(target_shape) + (RESNET50_REQUIRED_NUMBER_CHANNELS,),
            weights='imagenet',
            pooling='avg')

        model = Model(
            input=base_model.input,
            output=base_model.get_layer('global_average_pooling2d_1').output)
        
        # Load the patient manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f) 

        # Generate a random patient partition. Split the partition the data into train, test, validate.
        # Shuffle the patients as best practice to prevent even out training over classes

        partition = patient_train_test_validation_split(
            benign_top_level_path,
            malignant_top_level_path)

        training_partition = partition["benign_train"] + partition["malignant_train"]
        validation_partition = partition["benign_cval"] + partition["malignant_cval"]
        test_partition = partition["benign_test"] + partition["malignant_test"]

        np.random.shuffle(training_partition)
        np.random.shuffle(validation_partition)
        np.random.shuffle(test_partition)

        print("Training Partition: {}".format(len(training_partition)))
        print("Validation Partition: {}".format(len(validation_partition)))
        print("Test Partition: {}".format(len(test_partition)))

        # PatientSampleGenerator over the training patients

        training_sample_generator = PatientSampleGenerator(
            training_partition,
            benign_top_level_path,
            malignant_top_level_path,
            manifest,
            target_shape=target_shape,
            number_channels=number_channels,
            batch_size=batch_size,
            image_type=image_type, 
            timestamp=timestamp,
            kill_on_last_patient=True)

        validation_sample_generator = PatientSampleGenerator(
            validation_partition,
            benign_top_level_path,
            malignant_top_level_path,
            manifest,
            target_shape=target_shape,
            number_channels=number_channels,
            batch_size=batch_size,
            image_type=image_type, 
            timestamp=timestamp,
            kill_on_last_patient=True)

        test_sample_generator = PatientSampleGenerator(
            test_partition,
            benign_top_level_path,
            malignant_top_level_path,
            manifest,
            target_shape=target_shape,
            number_channels=number_channels,
            batch_size=batch_size,
            image_type=image_type, 
            timestamp=timestamp,
            kill_on_last_patient=True)

        X_validation = y_validation = None
        try:
            gen = next(validation_sample_generator)
            while True:
                current_batch = next(gen)
                current_predictions = model.predict(current_batch[0])

                X_validation = current_predictions if X_validation is None else np.concatenate((
                    X_validation, 
                    current_predictions), 
                    axis=0)

                y_validation = current_batch[1] if y_validation is None else np.concatenate((
                    y_validation,
                    current_batch[1]),
                    axis=0)

        except Exception as e:
            print(e)


        X_training = y_training = None
        try:
            gen = next(training_sample_generator)
            while True:
                current_batch = next(gen)
                current_predictions = model.predict(current_batch[0])

                X_training = current_predictions if X_training is None else np.concatenate((
                    X_training, 
                    current_predictions), 
                    axis=0)

                y_training = current_batch[1] if y_training is None else np.concatenate((
                    y_training,
                    current_batch[1]),
                    axis=0)

        except Exception as e:
            print(e)

        X_test = y_test = None
        try:
            gen = next(test_sample_generator)
            while True:
                current_batch = next(gen)
                current_predictions = model.predict(current_batch[0])

                X_test = current_predictions if X_test is None else np.concatenate((
                    X_test, 
                    current_predictions), 
                    axis=0)

                y_test = current_batch[1] if y_test is None else np.concatenate((
                    y_test,
                    current_batch[1]),
                    axis=0)

        except Exception as e:
            print(e)


        output_hash = uuid.uuid4()

        # TODO: Should probably add a hash to the output of this function
        with open("{}/features_{}_{}.npy".format(output_directory_path, timestamp, output_hash), "wb") as f:
            data = {
                "test_features": X_test, 
                "test_labels": y_test,
                "training_features": X_training,
                "training_labels": y_training,
                "validation_features": X_validation,
                "validation_labels": y_validation
            }
            np.save(f, data)
            print("Saved generated features to output directory. Output hash: {}".format(output_hash))

        return output_hash

    except Exception as e:
        print(e)
        return 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('benign_top_level_path',
                        help='absolute path to top level directory containing benign patient folders')

    parser.add_argument('malignant_top_level_path',
                        help='absolute path to top level directory containing malignant patient folders')

    parser.add_argument('manifest_path',
                        help="absolute path to complete manifest file (merged benign and malignant manifests")

    parser.add_argument("output_directory_path",
                        help="absolute path to complete output directory for generated features and labels")

    parser.add_argument("-bs",
                        "--batch_size",
                        type=int,
                        default=DEFAULT_BATCH_SIZE,
                        help="Number of samples to extract from each input frame")

    parser.add_argument("-idg",
                        "--image_data_generator",
                        type=ImageDataGenerator,
                        default=None,
                        help="Keras ImageDataGenerator to preprocess sample image data")

    # TODO: For now only support building dataset on either grayscale or color. Not both at the same time. Should build
    # support into this application
    parser.add_argument("-it",
                        "--image_type",
                        type=IMAGE_TYPE,
                        default=IMAGE_TYPE.COLOR,
                        help="Image class to consider in manifest")

    parser.add_argument("-nc",
                        "--number_channels",
                        type=int,
                        default=NUMBER_CHANNELS_COLOR,
                        help="Number of channels for image type (# color format channels)")

    parser.add_argument("-ts",
                        "--target_shape",
                        type=list,
                        default=[SAMPLE_HEIGHT, SAMPLE_WIDTH],
                        help="Size to use for image samples of taken frames. Used to pad frames that are smaller the target shape, and crop-sample images that are larger than the target shape")

    parser.add_argument('-T', '--timestamp',
                        type=str,
                        default=None,
                        help="String timestamp to use as prefix to focus directory and manifest directory")

    args = vars(parser.parse_args())

    try:
        code = extract_save_patient_features(
            args["benign_top_level_path"],
            args["malignant_top_level_path"],
            args["manifest_path"],
            args["output_directory_path"],
            batch_size=args["batch_size"],
            image_data_generator=args["image_data_generator"],
            image_type=args["image_type"],
            number_channels=args["number_channels"],
            target_shape=args["target_shape"],
            timestamp=args["timestamp"])

    except Exception as e:
        raise(e)

    # Load the manifest


    # Save all feature, labels files to directory here. 









    # SHOULD PROBABLY JUST BE USING THE MODEL TO PRODUCE FEATURES AS FEED-IN TO 
    # LINEAR SVM CONSIDERING THE DATA IS SMALL AND EXTREMELY DIFFERENT FROM TRAINING

    # Instantiate the resNet50 model
    # base_model = ResNet50(
    #     include_top=False,
    #     input_shape=(197, 197, 3),
    #     weights='imagenet',
    #     pooling='avg')

    # model = Model(
    #     input=base_model.input,
    #     output=base_model.get_layer('global_average_pooling2d_1').output)


    # # Output dimensionality is (batch_size, 2048)
    # feature_columns = list(map(lambda x: tf.feature_column.numeric_column(
    #     key="resNet_{}".format(str(x)),
    #     dtype=tf.float64,
    #     shape=NUMBER_SAMPLES_PER_BATCH),
    #     range(2048)))

    # print(feature_columns[:3])

    # # Estimator using the default optimizer.


    # predictions = model.predict_generator(
    #     next(training_sample_generator), 
    #     steps=1, 
    #     max_queue_size=10, 
    #     workers=1, 
    #     use_multiprocessing=False, 
    #     verbose=2)






    ############# OLD Code trying to retrain resNet50 - kind of silly 


    # model.summary()


    # # With a categorical crossentropy loss function, the network outputs must be categorical 
    # model.compile(loss=categorical_crossentropy,
    #             optimizer=Adam(),
    #             metrics=['accuracy'])


    # model.fit_generator(
    #     next(training_sample_generator), 
    #     steps_per_epoch=1, 
    #     validation_data=next(validation_sample_generator),
    #     validation_steps=1,
    #     epochs=5, 
    #     verbose=2,
    #     use_multiprocessing=True)

    # gen = next(patient_sample_generator)
    # print(next(gen)[0].shape)µ
