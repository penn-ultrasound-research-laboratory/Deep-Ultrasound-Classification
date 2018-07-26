import argparse, json
import numpy as np
import tensorflow as tf
from models.resNet50 import ResNet50
from models.patientsPartition import patient_train_test_validation_split
from models.PatientSampleGenerator import PatientSampleGenerator
from keras.preprocessing.image import ImageDataGenerator
from constants.ultrasoundConstants import IMAGE_TYPE
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Model

def extract_save_patient_features():
    """Builds and saves a Numpy dataset
    
    Arguments:
        benign_top_level_path: absolute path to benign directory
        malignant_top_level_path: absolute path to malignant directory
        manifest_path: absolute path to JSON containing all information from image OCR, tumor types, etc
        output_directory_path: absolute path to output directory
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

    # Constants

    NUMBER_SAMPLES_PER_BATCH = 16

    # Image Augmentation function

    image_data_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        zca_whitening=True,
        horizontal_flip=True)

    # Partition the data into train, test, validate

    training_partition = partition["benign_train"] + partition["malignant_train"]
    validation_partition = partition["benign_cval"] + partition["malignant_cval"]
    test_partition = partition["benign_test"] + partition["malignant_test"]

    np.random.shuffle(training_partition)
    np.random.shuffle(validation_partition)
    np.random.shuffle(test_partition)

    print("Training Partition: {}".format(len(training_partition)))
    print("Validation Partition: {}".format(len(validation_partition)))
    print("Test Partition: {}".format(len(test_partition)))

    training_sample_generator = PatientSampleGenerator(
        training_partition,
        arguments["benign_top_level_path"],
        arguments["malignant_top_level_path"],
        manifest,
        target_shape=np.array([197, 197]),
        number_channels=3,
        batch_size=NUMBER_SAMPLES_PER_BATCH,
        image_type=IMAGE_TYPE.COLOR, 
        timestamp=arguments["timestamp"],
        kill_on_last_patient=True)

    validation_sample_generator = PatientSampleGenerator(
        validation_partition,
        arguments["benign_top_level_path"],
        arguments["malignant_top_level_path"],
        manifest,
        target_shape=np.array([197, 197]),
        number_channels=3,
        batch_size=NUMBER_SAMPLES_PER_BATCH,
        image_type=IMAGE_TYPE.COLOR, 
        timestamp=arguments["timestamp"],
        kill_on_last_patient=True)

    test_sample_generator = PatientSampleGenerator(
        test_partition,
        arguments["benign_top_level_path"],
        arguments["malignant_top_level_path"],
        manifest,
        target_shape=np.array([197, 197]),
        number_channels=3,
        batch_size=1,
        image_type=IMAGE_TYPE.COLOR, 
        timestamp=arguments["timestamp"],
        kill_on_last_patient=True)

    base_model = ResNet50(
        include_top=False,
        input_shape=(197, 197, 3),
        weights='imagenet',
        pooling='avg')

    model = Model(
        input=base_model.input,
        output=base_model.get_layer('global_average_pooling2d_1').output)

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
                y_training,
                current_batch[1]),
                axis=0)

    except Exception as e:
        print(e)


    print("Training Shape: {} | {}".format(X_training.shape, y_training.shape))
    print("Test Shape: {} | {}".format(X_test.shape, y_test.shape))
    print("Validation Shape: {} | {}".format(X_validation.shape, y_validation.shape))


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
    # estimator = tf.estimator.LinearClassifier(
    #     feature_columns=feature_columns,
    #     n_classes=2)

    # estimator.train(
    #     input_fn=next(training_sample_generator))

    # estimator.evaluate(
    #     input_fn=next(validation_sample_generator))

    # estimator.predict(
    #     input_fn=next(test_sample_generator))

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
