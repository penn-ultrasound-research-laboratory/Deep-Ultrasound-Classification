
import argparse
import json
import yaml
import os
import pkg_resources

from dotmap import DotMap
from datetime import datetime
from importlib import import_module

import tensorflow as tf
import numpy as np

from tensorflow.python.lib.io import file_io
from tensorflow.python.framework.errors_impl import NotFoundError

from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras_preprocessing.image import ImageDataGenerator

from constants.ultrasound import string_to_image_type, TUMOR_TYPES
from pipeline.patientsample.patient_sample_generator import PatientSampleGenerator
from utilities.partition.patient_partition import patient_train_test_split
from utilities.general.general import default_none
from utilities.manifest.manifest import patient_type_lists, patient_lists_to_dataframe
from utilities.image.image import crop_generator

DEFAULT_CONFIG = "../config/default.yaml"


def train_model(args):

    IN_LOCAL_TRAINING_MODE = not args.job_dir
    JOB_DIR = default_none(args.job_dir, ".")
    LOGS_PATH = "{0}/logs".format(JOB_DIR)
    CONFIG_FILE = default_none(args.config, DEFAULT_CONFIG)
    MODEL_FILE = "{0}.h5".format(args.identifier)
    GC_MODEL_SAVE_PATH = "{0}/model/{1}".format(JOB_DIR, MODEL_FILE)

    with tf.device('/device:GPU:0'):

        # Load the configuration file yaml file if provided
        try:
            print("Loading configuration file from: {0}".format(CONFIG_FILE))
            with file_io.FileIO(CONFIG_FILE, mode='r') as stream:
                config = DotMap(yaml.load(stream))
        except NotFoundError as _:
            print("Configuration file not found: {0}".format(CONFIG_FILE))
            return
        except Exception as _:
            print("Unable to load configuration file: {0}".format(CONFIG_FILE))
            return

        # Load the manifest file
        try:
            print("Loading manifest file from: {0}".format(args.manifest))
            with file_io.FileIO(args.manifest, mode='r') as stream:
                manifest = json.load(stream)
        except NotFoundError as _:
            print("Manifest file not found: {0}".format(args.manifest))
            return
        except Exception as _:
            print("Unable to load manifest file: {0}".format(args.manifest))
            return

        benign_patients, malignant_patients = patient_type_lists(manifest)

        # For local testing of models/configuration, limit to six patients of each type
        if IN_LOCAL_TRAINING_MODE:
            print("Local training test. Limiting to six patients from each class.")
            benign_patients = np.random.choice(
                benign_patients, 6, replace=False).tolist()
            malignant_patients = np.random.choice(
                malignant_patients, 6, replace=False).tolist()

        # Train/test split according to config
        patient_split = DotMap(patient_train_test_split(
            benign_patients,
            malignant_patients,
            config.train_split,
            validation_split=config.validation_split,
            random_seed=config.random_seed
        ))

        tb_callback = TensorBoard(
            log_dir=LOGS_PATH,
            histogram_freq=0,
            batch_size=config.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False)

        # Crawl the manifest to assemble training DataFrame of matching patient frames
        train_df = patient_lists_to_dataframe(
            patient_split.train,
            manifest,
            string_to_image_type(config.image_type),
            args.images + "/Benign",
            args.images + "/Malignant")

        # Print some sample information
        print("Training DataFrame shape: {0}".format(train_df.shape))
        print("Training DataFrame class breakdown")
        print(train_df["class"].value_counts())
       

        print(train_df.iloc[:2])

        train_data_generator = ImageDataGenerator(
            **config.image_preprocessing_train.toDict())
        test_data_generator = ImageDataGenerator(
            **config.image_preprocessing_test.toDict())

        train_generator = train_data_generator.flow_from_dataframe(
            dataframe=train_df,
            directory=None,
            x_col="filename",
            y_col="class",
            target_size=config.target_shape,
            color_mode="rgb",
            class_mode="binary",
            classes=TUMOR_TYPES,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.random_seed,
            drop_duplicates=False
        )

        # Optional: subsample each input to batch of randomly placed crops
        if config.subsample.subsample_shape:
            train_generator = crop_generator(
                train_generator,
                config.subsample.subsample_shape,
                config.subsample.subsample_batch_size)

        # Optional: assemble validation DataFrame and validation generator
        if config.validation_split:
            validation_df = patient_lists_to_dataframe(
                patient_split.validation,
                manifest,
                string_to_image_type(config.image_type),
                args.images + "/Benign",
                args.images + "/Malignant")

            print("Validation DataFrame class breakdown")
            print(validation_df["class"].value_counts())

            validation_generator = test_data_generator.flow_from_dataframe(
                dataframe=validation_df,
                directory=None,
                x_col="filename",
                y_col="class",
                target_size=config.target_shape,
                color_mode="rgb",
                class_mode="binary",
                classes=TUMOR_TYPES,
                batch_size=config.batch_size,
                shuffle=True,
                seed=config.random_seed,
                drop_duplicates=False
            )
        else:
            # Config does not specify validation split
            validation_generator = None

        # Load the model specified in config
        model = import_module("models.{0}".format(
            config.model)).get_model(config)

        # model.summary()

        model.compile(
            # default Adam parameters for now
            optimizer=Adam(lr=config.learning_rate),
            loss=config.loss,
            metrics=['accuracy'])

        model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_df) // config.batch_size,
            epochs=config.training_epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_df) // config.batch_size,
            verbose=2,
            use_multiprocessing=True,
            workers=args.num_workers,
            callbacks=[tb_callback]
        )
        
        # Save the model
        model.save(MODEL_FILE)

        # Save the model on GC storage in cloud mode
        if not IN_LOCAL_TRAINING_MODE:
            with file_io.FileIO(MODEL_FILE, mode="rb") as input_f:
                with file_io.FileIO(GC_MODEL_SAVE_PATH, mode="wb+") as output_f:
                    output_f.write(input_f.read())
            print("Model saved to {0}".format(GC_MODEL_SAVE_PATH))

        print("Training Complete")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-I",
        "--images",
        help="Path to training data images top level directory",
        required=True
    )

    parser.add_argument(
        "-M",
        "--manifest",
        help="Path to training data manifest",
        required=True
    )

    parser.add_argument(
        "-C",
        "--config",
        help="Experiment config yaml. i.e. experiment definition in code. Must be place in /src/config directory.",
        default=None
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=int,
        default=0,
        help="checkpoint (epoch id) that will be loaded. If a negative value is passed, default to zero"
    )

    parser.add_argument(
        "-j",
        "--job-dir",
        help="the directory for logging in GC",
        default=None
    )

    parser.add_argument(
        "-i",
        "--identifier",
        help="Base name to identify job in Google Cloud Storage & ML Engine",
        default=None
    )

    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of data loading workers')
    parser.add_argument('--disp-step', type=int, default=200,
                        help='display step during training')
    parser.add_argument('--cuda', type=bool, default=True, help='enable CUDA')

    args = parser.parse_args()
    arguments = DotMap(args.__dict__)

    # config argument passed-in is a filename. Locate the config file in the config directory
    if arguments.config:
        arguments.config = pkg_resources.resource_filename(
            __name__,
            "{0}/{1}".format("../config", arguments.config))

    # Train the model
    train_model(arguments)
