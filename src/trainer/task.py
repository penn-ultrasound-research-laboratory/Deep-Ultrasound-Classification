
import argparse
import json
import yaml
import os

from dotmap import DotMap
from datetime import datetime
from importlib import import_module


from tensorflow.python.lib.io import file_io
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from constants.ultrasound import string_to_image_type, TUMOR_TYPES
from pipeline.patientsample.patient_sample_generator import PatientSampleGenerator
from utilities.partition.patient_partition import patient_train_test_split
from utilities.general.general import default_none
from utilities.manifest.manifest import patient_type_lists, patient_lists_to_dataframe

import tensorflow.keras.preprocessing
print(tensorflow.keras.__version__)

DEFAULT_CONFIG = "../config/default.yaml"

def train_model(args):

    # Establish logging
    job_dir = default_none(args.job_dir, ".")
    logs_path = "{0}/logs/{1}".format(job_dir, datetime.now().isoformat())

    # Load the configuration file yaml file if provided
    config_file = default_none(args.config, DEFAULT_CONFIG)
    try:
        with file_io.FileIO(config_file, mode='r') as stream:
            config = DotMap(yaml.load(stream))
    except NotFoundError as _:
        print("Configuration file not found: {0}".format(config_file))
        return
    except Exception as _:
        print("Unable to load configuration file: {0}".format(config_file))
        return


    # Load the manifest file
    try:
        with file_io.FileIO(args.manifest, mode='r') as stream:
            manifest = json.load(stream)
    except NotFoundError as _:
        print("Manifest file not found: {0}".format(args.manifest))
        return
    except Exception as _:
        print("Unable to load manifest file: {0}".format(args.manifest))
        return
    

    benign_patients, malignant_patients = patient_type_lists(manifest)

    # Train/test split according to config
    patient_split = DotMap(patient_train_test_split(
        benign_patients,
        malignant_patients,
        config.train_split,
        config.random_seed
    ))

    # Crawl the manifest to assemble dataframe of matching patient frames
    train_df = patient_lists_to_dataframe(
        patient_split.train,
        manifest,
        string_to_image_type(config.image_type),
        "/Benign",
        "/Malignant")

    image_data_generator = ImageDataGenerator(**config.image_preprocessing.toDict())

    train_generator = image_data_generator.flow_from_dataframe(
        dataframe = train_df,
        directory = args.images,
        x_col = "filename",
        y_col = "class",
        target_size = config.target_shape,
        color_mode = "rgb",
        class_mode = "categorical",
        classes = TUMOR_TYPES,
        batch_size = config.batch_size,
        shuffle = True,
        seed = config.random_seed
    )

    # Load the model specified in config
    model = import_module("models.{0}".format(config.model)).get_model(config)

    model.summary()

    model.compile(
        Adam(), # default Adam parameters for now
        loss=config.loss,
        metrics=['accuracy'])


    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_df),
        epochs = 2, # Just for testing purposes
        verbose = 2,
        use_multiprocessing = False,
        workers = args.num_workers
    )


    # Evaluate the model

    # Save the model
    # model.save(config.identifier)
    
    # # Save the model on GC storage
    # with file_io.FileIO(config.identifier, mode='r') as input_f:
    #     with file_io.FileIO(args.job_dir + "/{0}".format(config.identifier), mode='w+') as output_f:
    #         output_f.write(input_f.read())

    return


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
        help="Experiment config yaml. i.e. experiment definition in code. Located in /src/config",
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

    parser.add_argument('--num-workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--disp-step', type=int, default=200, help='display step during training')
    parser.add_argument('--cuda', type=bool, default=True, help='enable CUDA')
    
    args=parser.parse_args()
    arguments= DotMap(args.__dict__)

    # Execute the model
    train_model(arguments)