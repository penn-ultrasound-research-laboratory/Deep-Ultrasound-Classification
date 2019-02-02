import tensorflow as tf
import json
import yaml

from tensorflow.python.lib.io import file_io
from tensorflow.python.framework.errors_impl import NotFoundError
from src.utilities.partition.patient_partition import *
from src.utilities.general.general import default_none

DEFAULT_CONFIG = "src/config/default.yaml"

def train_model(args):

    # Load the configuration file yaml file, if provided.
    config_file = default_none(args.config, DEFAULT_CONFIG)
    try:
        with file_io.FileIO(config_file, mode='r') as stream:
            exp_config = yaml.load(stream)
    except NotFoundError as _:
        print("Configuration file not found: {0}".format(config_file))
        return
    except Exception as _:
        print("Unable to load configuration file: {0}".format(config_file))
        return

    print(exp_config)

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

    # logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    # print('-----------------------')
    # print('Using train_file located at {}'.format(train_file))
    # print('Using logs_path located at {}'.format(logs_path))
    # print('-----------------------')
    # file_stream = 
    # x_train, y_train, x_test, y_test  = pickle.load(file_stream)
    


    print(exp_config)



    # Load the training manifest
    # Fail on error

    # Partition the training image set based on the configuration
    # Update patient partition utility to use random seed
    # Configuration file will store random seed --> patient partition

    # Train the provided model according to the configuration
    # Fail on model load failure

    # Evaluate the model

    # Save the model

    return