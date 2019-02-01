import tensorflow as tf
from tensorflow.python.lib.io import file_io

def train_model(
    train_file='sentiment_set.pickle',
    job_dir='./tmp/example-5', **args):

    # Load the configuration file yaml file if provided.
    # Else load the default experiment configuration file
    # By default, fall back to default experiment configuration file. 
    # Log all cases

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