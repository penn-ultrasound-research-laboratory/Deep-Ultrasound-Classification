import traceback
import argparse
import numpy as np
import tensorflow as tf
from constants.exceptions.customExceptions import TrainEvaluateLinearClassifierException

tf.logging.set_verbosity(tf.logging.INFO)

def get_input_fn(feature_ndarray, labels, num_epochs=None, shuffle=True):
    number_features = feature_ndarray.shape[1] # each row is a sample
    return tf.estimator.inputs.numpy_input_fn(
        x={ "res_{}".format(n):feature_ndarray[:, n] for n in range(number_features)},
        y=labels,
        num_epochs=num_epochs,
        shuffle=shuffle)

def train_evaluate_linear_classifier(path_to_numpy_data_file):

    with open(path_to_numpy_data_file, "rb") as f:
        data = np.load(f)
        data = data[()]
           
    # TODO: Convert this to a property of the bundled dataset
    number_features = data["training_features"].shape[1] # each row is a sample
    feature_names = ["res_{}".format(n) for n in range(number_features)]
    feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

    estimator = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=2)

    experiment = tf.contrib.learn.Experiment(
        estimator,
        get_input_fn(
            data["training_features"], 
            data["training_labels"],
            shuffle=True),
        get_input_fn(
            data["test_features"], 
            data["test_labels"],
            shuffle=False,
            num_epochs=1),
        train_steps=2000)

    experiment.train_and_evaluate()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    """
    Expected numpy data input format:
    {
        training_features: np.array[samples, features]
        training_labels: [1, samples]
        validation_features: [samples, features]
        validation_labels: [1, samples]
        test_features: [samples, features]
        test_labels: [1, samples]
    }
    """

    parser.add_argument(
        "path_to_numpy_data_file",
        type=str,
        help="Absolute path to numpy data file. Data format is dictionary")

    args = vars(parser.parse_args())

    try:
        train_evaluate_linear_classifier(
            args["path_to_numpy_data_file"])
    
    except TrainEvaluateLinearClassifierException as e:
        traceback.print_exc()
