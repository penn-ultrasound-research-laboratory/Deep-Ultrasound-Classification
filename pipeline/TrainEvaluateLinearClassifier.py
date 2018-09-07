import traceback
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from constants.exceptions.customExceptions import TrainEvaluateLinearClassifierException

tf.logging.set_verbosity(tf.logging.INFO)

def get_input_fn(feature_ndarray, labels, num_epochs=None, shuffle=True, feature_basename="res"):
    """Returns a tensorflow numpy input function

    Arguments:
        feature_ndarray                      feature matrix
        labels                               labels array

    Optional:
        num_epochs                           number of epochs to run over the data
        shuffle                              Boolean indicating whether to shuffle the queue
        feature_basename                     String to use as basename for generated features (e.g. res_1,...,res_i)

    Returns:
        A tensorflow input function that feeds dict of numpy arrays into model
    """
    number_features = feature_ndarray.shape[1] # each row is a sample
    return tf.estimator.inputs.numpy_input_fn(
        x={ "{}_{}".format(feature_basename, n): feature_ndarray[:, n] for n in range(number_features)},
        y=labels,
        num_epochs=num_epochs,
        shuffle=shuffle)



def train_evaluate_linear_classifier(abs_path_to_np_data):
    """Trains and evaluates a classifier on a passed in set of features and labels

    Numpy data file structure:

    {
        training_features:                   np.array[samples, features]
        training_labels:                     [1, samples]
        validation_features:                 [samples, features]
        validation_labels:                   [1, samples]
        test_features:                       [samples, features]
        test_labels:                         [1, samples]
    }   

    Arguments:
        abs_path_to_np_data                  absolute path to numpy data file

    Returns:
        Nothing
        TODO: dump evaluation summary to file? ROC plots, precision/recall, confusion matrix

    """
    with open(abs_path_to_np_data, "rb") as f:
        data = np.load(f)
        data = data[()]
    
    # TODO: Missing feature scaling

    # TODO: Convert this to a property of the bundled dataset
    number_features = data["training_features"].shape[1] # each row is a sample
    feature_names = ["res_{}".format(n) for n in range(number_features)]
    feature_columns = [tf.feature_column.numeric_column(k) for k in tqdm(feature_names, desc="NP->TF")]

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
        train_steps=1000)

    experiment.train_and_evaluate()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "abs_path_to_np_data",
        type=str,
        help="Absolute path to numpy data file. Data format is dictionary")

    args = vars(parser.parse_args())

    try:
        train_evaluate_linear_classifier(
            args["abs_path_to_np_data"])
    
    except TrainEvaluateLinearClassifierException as e:
        traceback.print_exc()
