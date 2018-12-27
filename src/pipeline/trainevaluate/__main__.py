import argparse
import traceback

from src.constants.exceptions.customExceptions import TrainEvaluateLinearClassifierException
from src.pipeline.trainevaluate.train_evaluate_linear_classifier import train_evaluate_linear_classifier

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
