import argparse

from src.pipeline.train.train import train_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-I",
        "--training-data-images",
        help="Path to training data images",
        required=True
    )

    parser.add_argument(
        "-M",
        "--training-data-manifest",
        help="Path to training data manifest",
        required=True
    )

    parser.add_argument(
        "-C"
        "--training-config",
        help="Path to training configuration yaml. i.e. experiment definition in code",
        default=None
    )

    parser.add_argument(
        "-j",
        "--job-dir",
        help="Google Cloud Service location to write checkpoints and export models",
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    # job_dir = arguments.pop("job_dir")

    train_model(**arguments)
