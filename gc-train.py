import argparse
import datetime
import subprocess

# !!!!!!!!! USER SHOULD SET THIS !!!!!!!!!!!
TRAINER_PACKAGE_PATH="/Users/Matthew/Documents/Research/UltrasoundResearch/src"
MAIN_TRAINER_MODULE="trainer.task"
STORAGE_BUCKET="research-storage"

# Google Cloud ML-Engine configs
REGION="us-west1"
SCALE_TIER="BASIC_GPU"
RUNTIME_VERSION=1.12
PYTHON_VERSION=3.5

def google_cloud_train(args):

    DATASET=args["images"]
    MANIFEST=args["manifest"]
    CONFIG_FILE=args["config"]
    
    if args["identifier"]:
        JOB_BASE_NAME=args["identifier"]
    else:
        JOB_BASE_NAME="train"

    NOW=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    JOB_NAME="{0}_{1}".format(JOB_BASE_NAME, NOW)
    JOB_DIR="gs://{0}/staging/{1}".format(STORAGE_BUCKET, JOB_NAME)

    # Paths to images
    IMAGES_PATH="gs://{0}/{1}".format(STORAGE_BUCKET, DATASET)
    MANIFEST_PATH="gs://{0}/{1}".format(STORAGE_BUCKET, MANIFEST)

    # Number workers - will eventually be based on scale-tier
    NUM_WORKERS=2

    # Production run
    command = ["gcloud", "ml-engine", "jobs", "submit", "training", JOB_NAME,
            "--module-name", MAIN_TRAINER_MODULE,
            "--package-path", TRAINER_PACKAGE_PATH,
            "--job-dir", JOB_DIR,
            "--scale-tier", SCALE_TIER,
            "--region", REGION,
            "--runtime-version", RUNTIME_VERSION,
            "--python-version", PYTHON_VERSION,
            "--",
            "--images", IMAGES_PATH,
            "--manifest", MANIFEST_PATH,
            "--config", CONFIG_FILE,
            "--num-workers", NUM_WORKERS,
            "--identifier", JOB_BASE_NAME]

    command = [str(s) for s in command]

    subprocess.run(command)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-I",
        "--images",
        help="Name of Images dataset in Google Cloud bucket top-level",
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
        "-i",
        "--identifier",
        help="Base name to identify job in Google Cloud Storage & ML Engine",
        default=None
    )

    parser.add_argument('--num-workers', type=int, default=1, help='number of data loading workers')
    
    args=parser.parse_args().__dict__

    google_cloud_train(args)