import argparse

from dotmap import DotMap
from src.trainer.task import train_model

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
        help="Google Cloud Service location to write checkpoints and export models",
        required=True
    )

    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--disp_step', type=int, default=200, help='display step during training')
    parser.add_argument('--cuda', type=bool, default=True, help='enable CUDA')
    
    args=parser.parse_args()
    arguments= DotMap(args.__dict__)

    # Execute the model
    train_model(arguments)
