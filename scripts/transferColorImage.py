import argparse
import json
import os
from shutil import copyfile

# Iterate over the manifest and find everything where the image_type is COLOR
# If the file exists in V2.0_Processed, copy to the corresponding subdirectory of V4.0_Processed

def transfer_images(args):

    with open (args["manifest"]) as f:
        manifest = json.load(f)

    # Searching for grayscale images that are listed in the manifest but the files do not exist
    for k in manifest:
        for frame in manifest[k]:

            src = "{0}/{1}/{2}/{3}".format(args["source"], frame["TUMOR_TYPE"], k, frame["FRAME"])
            dst = "{0}/{1}/{2}/{3}".format(args["destination"], frame["TUMOR_TYPE"], k, frame["FRAME"])

            if frame["IMAGE_TYPE"] == "COLOR" and os.path.isfile(src):
                copyfile(src, dst)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-M",
        "--manifest",
        help="Path to manifest",
        required=True
    )

    parser.add_argument(
        "-S",
        "--source",
        help="Path to source data directory",
        required=True
    )
    
    parser.add_argument(
        "-D",
        "--destination",
        help="Path to destination data directory",
        required=True
    )

    args = parser.parse_args()
    transfer_images(args.__dict__)
