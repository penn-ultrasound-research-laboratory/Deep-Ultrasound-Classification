import argparse
import logging
import uuid

from pipeline.extractsave.extract_save_patient_features import extract_save_patient_features

from constants.model import (
    DEFAULT_BATCH_SIZE,
    RESNET50_REQUIRED_NUMBER_CHANNELS,
    SAMPLE_WIDTH,
    SAMPLE_HEIGHT,
    INCEPTION_RESNET_V2_WIDTH,
    INCEPTION_RESNET_V2_HEIGHT,
    RESNET_50_HEIGHT,
    RESNET_50_WIDTH)

from constants.ultrasound import (
    IMAGE_TYPE,
    NUMBER_CHANNELS_COLOR,
    NUMBER_CHANNELS_GRAYSCALE)

from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser()

parser.add_argument('benign_top_level_path',
                    help = 'absolute path to top level directory containing benign patient folders')

parser.add_argument('malignant_top_level_path',
                    help = 'absolute path to top level directory containing malignant patient folders')

parser.add_argument('manifest_path',
                    help = "absolute path to complete manifest file (merged benign and malignant manifests")

parser.add_argument("output_directory_path",
                    help = "absolute path to complete output directory for generated features and labels")

parser.add_argument("-bs",
                    "--batch_size",
                    type = int,
                    default = DEFAULT_BATCH_SIZE,
                    help = "Number of samples to extract from each input frame")

parser.add_argument("-idg",
                    "--image_data_generator",
                    type = ImageDataGenerator,
                    default = None,
                    help = "Keras ImageDataGenerator to preprocess sample image data")

parser.add_argument("-it",
                    "--image_type",
                    type = str,
                    default = "ALL",
                    help = "Image class to consider in manifest")

parser.add_argument("-ts",
                    "--target_shape",
                    type = list,
                    default = [RESNET_50_HEIGHT, RESNET_50_WIDTH],
                    help = "Size to use for image samples of taken frames. Used to pad frames that are smaller the target shape, and crop-sample images that are larger than the target shape")

parser.add_argument('-T', '--timestamp',
                    type = str,
                    default = None,
                    help = "String timestamp to use as prefix to focus directory and manifest directory")

parser.add_argument('-ofp', '--override_filename_prefix',
                    type = str,
                    default = None,
                    help = "String to use as filename for extracted features and logfile. Prefix only - do not include extension.")

args = parser.parse_args()

try:
    if args.override_filename_prefix is not None:
        logging_filename = "{}/{}.log".format(
            args.output_directory_path, 
            args.override_filename_prefix)
    else:
        logging_filename = "{}/extraction_{}_{}.log".format(
            args.output_directory_path,
            args.timestamp,
            uuid.uuid4())

    logging.basicConfig(level = logging.INFO, filename = logging_filename)

    exit_code = extract_save_patient_features(
        args.benign_top_level_path,
        args.malignant_top_level_path,
        args.manifest_path,
        args.output_directory_path,
        batch_size = args.batch_size,
        image_data_generator = args.image_data_generator,
        image_type = IMAGE_TYPE[args.image_type],
        target_shape = args.target_shape,
        timestamp = args.timestamp,
        override_filename_prefix = args.override_filename_prefix)


except Exception as e:
    raise(e)