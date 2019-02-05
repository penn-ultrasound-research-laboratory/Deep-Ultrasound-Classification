import cv2
import json
import logging
import os
import uuid

import numpy as np

from src.pipeline.patientsample.patient_sample_generator import PatientSampleGenerator
from src.constants.ultrasound import IMAGE_TYPE

from keras.preprocessing.image import ImageDataGenerator

dirname = os.path.dirname(__file__)

with open(os.path.abspath("../Datasets/V2.0_Processed/manifest.json"), "r") as fp:
    manifest = json.load(fp)

logging.basicConfig(level=logging.INFO, filename="./main_output.log")

# W/ aim of generating graphics for paper / email. Featurwise normalize according to mean or std.
# That should be used only in image preprocessing pipeline. Issue is that negative scaled values are
# meaningless when saving to file or displaying as negative values are truncated to zero.

image_data_generator = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True)

BATCH_SIZE = 5
MAX_PLOTTING_ROWS = 10

# Limit to small subset of patients

patient_sample_generator = next(PatientSampleGenerator([
    ("01PER2043096", "BENIGN"),
    ("79BOY3049163", "MALIGNANT"),
    ("93KUD3041008", "MALIGNANT")],
    os.path.join(dirname, "../Datasets/V2.0_Processed/Benign"),
    os.path.join(dirname, "../Datasets/V2.0_Processed/Malignant"),
    manifest,
    target_shape=[50, 50],
    image_type=IMAGE_TYPE.ALL,
    image_data_generator=image_data_generator,
    batch_size=BATCH_SIZE,
    kill_on_last_patient=True
))

count = 0
plot_rows = []
try:
    while count < MAX_PLOTTING_ROWS:
        raw_image_batch, labels = next(patient_sample_generator)

        split = np.split(raw_image_batch, raw_image_batch.shape[0], axis=0)
        split = [np.squeeze(img, axis=0) for img in split]

        # cv2.imshow("sample", np.hstack(split))
        # cv2.waitKey(0)

        plot_rows.append(np.hstack(split))

        count += 1

except Exception as e:
    print(e)

sample_thumbnails = np.vstack(plot_rows)

# cv2.imshow("sample", sample_thumbnails)
# cv2.waitKey(0)

cv2.imwrite('thumbnails_{}.png'.format(uuid.uuid4()), sample_thumbnails)
