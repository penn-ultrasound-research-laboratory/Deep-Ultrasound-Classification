import argparse
import os

from src.constants.ultrasound import (
    FOCUS_HASH_LABEL,
    FRAME_LABEL,
    HSV_COLOR_THRESHOLD,
    HSV_GRAYSCALE_THRESHOLD,
    IMAGE_TYPE,
    IMAGE_TYPE_LABEL,
    INTERPOLATION_FACTOR_LABEL,
    READOUT_ABBREVS as RA,
    SCALE_LABEL,
    TUMOR_BENIGN,
    TUMOR_MALIGNANT,
    TUMOR_TYPE_LABEL
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--dir-path", help="Top level dataset path", required=True
    )

    parser.add_argument(
        "-M",
        "--manifest-output",
        help="Path to directory to output manifest",
        default=".",
        required=False,
    )

    parser.add_argument(
        "-f",
        "--frames-folder",
        help="Path to patient files in the patient folder",
        default=".",
        required=False,
    )

    args = parser.parse_args().__dict__

    path_to_benign_dir = os.path.join(args["dir_path"], "Benign")
    path_to_malignant_dir = os.path.join(args["dir_path"], "Malignant")

    patient_records = {}

    all_patients = [
        (patient_label, patient_type_label, path)
        for path, patient_type_label in [
            (path_to_benign_dir, TUMOR_BENIGN),
            (path_to_malignant_dir, TUMOR_MALIGNANT),
        ]
        for patient_label in [
            name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
        ]
    ]

    print(all_patients)
    exit

    # Process each individual patient
    for patient_label, patient_type_label, path in tqdm(all_patients, desc="OCR"):

        LOGGER.info("OCR | Processing patient: %s", patient_label)

        patient_directory_absolute_path = "{}/{}".format(
            path.rstrip("/"), patient_label
        )

        acquired_records = patient_ocr(
            patient_directory_absolute_path,
            patient_label,
            rel_path_to_frames_folder,
            rel_path_to_focus_output_folder,
            timestamp,
            composite_records=None,
            patient_type_label=patient_type_label,
        )

        patient_records[patient_label] = acquired_records

    # Dump the patient records to file
    manifest_absolute_path = "{}/manifest_{}.json".format(
        path_to_manifest_output_dir.rstrip("/"), timestamp
    )

    manifest_file = open(manifest_absolute_path, "a")

    json.dump(patient_records, manifest_file)

    # Cleanup
    manifest_file.close()
