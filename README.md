# Deep Ultrasound Classification
Repository for ultrasound research collaboration between Penn SEAS and Radiology at Penn Medicine

*************************************************************************************************
Notice: This repository will be shifting to Google Cloud Platform in the near-future
*************************************************************************************************

## Objective

Improve the classification of breast ultrasound images using state-of-the-art techniques in deep learning. Our goal is to use recent advances in CNNs to achieve best in domain performance on classifying breast tumor ultrasound to diagnosis whether a tumor is benign or malignant. 

## Data Directory Structure

Data is expected to loosely match the following hierarchy. Note that folder names can be anything - i.e. Malignant folder can be named "Malignant", "Mal", "M" etc. Loose structure is enforced - malignant/benign top level folders contain all patients for that class. All frames must be present at the top level of each patient folder. Do **not** place patient frames in a nested folder. Use the file tree below as a guide.

```bash
PatientData/
├── Malignant/
│   ├── Patient1/
│   │   └── frames/
│   │       ├── frame_0001.png
│   │       └── frame_0002.png
│   └── Patient2
│       └── frames/
│           ├── frame_0001.png
│           └── frame_0002.png
└── Benign/
    ├── Patient1/
    │   └── frames/
    │       ├── frame_0001.png
    │       └── frame_0002.png
    └── Patient2
        └── frames/
            ├── frame_0001.png
            └── frame_0002.png
```

## Dependency Notes

Notice that the source code (/src) includes its own copy of keras-preprocessing. This is a fork of keras-preprocessing v1.0.9 and includes two fixes: 

- allows file loading from Google Cloud storage by wrapping pillow image load with TensorFlow:
    - `pil_image.open(path, mode='rb')` (*before*)
    - `pil_image.open(tensorflow.python.lib.io.ile_io.FileIO(path, mode='rb'))` (*after*)
- Removes os file exist check from dataframe_iterator.py

## Local Testing

Execute the module as main (i.e. `python3 -m trainer.task`) to see argparse output detailing what arguments are needed to run the training module.

An example execution of the training module that uses the "local.yaml" configuration file:

`python3 -m trainer.task --images [../relative/path/from_current_directory_to_images] --manifest [../relative/path/from current_directory_to_manifest] --config config/local.yaml`

## Google Cloud Training

Run the "gc-train.py" script via command line to push a new training job to Google Cloud. NOTE: Examine the file gc-train.py file. Some user configuration IS required - i.e. setting your local path and bucket name. 

Run the script via command line with `python3 -m gc-train [arguments]`

Example:

`python3 -m gc-train -I V1.0 -M manifest.json -C default.yaml`