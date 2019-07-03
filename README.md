# Deep Ultrasound Classification
Repository for ultrasound research collaboration between Penn SEAS and Radiology at Penn Medicine


## Objective

Improve the classification of breast ultrasound images using state-of-the-art techniques in deep learning. Our goal is to use recent advances in CNNs to achieve best in domain performance on classifying breast tumor ultrasound to diagnosis whether a tumor is benign or malignant. 

## Data Directory Structure

Data stored in Google Cloud Storage is expected to match the following hierarchy. All frames must be present at the top level of each patient folder. Do **not** place patient frames in a nested folder. Use the file tree below as a guide.

```bash
PatientData/
├── Malignant/
│   ├── Patient1/
│   │       ├── frame_0001.png
│   │       └── frame_0002.png
│   └── Patient2
│           ├── frame_0001.png
│           └── frame_0002.png
└── Benign/
    ├── Patient1/
    │       ├── frame_0001.png
    │       └── frame_0002.png
    └── Patient2
            ├── frame_0001.png
            └── frame_0002.png
```

# Installation for Local Development

**Do not use setup.py**. The setup.py file is used for Google Cloud to install the code on the cloud instance. Case in point, TensorFlow is not listed as a dependency in setup.py but is obviously a dependency of this repository. This is because Google AI Platform cloud instances already have TensorFlow installed. Instead, use your local package manager (pip or Anaconda) to install the dependencies listed in setup.py + TensorFlow v1.5.1.

## Dependency Notes

Notice that the source code (/src) includes its own copy of keras-preprocessing. This is a fork of keras-preprocessing v1.0.9 and includes two fixes: 

- allows file loading from Google Cloud storage by wrapping pillow image load with TensorFlow:
    - `pil_image.open(path, mode='rb')` (*before*)
    - `pil_image.open(tensorflow.python.lib.io.ile_io.FileIO(path, mode='rb'))` (*after*)
- Removes os file exist check from dataframe_iterator.py

## Local Testing

Execute the module as main (i.e. `python3 -m trainer.task`) to see argparse output detailing what arguments are needed to run the training module.

An example execution of the training module that uses the "local.yaml" configuration file. Note, you must execute from the \src directory:

```
cd src
python3 -m trainer.task --images ../../Dataset/V4.0_Processed --manifest ../../Dataset/V4.0_Processed/manifest.json --config local.yaml
```

## Google Cloud Training

1. **Navigate to gc-train.py and modify lines 7-9 to reflect your local development environment**

2. Run the "gc-train.py" script via command line to push a new training job to Google Cloud (i.e. `python3 -m gc-train [arguments]`)

Example to run a Google Cloud job with a dataset named *V4.0_Processed*, manifest named *manifest.json* stored in Google Cloud Storage, configuration file *local.yaml* stored in \src\config, and test run identifier *my_test_run*:

```
python3 -m gc-train -I V4.0_Processed -M manifest.json -C local.yaml -i my_test_run
```