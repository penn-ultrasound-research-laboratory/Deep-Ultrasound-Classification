# Deep Ultrasound Classification
Repository for ultrasound research collaboration between Penn SEAS and Radiology at Penn Medicine

*************************************************************************************************
Notice: This repository will be shifting to Google Cloud Platform in the near-future
*************************************************************************************************

## Objective

Improve the classification of breast ultrasound images using state-of-the-art techniques in deep learning. Our goal is to use recent advances in CNNs to achieve best in domain performance on classifying breast tumor ultrasound to diagnosis whether a tumor is benign or malignant. 

## Data Directory Structure

Data is expected to loosely match the following hierarchy. Note that folder names can be anything - i.e. Malignant folder can be named "Malignant", "Mal", "M" etc. Loose structure is enforced - malignant/benign top level folders contain all patients for that class. In each patient folder there must be a "frames" folder and a "focus" folder at some nested level. It is easiest to leave all folders exactly one level below the parent folder, but not required. Paths to various directories (PatientData, benign, malignant, focus, etc.) passed at runtime as arguments to CLI scripts. 

```bash
PatientData/
├── Malignant/
│   ├── Patient1/
│   │   ├── frames/
│   │   │   ├── frame_0001.png
│   │   │   └── frame_0002.png
│   │   └── focus_timestamp/
│   │       ├── focus_49d0df73-3a1e-4cb1-94fa-5a7bf0af069f.png    # focus of frame_0001.png       
│   │       └── focus_64919505-864a-41d5-a9ac-e128cd17c5e0.png    # focus of frame_0002.png
│   └── Patient2
│       ├── frames/
│       │   ├── frame_0001.png
│       │   └── frame_0002.png
│       └── focus_timestamp/
│           ├── focus_cf69642e-0153-4199-9b78-49fd1a6f6334.png    # focus of frame_0001.png   
│           └── focus_53f5006e-4e0a-4a5c-8f90-952486500d81.png    # focus of frame_0002.png
└── Benign/
    ├── Patient1/
    │   ├── frames/
    │   │   ├── frame_0001.png
    │   │   └── frame_0002.png
    │   └── focus_timestamp/
    │       ├── focus_55b023ed-76c4-4bf5-94cc-eb5bc88a7b21.png    # focus of frame_0001.png
    │       └── focus_835511b3-aff1-4ad7-a3e1-2a44c81a4a54.png    # focus of frame_0002.png
    └── Patient2
        ├── frames/
        │   ├── frame_0001.png
        │   └── frame_0002.png
        └── focus_timestamp/
            ├── focus_b9ad730b-2f1e-42d8-b532-deaf4d6ec696.png    # focus of frame_0001.png 
            └── focus_1ac74dff-f5ed-440b-a1fe-72e4d19d076f.png    # focus of frame_0002.png
```

## Dependency Notes

Notice that the source code (/src) includes its own copy of keras-preprocessing. This is a fork of keras-preprocessing v1.0.9 and includes two fixes: 

- allows file loading from Google Cloud storage by wrapping pillow image load with TensorFlow:
    - `pil_image.open(path, mode='rb')` (*before*)
    - `pil_image.open(tensorflow.python.lib.io.ile_io.FileIO(path, mode='rb'))` (*after*)
- Removes os file exist check from dataframe_iterator.py