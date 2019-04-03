
import argparse
import json
import yaml
import os
import pkg_resources

from dotmap import DotMap
from datetime import datetime
from importlib import import_module

import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.python.lib.io import file_io
from tensorflow.python.framework.errors_impl import NotFoundError

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard
from keras_preprocessing.image import ImageDataGenerator

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score


from constants.ultrasound import string_to_image_type, TUMOR_TYPES
from pipeline.patientsample.patient_sample_generator import PatientSampleGenerator
from utilities.partition.patient_partition import patient_train_test_split
from utilities.general.general import default_none
from utilities.manifest.manifest import patient_type_lists, patient_lists_to_dataframe
from utilities.image.image import crop_generator

def train_model(args):

    IN_LOCAL_TRAINING_MODE = not args.job_dir
    JOB_DIR = default_none(args.job_dir, ".")
    LOGS_PATH = "{0}/logs".format(JOB_DIR)
    CONFIG_FILE = default_none(args.config, "../config/default.yaml")
    MODEL_FILE = "{0}.h5".format(args.identifier)
    TRAIN_DF_FILE = "{0}_train.csv".format(args.identifier)
    VALIDATION_DF_FILE = "{0}_validation.csv".format(args.identifier)
    ROC_DF_FILE = "{0}_roc.csv".format(args.identifier)
    PR_DF_FILE = "{0}_precision_recall.csv".format(args.identifier)
    SCORES_DF_FILE = "{0}_scores.csv".format(args.identifier)
    HISTORY_DF_FILE = "{0}_history.csv".format(args.identifier)
    GC_MODEL_SAVE_PATH = "{0}/model/{1}".format(JOB_DIR, MODEL_FILE)
    GC_TRAIN_DF_SAVE_PATH = "{0}/data/{1}".format(JOB_DIR, TRAIN_DF_FILE)
    GC_VALIDATION_DF_SAVE_PATH = "{0}/data/{1}".format(JOB_DIR, VALIDATION_DF_FILE)
    GC_HISTORY_DF_SAVE_PATH = "{0}/data/{1}".format(JOB_DIR, HISTORY_DF_FILE)
    GC_ROC_DF_SAVE_PATH = "{0}/data/{1}".format(JOB_DIR, ROC_DF_FILE)
    GC_PR_DF_SAVE_PATH = "{0}/data/{1}".format(JOB_DIR, PR_DF_FILE)
    GC_SCORES_DF_SAVE_PATH = "{0}/data/{1}".format(JOB_DIR, SCORES_DF_FILE)

    # Load the configuration file yaml file if provided
    try:
        print("Loading configuration file from: {0}".format(CONFIG_FILE))
        with file_io.FileIO(CONFIG_FILE, mode='r') as stream:
            config = DotMap(yaml.load(stream))
    except NotFoundError as _:
        print("Configuration file not found: {0}".format(CONFIG_FILE))
        return
    except Exception as _:
        print("Unable to load configuration file: {0}".format(CONFIG_FILE))
        return

    # Load the manifest file
    try:
        print("Loading manifest file from: {0}".format(args.manifest))
        with file_io.FileIO(args.manifest, mode='r') as stream:
            manifest = json.load(stream)
    except NotFoundError as _:
        print("Manifest file not found: {0}".format(args.manifest))
        return
    except Exception as _:
        print("Unable to load manifest file: {0}".format(args.manifest))
        return

    np.random.seed(config.random_seed)
    tf.set_random_seed(config.random_seed)

    tb_callback = TensorBoard(
        log_dir=LOGS_PATH,
        batch_size=config.batch_size,
        write_graph=False)

    benign_patients, malignant_patients = patient_type_lists(manifest)

    # For local testing of models/configuration, limit to six patients of each type
    if IN_LOCAL_TRAINING_MODE:
        print("Local training test. Limiting to six patients from each class.")
        benign_patients = np.random.choice(
            benign_patients, 6, replace=False).tolist()
        malignant_patients = np.random.choice(
            malignant_patients, 6, replace=False).tolist()

    # Train/test split according to config
    patient_split = DotMap(patient_train_test_split(
        benign_patients,
        malignant_patients,
        config.train_split,
        validation_split=config.validation_split,
        random_seed=config.random_seed
    ))

    # Crawl the manifest to assemble training DataFrame of matching patient frames
    train_df = patient_lists_to_dataframe(
        patient_split.train,
        manifest,
        string_to_image_type(config.image_type),
        args.images + "/Benign",
        args.images + "/Malignant")

    # Print some sample information
    print("Training DataFrame shape: {0}".format(train_df.shape))
    print("Training DataFrame class breakdown")
    print(train_df["class"].value_counts())
    
    train_data_generator = ImageDataGenerator(
        **config.image_preprocessing_train.toDict())
    test_data_generator = ImageDataGenerator(
        **config.image_preprocessing_test.toDict())

    train_generator = train_data_generator.flow_from_dataframe(
        dataframe=train_df,
        directory=None,
        x_col="filename",
        y_col="class",
        target_size=config.target_shape,
        color_mode="rgb",
        class_mode="binary",
        classes=TUMOR_TYPES,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.random_seed,
        drop_duplicates=False
    )

    # Optional: subsample each input to batch of randomly placed crops
    if config.subsample.subsample_shape:
        train_generator = crop_generator(
            train_generator,
            config.subsample.subsample_shape,
            config.subsample.subsample_batch_size)

    # Optional: assemble validation DataFrame and validation generator
    if config.validation_split:
        validation_df = patient_lists_to_dataframe(
            patient_split.validation,
            manifest,
            string_to_image_type(config.image_type),
            args.images + "/Benign",
            args.images + "/Malignant")

        print("Validation DataFrame class breakdown")
        print(validation_df["class"].value_counts())

        validation_generator = test_data_generator.flow_from_dataframe(
            dataframe=validation_df,
            directory=None,
            x_col="filename",
            y_col="class",
            target_size=config.target_shape,
            color_mode="rgb",
            class_mode="binary",
            classes=TUMOR_TYPES,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.random_seed,
            drop_duplicates=False
        )
    else:
        # Config does not specify validation split
        validation_generator = None

    if not IN_LOCAL_TRAINING_MODE:
        # Save the training data on GC storage
        with file_io.FileIO(GC_TRAIN_DF_SAVE_PATH, mode="wb+") as output_f:
            train_df.to_csv(output_f)

        # Save the validation data on GC storage
        with file_io.FileIO(GC_VALIDATION_DF_SAVE_PATH, mode="wb+") as output_f:
            validation_df.to_csv(output_f)

    with tf.device('/device:GPU:0'):

        # Load the model specified in config
        model = import_module("models.{0}".format(
            config.model)).get_model(config)

        model.compile(
            optimizer=Adam(lr=config.learning_rate),
            loss=config.loss,
            metrics=['accuracy'])

        # Train the classifier on top of the base model
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_df) // config.batch_size,
            epochs=config.training_epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_df) // config.batch_size,
            verbose=2,
            use_multiprocessing=True,
            workers=args.num_workers
        )

        # Fine tune the base model if specified in config
        if config.fine_tune:
            for layer_name in config.fine_tune.layers:                
                print("Setting layer {0} trainable".format(layer_name))
                # Set the next layer down as Trainable
                model.get_layer(layer_name).trainable = True
                       
            # Recompile the model to reflect new trainable layer
            model.compile(
                optimizer=Adam(lr=config.fine_tune.learning_rate),
                loss=config.loss,
                metrics=['accuracy'])

            start_epoch = history.params['epochs']
            print("Starting fine-tune training at epoch: {0}".format(start_epoch))

            # Fit the generator with this number of epochs
            history = model.fit_generator(
                train_generator,
                steps_per_epoch=len(train_df) // config.batch_size,
                epochs=config.fine_tune.epochs,
                validation_data=validation_generator,
                validation_steps=len(validation_df) // config.batch_size,
                verbose=2,
                use_multiprocessing=True,
                workers=args.num_workers,
                callbacks=[tb_callback]
            )

    '''
    Evaluate
    '''

    test_generator = test_data_generator.flow_from_dataframe(
        dataframe=validation_df,
        directory=None,
        x_col="filename",
        y_col="class",
        target_size=config.target_shape,
        color_mode="rgb",
        class_mode="binary",
        classes=TUMOR_TYPES,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.random_seed,
        drop_duplicates=False
    )

    model_predictions = model.predict_generator(
        test_generator,
        steps=(len(validation_df) // config.batch_size) + 1,
        use_multiprocessing=True,
        verbose=1
    )

    # Compute AUC score
    auc_score = roc_auc_score(validation_df['class'].astype('category').cat.codes, model_predictions)
    # ROC curve 
    fpr, tpr, roc_thresholds = roc_curve(validation_df['class'], model_predictions, pos_label='MALIGNANT')
    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(validation_df['class'], model_predictions, pos_label='MALIGNANT')
    # Confusion matrix based metrics    
    cm = confusion_matrix(validation_df['class'].astype('category').cat.codes, model_predictions.round())               
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    TPR = TP/(TP+FN) # Sensitivity, hit rate, recall
    TNR = TN/(TN+FP) # Specificity or true negative rate
    PPV = TP/(TP+FP) # Precision or positive predictive value
    NPV = TN/(TN+FN) # Negative predictive value
    FNR = FN/(TP+FN) # False negative rate

    roc_df = pd.DataFrame(data={'fpr':fpr, 'tpr':tpr, 'thresholds':roc_thresholds})

    pr_df = pd.DataFrame(data={'recall':recall[:-1], 'precision':precision[:-1], 'thresholds':pr_thresholds})

    scores_df = pd.DataFrame(data={'AUC':auc_score, 'Sensitivity': TPR, 'Specificity':TNR, 'PPV': PPV, 'NPV':NPV, 'FNR':FNR, 'TP':TP, 'FP':FP, 'FN':FN, 'TN':TN}, index=[0])
    
    # Enforce column headers
    scores_df = scores_df[['AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'FNR', 'TP', 'FP', 'FN', 'TN']]

    if not IN_LOCAL_TRAINING_MODE:
        # Save the model
        model.save_weights(MODEL_FILE)

        # Save the model on GC storage
        with file_io.FileIO(MODEL_FILE, mode="rb") as input_f:
            with file_io.FileIO(GC_MODEL_SAVE_PATH, mode="wb+") as output_f:
                output_f.write(input_f.read())

        # Save the training history on GC storage
        with file_io.FileIO(GC_HISTORY_DF_SAVE_PATH, mode="wb+") as output_f:
            pd.DataFrame(history.history).to_csv(output_f, index=False)

        # Save the ROC curve on GC storage
        with file_io.FileIO(GC_ROC_DF_SAVE_PATH, mode="wb+") as output_f:
            roc_df.to_csv(output_f, index=False)

        # Save the Precision-Recall Curve on GC storage
        with file_io.FileIO(GC_PR_DF_SAVE_PATH, mode="wb+") as output_f:
            pr_df.to_csv(output_f, index=False)
        
        # Save the confusion matrix metrics on GC storage
        with file_io.FileIO(GC_SCORES_DF_SAVE_PATH, mode="wb+") as output_f:
            scores_df.to_csv(output_f, index=False)
        

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
        help="Experiment config yaml. i.e. experiment definition in code. Must be place in /src/config directory.",
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
        help="the directory for logging in GC",
        default=None
    )

    parser.add_argument(
        "-i",
        "--identifier",
        help="Base name to identify job in Google Cloud Storage & ML Engine",
        default=None
    )

    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of data loading workers')
    parser.add_argument('--disp-step', type=int, default=200,
                        help='display step during training')
    parser.add_argument('--cuda', type=bool, default=True, help='enable CUDA')

    args = parser.parse_args()
    arguments = DotMap(args.__dict__)

    # config argument passed-in is a filename. Locate the config file in the config directory
    if arguments.config:
        arguments.config = pkg_resources.resource_filename(
            __name__,
            "{0}/{1}".format("../config", arguments.config))

    # Train the model
    train_model(arguments)
