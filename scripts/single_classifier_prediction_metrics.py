import argparse
import os
import pkg_resources

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score
# Build a patient level DataFrame
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from scipy import stats

def prediction_metrics(args):

    validation_df = pd.read_csv(args["validation"])
    predictions_df = pd.read_csv(args["predictions"])

    validation_df['patient'] = validation_df['filename'].str.extract(r'gs://research-storage/V4.0_Processed/(?:Benign|Malignant)/(\w+)/', expand=False)
    validation_df['predictions'] = predictions_df['predictions']
    validation_df['rounded_mean'] = validation_df['predictions'].round()
    validation_df['rounded_any'] = validation_df['predictions'].round()

    patient_df = validation_df.groupby(['patient']).agg({
        'class': lambda x: stats.mode(x)[0][0],
        'predictions':'mean',
        'rounded_mean':'mean',
        'rounded_any':'any'
        })

    # Compute AUC score
    auc_score = roc_auc_score(patient_df['class'].astype('category').cat.codes, patient_df['predictions'])
    acc_score = accuracy_score(patient_df['class'].astype('category').cat.codes, patient_df['predictions'].round())

    cm = confusion_matrix(patient_df['class'].astype('category').cat.codes, patient_df['predictions'].round())               
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    TPR = TP/(TP+FN) # Sensitivity, hit rate, recall
    TNR = TN/(TN+FP) # Specificity or true negative rate
    PPV = TP/(TP+FP) # Precision or positive predictive value
    NPV = TN/(TN+FN) # Negative predictive value
    FNR = FN/(TP+FN) # False negative rate

    print("Accuracy: {0}".format(acc_score))
    print("AUC: {0}".format(auc_score))
    print("Sensitivity: {0}".format(TPR))
    print("Specificity: {0}".format(TNR))
    print("PPV: {0}".format(PPV))
    print("NPV: {0}".format(NPV))
    print("FNR: {0}".format(FNR))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-V",
        "--validation",
        help="Path to validation DataFrame",
        required=True
    )

    parser.add_argument(
        "-P",
        "--predictions",
        help="Path to prediction DataFrame",
        required=True
    )

    args = parser.parse_args()
    prediction_metrics(args.__dict__)
