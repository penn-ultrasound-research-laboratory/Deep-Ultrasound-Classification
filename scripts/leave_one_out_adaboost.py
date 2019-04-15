import argparse

import pandas as pd
import numpy as np

from os import listdir
from os.path import isfile, join
from itertools import combinations

from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score
)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def leave_one_out_adaboost(args):

    Accuracy = []
    AUC = []
    Sensitivity = []
    Specificity = []
    PositivePredictiveValue = []
    NegativePredictiveValue = []
    FalseNegativeRate = []

    composite_df = pd.read_csv(args["path"])

    for seed in composite_df["seed"].unique():
        
        train_df = composite_df
        test_df = composite_df[composite_df["seed"] == seed]

        bdt = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=1),
                n_estimators=10)

        # Fit AdaBoostClassifier on grouped training predictions
        train_features = train_df[["gray_predictions", "color_predictions"]].to_numpy()
        train_predictions = train_df["class"].astype("category").cat.codes

        test_features = test_df[["gray_predictions", "color_predictions"]].to_numpy()

        bdt.fit(train_features, train_predictions)

        pred_probs = bdt.predict_proba(test_features)[:,1]

        auc_score = roc_auc_score(test_df['class'].astype('category').cat.codes, pred_probs)
        acc_score = accuracy_score(test_df['class'].astype('category').cat.codes, pred_probs.round())

        cm = confusion_matrix(test_df['class'].astype('category').cat.codes, pred_probs.round())               
        TP = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[1][1]
        TPR = TP/(TP+FN) # Sensitivity, hit rate, recall
        TNR = TN/(TN+FP) # Specificity or true negative rate
        PPV = TP/(TP+FP) # Precision or positive predictive value
        NPV = TN/(TN+FN) # Negative predictive value
        FNR = FN/(TP+FN) # False negative rate

        Accuracy.append(acc_score)
        AUC.append(auc_score)
        Sensitivity.append(TPR)
        Specificity.append(TNR)
        PositivePredictiveValue.append(PPV)
        NegativePredictiveValue.append(NPV)
        FalseNegativeRate.append(FNR)

    print("Accuracy: {0}".format(np.mean(Accuracy)))
    print("AUC: {0}".format(np.mean(AUC)))
    print("Sensitivity: {0}".format(np.mean(Sensitivity)))
    print("Specificity: {0}".format(np.mean(Specificity)))
    print("PPV: {0}".format(np.mean(PositivePredictiveValue)))
    print("NPV: {0}".format(np.mean(NegativePredictiveValue)))
    print("FNR: {0}".format(np.mean(FalseNegativeRate)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-P",
        "--path",
        help="Path to file containing composite patient aggregated DataFrame splits",
        required=True
    )

    args = parser.parse_args()
    leave_one_out_adaboost(args.__dict__)