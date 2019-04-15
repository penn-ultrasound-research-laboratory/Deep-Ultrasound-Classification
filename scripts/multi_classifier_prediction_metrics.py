import argparse
import os
import pkg_resources

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Build a patient level DataFrame
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from scipy import stats

def prediction_metrics(args):

    gray_validation_df = pd.read_csv(args["gray_validation"])
    gray_predictions_df = pd.read_csv(args["gray_predictions"])

    color_validation_df = pd.read_csv(args["color_validation"])
    color_predictions_df = pd.read_csv(args["color_predictions"])

    gray_validation_df["patient"] = gray_validation_df["filename"].str.extract(r"gs://research-storage/V(?:\d+).0_Processed/(?:Benign|Malignant)/(\w+)/", expand=False)

    color_validation_df["patient"] = color_validation_df["filename"].str.extract(r"gs://research-storage/V(?:\d+).0_Processed/(?:Benign|Malignant)/(\w+)/", expand=False)

    # Wrong. This needs to be a JOIN
    gray_validation_df["gray_predictions"] = gray_predictions_df["predictions"]
    color_validation_df["color_predictions"] = color_predictions_df["predictions"]

    gray_patient_df = gray_validation_df.groupby(["patient"]).agg({
        "class": lambda x: stats.mode(x)[0][0],
        "gray_predictions":"mean"
    }).reset_index()

    color_patient_df = color_validation_df.groupby(["patient"]).agg({
        "class": lambda x: stats.mode(x)[0][0],
        "color_predictions": "mean"
    }).reset_index()

    comp_patient_df = gray_patient_df.merge(color_patient_df.drop("class", axis=1), on="patient", how="inner")
    # comp_patient_df.drop("class_y", inplace=True, axis=1)

    comp_patient_df.to_csv("composite.csv", index=False)


    bdt = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators=50)

    # Fit AdaBoostClassifier on grouped training predictions
    train_features = comp_patient_df.as_matrix(columns=["gray_predictions", "color_predictions"])
    train_predictions = comp_patient_df["class"].astype("category").cat.codes

    bdt.fit(train_features, train_predictions)

    pred_probs = bdt.predict_proba(train_features)[:,1]
    pred_log_probs = bdt.predict_log_proba(train_features)[:,1]

    # Plot the decision boundaries
    # plt.subplot(121)
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
    #                     np.arange(y_min, y_max, plot_step))

    # Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    # cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    # plt.axis("tight")

    # # Plot the training points
    # for i, n, c in zip(range(2), class_names, plot_colors):
    #     idx = np.where(y == i)
    #     plt.scatter(X[idx, 0], X[idx, 1],
    #                 c=c, cmap=plt.cm.Paired,
    #                 s=20, edgecolor='k',
    #                 label="Class %s" % n)
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    # plt.legend(loc='upper right')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Decision Boundary')

    # patient_df = patient_df[(patient_df["gray_predictions"] > 0])&(patient_df['color_predictions'] > 0))

    # Compute AUC score
    auc_score = roc_auc_score(comp_patient_df['class'].astype('category').cat.codes, pred_probs)
    acc_score = accuracy_score(comp_patient_df['class'].astype('category').cat.codes, pred_probs.round())

    cm = confusion_matrix(comp_patient_df['class'].astype('category').cat.codes, pred_probs.round())               
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
        "-GV",
        "--gray-validation",
        help="Path to validation DataFrame",
        required=True
    )

    parser.add_argument(
        "-CV",
        "--color-validation",
        help="Path to validation DataFrame",
        required=True
    )

    parser.add_argument(
        "-GP",
        "--gray-predictions",
        help="Path to prediction DataFrame",
        required=True
    )

    parser.add_argument(
        "-CP",
        "--color-predictions",
        help="Path to prediction DataFrame",
        required=True
    )

    args = parser.parse_args()
    prediction_metrics(args.__dict__)
