# Import libraries
import argparse
import glob
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


# Load the prepared data file in the training folder
def load_data(data_path):
    print("Loading Data...")
    all_files = glob.glob(data_path + "/*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)

    return df


# Separate features and labels
def separate_data(df):
    X, y = df[['Pregnancies', 'PlasmaGlucose',
               'DiastolicBloodPressure', 'TricepsThickness',
               'SerumInsulin', 'BMI', 'DiabetesPedigree',
               'Age']].values, df['Diabetic'].values
   
    return X, y


# Split data into training set and test set
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=0)

    return X_train, X_test, y_train, y_test


# Train a decision tree model
def train_model(X_train, y_train):
    print('Training a decision tree model...')
    model = DecisionTreeClassifier().fit(X_train, y_train)

    return model


# Calculate accuracy
def calculate_acc(model, X_test, y_test):
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)

    return acc


# Calculate AUC
def calculate_auc(model, X_test, y_test):
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:, 1])
    print('AUC: ' + str(auc))

    return y_scores, auc


# Plot ROC curve
def plot_roc(y_test, y_scores):
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])
    plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("ROCcurve.png")
    mlflow.log_artifact("ROCcurve.png")


def main():
    print("Running train.py")

    # Get parameters
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--training_data", type=str, help="Path to training data")
    parser.add_argument("--model_output", type=str, help="Path of output model")

    args = parser.parse_args()

    data_path = args.training_data

    # Load data
    df = load_data(data_path)

    # Separate features and labels
    X, y = separate_data(df)

    # Split training and test data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train)
    # Output the model
    pickle.dump(model, open((Path(args.model_output) / "model.sav"), "wb"))

    # Calculate accuracy
    acc = calculate_acc(model, X_test, y_test)
    mlflow.log_metric('Accuracy', np.float(acc))

    # Calculate AUC
    y_scores, auc = calculate_auc(model, X_test, y_test)
    mlflow.log_metric('AUC', np.float(auc))

    # Plot ROC
    plot_roc(y_test, y_scores)


main()
