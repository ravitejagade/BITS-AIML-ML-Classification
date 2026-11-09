"""Train five classifiers for BITS Pilani Machine Learning Assignment 2.

Run from the project directory:
    python train_model.py

Creates dataset.csv, test_data.csv, metrics.csv, feature_columns.json and
five serialized estimators under models/.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load the public Wisconsin Diagnostic Breast Cancer dataset bundled with
    # scikit-learn. It contains the original 569 records and 30 numeric features.
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.copy()

    # In the source data, target 0 = malignant and target 1 = benign.
    # Keep the clinically familiar source labels in the CSV: M and B.
    diagnosis = pd.Series(np.where(bunch.target.to_numpy() == 0, "M", "B"), name="diagnosis")
    dataset = X.copy()
    dataset["diagnosis"] = diagnosis
    dataset.to_csv(BASE_DIR / "dataset.csv", index=False)

    feature_columns = X.columns.tolist()
    with (BASE_DIR / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    # Positive class is M (malignant) = 1; B (benign) = 0.
    y = (diagnosis == "M").astype(int)
    X_train, X_test, y_train, y_test, diag_train, diag_test = train_test_split(
        X,
        y,
        diagnosis,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_data = X_test.copy()
    test_data["diagnosis"] = diag_test.to_numpy()
    test_data.to_csv(BASE_DIR / "test_data.csv", index=False)

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=5, min_samples_leaf=2
        ),
        "KNN": Pipeline(
            [("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]
        ),
        "Gaussian Naive Bayes": Pipeline(
            [("scaler", StandardScaler()), ("classifier", GaussianNB())]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]
        ),
    }

    file_names = {
        "Logistic Regression": "Logistic_Regression.pkl",
        "Decision Tree": "Decision_Tree.pkl",
        "KNN": "KNN.pkl",
        "Gaussian Naive Bayes": "Gaussian_Naive_Bayes.pkl",
        "SVM": "SVM.pkl",
    }

    rows: list[dict[str, float | str]] = []
    for model_name, model in models.items():
        print(f"Training: {model_name}")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        row = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": roc_auc_score(y_test, probabilities),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1 Score": f1_score(y_test, predictions, zero_division=0),
            "MCC": matthews_corrcoef(y_test, predictions),
        }
        rows.append(row)
        joblib.dump(model, MODEL_DIR / file_names[model_name])

    metrics = pd.DataFrame(rows)
    metrics.to_csv(BASE_DIR / "metrics.csv", index=False)

    print("\nALL FIVE MODELS TRAINED SUCCESSFULLY")
    print(metrics.to_string(index=False, formatters={c: (lambda v: f"{v:.4f}") for c in metrics.columns if c != "Model"}))
    print(f"\nDataset records: {len(dataset)}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print("Training and evaluation completed!")
    print(f"Files saved in: {BASE_DIR}")


if __name__ == "__main__":
    main()