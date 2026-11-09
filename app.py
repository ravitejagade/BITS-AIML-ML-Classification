import streamlit as st
import pandas as pd
import joblib
import os
import json

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Breast Cancer Classification Dashboard")
st.write("Performance evaluation of five machine learning classification models.")

MODEL_DIR = "models"

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Gaussian NB",
    "SVM"
]

# These filenames match the files in your models folder.
MODEL_FILES = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Gaussian NB": "Gaussian_Naive_Bayes.pkl",
    "SVM": "SVM.pkl"
}

@st.cache_resource
def load_models():
    loaded_models = {}
    for name, filename in MODEL_FILES.items():
        model_path = os.path.join(MODEL_DIR, filename)
        loaded_models[name] = joblib.load(model_path)
    return loaded_models

@st.cache_data
def load_features():
    with open("feature_columns.json", "r") as file:
        return json.load(file)

try:
    models = load_models()
    feature_columns = load_features()
except Exception as error:
    st.error(f"Unable to load saved models or feature columns: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload test dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(data.head())

    if "diagnosis" not in data.columns:
        st.error("The CSV must contain a 'diagnosis' column with M/B values.")
        st.stop()

    y = data["diagnosis"].astype(str).str.strip().str.upper().map(
        {"B": 0, "M": 1}
    )

    if y.isnull().any():
        st.error("The diagnosis column must contain only M or B values.")
        st.stop()

    missing_features = [
        column for column in feature_columns
        if column not in data.columns
    ]

    if missing_features:
        st.error(f"Missing required features: {missing_features}")
        st.stop()

    X = data[feature_columns]

    if X.isnull().any().any():
        st.error("The uploaded feature data contains missing values.")
        st.stop()

    st.subheader("Select Machine Learning Model")
    selected_model = st.selectbox("Choose a model", MODEL_NAMES)

    results = []

    for name, model in models.items():
        predictions = model.predict(X)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            probabilities = (
                scores - scores.min()
            ) / (scores.max() - scores.min() + 1e-10)
        else:
            probabilities = predictions

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y, predictions),
            "AUC": roc_auc_score(y, probabilities),
            "Precision": precision_score(y, predictions, zero_division=0),
            "Recall": recall_score(y, predictions, zero_division=0),
            "F1 Score": f1_score(y, predictions, zero_division=0),
            "MCC": matthews_corrcoef(y, predictions)
        })

    comparison = pd.DataFrame(results)

    st.subheader("Comparison of All Five Models")
    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "MCC": "{:.4f}"
        }),
        use_container_width=True
    )

    selected_model_object = models[selected_model]
    selected_predictions = selected_model_object.predict(X)

    selected_metrics = comparison[
        comparison["Model"] == selected_model
    ].iloc[0]

    st.subheader(f"{selected_model} - Evaluation Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{selected_metrics['Accuracy']:.4f}")
    col2.metric("AUC", f"{selected_metrics['AUC']:.4f}")
    col3.metric("Precision", f"{selected_metrics['Precision']:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{selected_metrics['Recall']:.4f}")
    col5.metric("F1 Score", f"{selected_metrics['F1 Score']:.4f}")
    col6.metric("MCC", f"{selected_metrics['MCC']:.4f}")

    st.subheader("Confusion Matrix")
    matrix = confusion_matrix(y, selected_predictions)
    st.dataframe(
        pd.DataFrame(
            matrix,
            index=["Actual Benign", "Actual Malignant"],
            columns=["Predicted Benign", "Predicted Malignant"]
        )
    )

    st.subheader("Classification Report")
    report = classification_report(
        y,
        selected_predictions,
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Upload your labelled test CSV file to evaluate the models.")