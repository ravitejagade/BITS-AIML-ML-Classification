# Breast Cancer Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to build and evaluate five supervised machine learning classification models to classify breast cancer tumours as Benign (B) or Malignant (M). The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository
- **Number of instances:** 569
- **Number of input features:** 30
- **Target column:** Diagnosis
- **Target classes:** Benign (B) and Malignant (M)
- **Training records:** 455
- **Testing records:** 114
- **Split:** 80% training and 20% testing

The dataset contains numeric measurements computed from digitized images of breast mass samples. The target diagnosis is used to classify each sample as benign or malignant.

## 3. GitHub Repository

https://github.com/ravitejagade/BITS-AIML-ML-Classification

## 4. Model Comparison

All five models were trained on the same training dataset and evaluated on the same held-out test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9123 | 0.9444 | 0.9444 | 0.8095 | 0.8718 | 0.8113 |
| K-Nearest Neighbors (KNN) | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine (SVM) | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

## 5. Observations

### Logistic Regression
Logistic Regression achieved 96.49% accuracy and an AUC of 0.9960. It provided strong precision and recall, showing balanced classification performance.

### Decision Tree
Decision Tree achieved 91.23% accuracy. Its recall of 0.8095 was the lowest among the five models, indicating that it missed more malignant cases than the other models.

### K-Nearest Neighbors (KNN)
KNN achieved 95.61% accuracy and an AUC of 0.9823. It performed strongly across the evaluation metrics, although its recall was lower than Logistic Regression and SVM.

### Gaussian Naive Bayes
Gaussian Naive Bayes achieved 92.11% accuracy and an AUC of 0.9891. Its performance was reasonable, but its precision, recall and F1 score were below those of Logistic Regression, KNN and SVM.

### Support Vector Machine (SVM)
SVM achieved the highest accuracy (97.37%), precision (1.0000), F1 score (0.9630) and MCC (0.9442). It also achieved a high AUC of 0.9947 and recall of 0.9286.

### Overall Best Model
Based on the held-out test results, Support Vector Machine (SVM) was the strongest overall model. It achieved the highest accuracy, precision, F1 score and MCC. Logistic Regression achieved the highest AUC (0.9960), so it also demonstrated excellent class-separation ability.

## 6. Streamlit Application

The Streamlit application provides:
- CSV upload for labelled test data.
- A dropdown to select one of the five trained models.
- A comparison table showing all six evaluation metrics for all five models.
- Evaluation metrics for the selected model.
- Confusion matrix and classification report.

The application loads the saved model files and evaluates the uploaded test dataset. It does not retrain the models during app usage.

## 7. Project Files

- `train_model.py` – trains and evaluates the five machine learning models.
- `app.py` – Streamlit application.
- `requirements.txt` – Python package dependencies.
- `test_data.csv` – held-out test dataset used for evaluation.
- `dataset.csv` – dataset used for the project.
- `metrics.csv` – model comparison metrics.
- `feature_columns.json` – list of input feature names.
- `models/` – contains the five saved trained model files.

## 8. How to Run the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

Upload a labelled test CSV file containing the required 30 feature columns and a `diagnosis` column with `B` or `M` values.
