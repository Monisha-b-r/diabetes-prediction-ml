# 🩺 Diabetes Prediction using Machine Learning

## 📌 Project Overview

This is an end-to-end Machine Learning project that predicts whether a person is diabetic based on medical features.

The model is trained using the Pima Indians Diabetes dataset and deployed using Flask.

---

## 🎯 Objective

To build a binary classification model that predicts diabetes risk using patient health data.

---

## 🧠 Problem Type

Supervised Learning – Classification

Target Variable:
- Outcome (0 = Non-Diabetic, 1 = Diabetic)

---

## 📊 Dataset

Dataset used:
Pima Indians Diabetes Dataset (768 rows, 8 features)

Features include:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Dataset Source:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Note: Dataset is not included in this repository due to licensing.

---

## 🛠️ Data Preprocessing

The following preprocessing steps were applied:

- Replaced invalid zero values with NaN
- Applied mean imputation using SimpleImputer
- Scaled features using StandardScaler
- Train-test split (80% training, 20% testing)

---

## 🤖 Model Used

RandomForestClassifier

Why Random Forest?
- Handles non-linear relationships
- Works well with structured medical data
- Reduces overfitting using ensemble learning

---

## 📏 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision
- Recall

---

## 🏆 Model Performance

- Accuracy: ~75–80%
- Balanced performance on both classes

---

## 🌐 Flask Deployment

The trained model is deployed using Flask.

To run locally:

```bash
pip install -r requirements.txt
python app.py

