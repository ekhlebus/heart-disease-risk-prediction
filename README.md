# Heart Disease Risk Prediction

This project is a machine learning capstone project developed as part of the **ML Zoomcamp** program.

The goal of the project is to predict the probability of heart disease in patients based on clinical measurements, and to compare the performance of different classical machine learning models.

---

## 📌 Problem Description

Cardiovascular diseases are among the leading causes of mortality worldwide. Early risk assessment based on routine clinical measurements can support timely medical intervention.

In this project, we build a binary classification model that predicts whether a patient has heart disease using tabular clinical data. The task closely resembles real-world decision-making problems where models output probabilities rather than hard decisions.

---

## 📊 Dataset

The dataset used in this project is the **Heart Disease UCI dataset**, publicly available on Kaggle:

- https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Each row represents a patient, and each column corresponds to a clinical feature collected during medical examination.

### Features include:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression
- Slope of peak exercise ST segment

### Target:
- **Heart disease presence** (1 = disease, 0 = no disease)

---

## 🎯 Project Objective

- Train and evaluate multiple machine learning models to predict heart disease risk
- Compare linear and tree-based models
- Evaluate model performance using classification metrics
- Interpret predicted probabilities in a healthcare context

---

## 🧠 Models Used

The following models were implemented and compared:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**

These models were chosen to mirror real-world applied machine learning workflows and to highlight differences between linear and non-linear approaches.

---

## 📐 Evaluation Metrics

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- ROC-AUC
- Confusion Matrix

ROC-AUC was used as the primary metric due to its robustness to class imbalance and relevance for probability-based predictions.

---

## 🛠 Project Structure

```text
.
├── data/
│   └── heart.csv
├── notebooks/
│   └── exploration.ipynb
├── train.py
├── predict.py
├── model.bin
├── requirements.txt
└── README.md

How to Run the Project
1. Install dependencies
pip install -r requirements.txt

2. Train the model
python train.py

3. Make predictions
python predict.py

📌 Key Takeaways

Logistic regression provides a strong baseline with interpretable coefficients

Tree-based models capture non-linear interactions between clinical features

Random Forest achieved the best overall performance in this dataset

The project demonstrates how classic ML models can be applied to healthcare risk prediction problems

📎 Notes

This project is designed for educational purposes and does not constitute medical advice.
