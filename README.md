# Credit Scoring Model - Machine Learning Project

## Objective
Predict whether a customer is **Good Credit** or **Bad Credit** using financial and credit-history features.

## Algorithms
- Logistic Regression
- Random Forest

## Dataset
`credit_scoring_dataset.csv` is a reproducible **synthetic educational dataset** containing 1,500 customer records. It is used to demonstrate the ML workflow without using real personal financial information.

## Features
Age, Income, Debt, Loan Amount, Employment Years, Credit History Years, Number of Loans, Late Payments, Payment History Score, and Credit Utilization.

## Target
- Good Credit
- Bad Credit

## Workflow
1. Load dataset
2. Separate features and target
3. Train-test split (80:20)
4. Handle missing values and scale numeric features
5. Train Logistic Regression and Random Forest
6. Evaluate Accuracy, Precision, Recall, F1-Score and ROC-AUC
7. Plot ROC curve, confusion matrix and feature importance

## Results from the included dataset
              Model  Accuracy  Precision  Recall  F1-Score  ROC-AUC
Logistic Regression    0.6433     0.4472  0.5851    0.5069   0.6585
      Random Forest    0.6500     0.4267  0.3404    0.3787   0.6297

**Best model by F1-Score:** Logistic Regression

## How to run
```bash
pip install -r requirements.txt
python credit_scoring.py
```

## Academic note
This is an educational demonstration using synthetic data. It should not be used for real lending decisions. Real deployment requires validated real-world data, fairness testing, explainability, security, regulatory compliance, and independent model validation.
