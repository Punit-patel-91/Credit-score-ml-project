"""
Credit Scoring Model - Machine Learning Project
Dataset: credit_scoring_dataset.csv

Run:
    python credit_scoring.py

The script trains Logistic Regression and Random Forest models,
compares Accuracy, Precision, Recall, F1-Score and ROC-AUC,
and saves evaluation plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)

df = pd.read_csv("credit_scoring_dataset.csv")

X = df.drop(columns=["Credit_Status"])
y = (df["Credit_Status"] == "Bad Credit").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

features = list(X.columns)

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), features)
])

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=250, max_depth=9,
            min_samples_leaf=3, random_state=42,
            class_weight="balanced"
        ))
    ])
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("\\n" + "=" * 55)
    print(name)
    print("=" * 55)
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, pred, zero_division=0):.4f}")
    print(f"F1-Score : {f1_score(y_test, pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, proba):.4f}")

    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Good Credit", "Bad Credit"]
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()

print("\\nProject completed successfully.")
