import os
import ssl
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# Bypass macOS SSL Certificate Verification Error
ssl._create_default_https_context = ssl._create_unverified_context

# 1. Fetch UCI Wine Quality Datasets
url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

df_red = pd.read_csv(url_red, sep=";")
df_white = pd.read_csv(url_white, sep=";")

df_red['is_white_wine'] = 0
df_white['is_white_wine'] = 1

df = pd.concat([df_red, df_white], ignore_index=True)

# Binary Target: 1 if quality >= 6, else 0
df['target'] = (df['quality'] >= 6).astype(int)
df = df.drop(columns=['quality'])

# 2. Train-Test Split (30% test set)
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save test dataset to root directory
test_df = X_test.copy()
test_df['target'] = y_test
test_df.to_csv("test_data.csv", index=False)
print("Saved test_data.csv successfully.")

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# 4. Define EXACT 5 Models from Assignment Table
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# 5. Train & Evaluate
results = []
for name, model in models.items():
    use_scaled = name in ["Logistic Regression", "kNN", "Naive Bayes"]
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled if use_scaled else X_test

    model.fit(X_tr, y_train)

    file_name = f"model/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, file_name)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_pred

    results.append({
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
    })

results_df = pd.DataFrame(results)
print("\nEvaluation Summary (UCI Wine Quality Dataset):")
print(results_df.to_string(index=False))