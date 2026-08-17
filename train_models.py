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

ssl._create_default_https_context = ssl._create_unverified_context

ARTIFACT_DIR = "model"
SEED = 101

def load_and_preprocess_wine_data():
    url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    red_df = pd.read_csv(url_red, sep=";")
    white_df = pd.read_csv(url_white, sep=";")

    red_df['is_white_wine'] = 0
    white_df['is_white_wine'] = 1

    combined_df = pd.concat([red_df, white_df], ignore_index=True)
    combined_df['target'] = (combined_df['quality'] >= 6).astype(int)
    combined_df = combined_df.drop(columns=['quality'])
    return combined_df

def run_training_pipeline():
    wine_df = load_and_preprocess_wine_data()

    X = wine_df.drop(columns=['target'])
    y = wine_df['target']

    # Train-Test Split (30% test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )

    # Export test dataset for Streamlit evaluation
    eval_dataset = X_test.copy()
    eval_dataset['target'] = y_test
    eval_dataset.to_csv("test_data.csv", index=False)

    # Feature Scaling Pipeline
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(feature_scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))

    # Custom model parameter configurations
    classifiers = {
        "Logistic Regression": LogisticRegression(C=0.85, max_iter=1000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=9, min_samples_split=6, random_state=SEED),
        "kNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Naive Bayes": GaussianNB(var_smoothing=1e-8),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=4, random_state=SEED)
    }

    metrics_log = []
    scale_required = ["Logistic Regression", "kNN", "Naive Bayes"]

    for name, clf in classifiers.items():
        X_tr = X_train_scaled if name in scale_required else X_train
        X_te = X_test_scaled if name in scale_required else X_test

        clf.fit(X_tr, y_train)

        file_slug = name.lower().replace(" ", "_")
        joblib.dump(clf, os.path.join(ARTIFACT_DIR, f"{file_slug}.pkl"))

        preds = clf.predict(X_te)
        probs = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else preds

        metrics_log.append({
            "ML Model Name": name,
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "AUC": round(roc_auc_score(y_test, probs), 4),
            "Precision": round(precision_score(y_test, preds), 4),
            "Recall": round(recall_score(y_test, preds), 4),
            "F1": round(f1_score(y_test, preds), 4),
            "MCC": round(matthews_corrcoef(y_test, preds), 4)
        })

    summary_df = pd.DataFrame(metrics_log)
    print("\n--- Model Evaluation Summary ---")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_training_pipeline()