import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classifier Evaluation", layout="wide")

st.title("Machine Learning Classification Evaluation App")
st.markdown("Upload test CSV data, pick a trained classifier, and inspect evaluation metrics and visualization.")

# Sidebar Feature 1: Dataset Upload Option
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

# Sidebar Feature 2: Model Selection Dropdown
st.sidebar.header("2. Choose Model")
model_options = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}
selected_model_name = st.sidebar.selectbox("Select Classification Model", list(model_options.keys()))

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("### Test Dataset Preview", test_df.head())

    if "target" not in test_df.columns:
        st.error("Uploaded CSV must contain a 'target' column!")
    else:
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]

        # Load Scaler & Model
        try:
            scaler = joblib.load("model/scaler.pkl")
            model = joblib.load(model_options[selected_model_name])

            # Apply scaling for distance/gradient-based models
            if selected_model_name in ["Logistic Regression", "kNN", "Naive Bayes"]:
                X_eval = scaler.transform(X_test)
            else:
                X_eval = X_test

            y_pred = model.predict(X_eval)
            y_prob = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred

            # Compute Metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            # Sidebar Feature 3: Display of Evaluation Metrics
            st.header(f"Performance Metrics: {selected_model_name}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            st.divider()

            # Sidebar Feature 4: Confusion Matrix & Classification Report
            col_cm, col_rep = st.columns(2)

            with col_cm:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                plt.xlabel("Predicted Class")
                plt.ylabel("True Class")
                st.pyplot(fig)
                plt.close(fig)  # Memory cleanup for Streamlit

            with col_rep:
                st.subheader("Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report_dict).transpose())

        except Exception as e:
            st.error(f"Error loading model or evaluating data: {e}")
else:
    st.info("Please upload `test_data.csv` in the sidebar to view evaluations.")