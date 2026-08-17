import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Wine Quality Classifier Portal",
    page_icon="🍷",
    layout="wide"
)

MODEL_REGISTRY = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

st.title("🍷 Chemical Classification & Evaluation Portal")
st.markdown("Upload test data, select a model architecture, and customize prediction thresholds.")

# Sidebar Configuration Controls
st.sidebar.title("⚙️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Evaluation File (test_data.csv)", type=["csv"])
selected_model = st.sidebar.selectbox("Choose Trained Architecture", list(MODEL_REGISTRY.keys()))

st.sidebar.markdown("---")
decision_threshold = st.sidebar.slider(
    "Classification Probability Threshold",
    min_value=0.1, max_value=0.9, value=0.5, step=0.05
)

if uploaded_file is not None:
    eval_df = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Inspection")
    st.dataframe(eval_df.head(4), use_container_width=True)

    if "target" not in eval_df.columns:
        st.error("Missing required target column: 'target' not found in uploaded CSV.")
    else:
        features_df = eval_df.drop(columns=["target"])
        ground_truth = eval_df["target"]

        try:
            fitted_scaler = joblib.load("model/scaler.pkl")
            active_model = joblib.load(MODEL_REGISTRY[selected_model])

            # Apply scaling for distance/gradient-sensitive estimators
            if selected_model in ["Logistic Regression", "kNN", "Naive Bayes"]:
                processed_features = fitted_scaler.transform(features_df)
            else:
                processed_features = features_df

            if hasattr(active_model, "predict_proba"):
                predicted_probabilities = active_model.predict_proba(processed_features)[:, 1]
                predicted_labels = (predicted_probabilities >= decision_threshold).astype(int)
            else:
                predicted_probabilities = active_model.predict(processed_features)
                predicted_labels = predicted_probabilities

            # Compute Evaluation Metrics
            acc_val = accuracy_score(ground_truth, predicted_labels)
            auc_val = roc_auc_score(ground_truth, predicted_probabilities)
            prec_val = precision_score(ground_truth, predicted_labels)
            rec_val = recall_score(ground_truth, predicted_labels)
            f1_val = f1_score(ground_truth, predicted_labels)
            mcc_val = matthews_corrcoef(ground_truth, predicted_labels)

            st.markdown(f"### 📊 Performance Overview: `{selected_model}`")

            # 2x3 Metric Card Layout
            r1_c1, r1_c2, r1_c3 = st.columns(3)
            r1_c1.metric("Accuracy", f"{acc_val:.4f}")
            r1_c2.metric("AUC Score", f"{auc_val:.4f}")
            r1_c3.metric("Precision", f"{prec_val:.4f}")

            r2_c1, r2_c2, r2_c3 = st.columns(3)
            r2_c1.metric("Recall", f"{rec_val:.4f}")
            r2_c2.metric("F1 Score", f"{f1_val:.4f}")
            r2_c3.metric("MCC Score", f"{mcc_val:.4f}")

            st.markdown("---")

            # Visualizations Section
            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                st.subheader("Confusion Matrix")
                matrix_data = confusion_matrix(ground_truth, predicted_labels)
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                sns.heatmap(matrix_data, annot=True, fmt="d", cmap="YlGnBu", ax=ax_cm, cbar=False)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            with viz_col2:
                st.subheader("Classification Summary")
                report_dict = classification_report(ground_truth, predicted_labels, output_dict=True)
                st.dataframe(pd.DataFrame(report_dict).transpose().style.highlight_max(axis=0))

            # Feature Importance Add-on for Tree-Based Estimators
            if hasattr(active_model, "feature_importances_"):
                st.markdown("---")
                st.subheader("🌲 Tree Feature Importances")
                imp_df = pd.DataFrame({
                    "Feature": features_df.columns,
                    "Importance": active_model.feature_importances_
                }).sort_values(by="Importance", ascending=True)

                fig_imp, ax_imp = plt.subplots(figsize=(7, 3.5))
                ax_imp.barh(imp_df["Feature"], imp_df["Importance"], color="#2b5c8f")
                ax_imp.set_xlabel("Relative Importance Score")
                st.pyplot(fig_imp)
                plt.close(fig_imp)

        except Exception as err:
            st.error(f"Execution Error: {err}")
else:
    st.info("👈 Upload `test_data.csv` in the sidebar panel to generate evaluation outputs.")