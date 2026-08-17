import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Wine Quality Classification Portal", layout="wide")

st.title("🍷 Wine Quality Classification & Evaluation Portal")
st.markdown("Upload test data, select a model architecture, and evaluate metrics.")

# 1. Dataset Upload Option
st.sidebar.header("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Evaluation File (test_data.csv)", type=["csv"])

# 2. Model Selection Dropdown
model_options = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}
selected_model = st.sidebar.selectbox("Choose Trained Architecture", list(model_options.keys()))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Dataset Inspection
    st.subheader("📋 Dataset Inspection")
    st.dataframe(df.head(4), use_container_width=True)

    # Separate features and target
    target_col = "target" if "target" in df.columns else df.columns[-1]
    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]

    try:
        model = joblib.load(model_options[selected_model])
        scaler = joblib.load("model/scaler.pkl")

        # Apply scaling to relevant algorithms
        if selected_model in ["Logistic Regression", "kNN", "Naive Bayes"]:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test

        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # 3. Display of Evaluation Metrics
        st.subheader(f"📊 Performance Overview: `{selected_model}`")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("AUC Score", f"{roc_auc_score(y_test, y_prob):.4f}")
        col3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        col5.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
        col6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")

        st.divider()

        # 4. Confusion Matrix and Classification Summary
        col_cm, col_rep = st.columns(2)

        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax, cbar=False)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)

        with col_rep:
            st.subheader("Classification Summary")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error evaluating model: {e}")
else:
    st.info("Please upload `test_data.csv` in the Control Panel to view evaluations.")