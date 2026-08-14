# Machine Learning Assignment 2: Classification Model Evaluation & Deployment

## a. Problem Statement
The objective of this assignment is to train, evaluate, and compare multiple machine learning classification algorithms on a tabular dataset containing at least 12 continuous features and 500 instances. The workflow involves proper feature scaling, training five distinct classification models, evaluating their performance across six comprehensive metrics (Accuracy, AUC, Precision, Recall, F1-Score, MCC), and deploying an interactive model evaluation application on Streamlit Community Cloud.

---

## b. Dataset Description
* **Dataset Name:** UCI Wine Quality Dataset (Combined Red & White Wine)
* **Source:** UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/wine+quality)
* **Total Instances:** 6,497 samples (1,950 reserved in `test_data.csv` for Streamlit evaluation)
* **Feature Count:** 12 numerical features (`fixed_acidity`, `volatile_acidity`, `citric_acid`, `residual_sugar`, `chlorides`, `free_sulfur_dioxide`, `total_sulfur_dioxide`, `density`, `pH`, `sulphates`, `alcohol`, `is_white_wine`)
* **Target Variable:** Binary Classification Target (`1` for good quality score $\ge 6$, `0` for average/poor score $< 6$)
* **Preprocessing:** `StandardScaler` feature scaling fitted on training set and applied to distance and gradient-sensitive models.

---

## c. Repository & Deployment Links
* **GitHub Repository:** [https://github.com/payalmariya/ml-assignment-2](https://github.com/payalmariya/ml-assignment-2)
* **Streamlit App URL:** [https://YOUR_APP_NAME.streamlit.app](https://YOUR_APP_NAME.streamlit.app)


---

## d. Models Used & Evaluation Metrics

### 1. Metric Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7318 | 0.7847 | 0.7784 | 0.8173 | 0.7974 | 0.4026 |
| **Decision Tree** | 0.7677 | 0.7463 | 0.8203 | 0.8197 | 0.8200 | 0.4925 |
| **kNN** | 0.7497 | 0.8143 | 0.8038 | 0.8102 | 0.8070 | 0.4513 |
| **Naive Bayes** | 0.6779 | 0.7433 | 0.7502 | 0.7514 | 0.7508 | 0.2957 |
| **Random Forest (Ensemble)** | 0.8221 | 0.8948 | 0.8613 | 0.8634 | 0.8624 | 0.6107 |

---

### 2. Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Serves as a reliable linear baseline, reaching 73.18% accuracy and 0.7847 AUC score. Feature scaling via `StandardScaler` ensured optimal gradient convergence across continuous chemical attributes. |
| **Decision Tree** | Outperforms the linear baseline with 76.77% accuracy and strong precision (82.03%), effectively capturing non-linear feature split boundaries across wine attributes. |
| **kNN** | Achieves 74.97% accuracy and 0.8143 AUC. Distance-based spatial clustering effectively groups similar wine composition profiles following standardization. |
| **Naive Bayes** | Obtains 67.79% accuracy and 0.2957 MCC. Performance is constrained by independence assumptions among correlated chemical features like sulfur dioxide levels and acidity. |
| **Random Forest (Ensemble)** | Highest overall performer across all metrics, achieving 82.21% accuracy, 0.8948 AUC, 0.8624 F1-score, and 0.6107 MCC. Bagging decision trees successfully reduces individual decision tree variance. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** is the overall winner on this dataset, leading across every single evaluation metric. |