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
* **Streamlit App URL:** [https://ml-assignment-2-dns68rarf7u8nqlidlbz5e.streamlit.app/](https://ml-assignment-2-dns68rarf7u8nqlidlbz5e.streamlit.app/)

---

## d. Models Used & Evaluation Metrics

### 1. Metric Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7344 | 0.7993 | 0.7729 | 0.8217 | 0.7965 | 0.4166 |
| **Decision Tree** | 0.7303 | 0.7590 | 0.7814 | 0.7966 | 0.7889 | 0.4156 |
| **kNN** | 0.7959 | 0.8771 | 0.8215 | 0.8655 | 0.8429 | 0.5537 |
| **Naive Bayes** | 0.6846 | 0.7412 | 0.7424 | 0.7682 | 0.7551 | 0.3130 |
| **Random Forest (Ensemble)** | 0.7949 | 0.8773 | 0.8218 | 0.8630 | 0.8419 | 0.5518 |

---

### 2. Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Serves as a reliable linear baseline, reaching 73.44% accuracy and 0.7993 AUC score. Feature scaling via `StandardScaler` ensured optimal gradient convergence across continuous chemical attributes. |
| **Decision Tree** | Achieves 73.03% accuracy and 0.7814 precision, capturing non-linear feature splits across chemical attributes but constrained by standard single-tree variance. |
| **kNN** | Demonstrates outstanding performance with 79.59% accuracy and 0.8429 F1-score. Distance-based spatial clustering effectively groups similar wine composition profiles following standardization. |
| **Naive Bayes** | Obtains 68.46% accuracy and 0.3130 MCC score. Performance is constrained by strict feature independence assumptions among correlated chemical variables like sulfur dioxide levels and acidity. |
| **Random Forest (Ensemble)** | Top-tier performer achieving 0.8773 AUC score, 79.49% accuracy, and 0.8419 F1-score. Bagging decision trees successfully mitigates individual tree overfitting and provides robust decision boundaries. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** and **kNN** emerge as joint top performers, with **Random Forest** achieving the highest ROC-AUC score (0.8773) and **kNN** achieving the top accuracy (0.7959). |