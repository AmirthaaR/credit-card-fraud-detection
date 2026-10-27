# Credit Card Fraud Detection using Machine Learning

## Project Overview
This project uses machine learning to identify potentially fraudulent credit card transactions. Since fraudulent transactions represent a very small proportion of the dataset, the project focuses on handling class imbalance and evaluating fraud detection using precision, recall, and F1-score.

## Objectives
- Analyze credit card transaction data.
- Understand the class imbalance between genuine and fraudulent transactions.
- Train and evaluate machine learning models.
- Compare Logistic Regression and Random Forest.
- Evaluate fraud detection performance using different classification thresholds.

## Dataset
The project uses the Credit Card Fraud Detection dataset from Kaggle.

**Dataset:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains 284,807 transactions and 31 columns, including:
- `Time`: Seconds elapsed between a transaction and the first transaction in the dataset.
- `Amount`: Transaction amount.
- `V1`–`V28`: Anonymized numerical features.
- `Class`: Target variable (`0` = genuine transaction, `1` = fraudulent transaction).

Only 492 transactions are fraudulent, making this a highly imbalanced classification problem.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Google Colab

## Project Workflow
1. Load and inspect the dataset.
2. Perform exploratory data analysis (EDA).
3. Examine the distribution of genuine and fraudulent transactions.
4. Split the data into training, validation, and test sets.
5. Scale the `Time` and `Amount` features using `StandardScaler`.
6. Train Logistic Regression and Random Forest classifiers.
7. Address class imbalance using class-weighted learning.
8. Compare model performance and classification thresholds.
9. Evaluate the final model using a held-out test set.

## Model Evaluation
The Random Forest classifier was evaluated at a classification threshold of `0.3`.

| Metric | Fraud Class (1) |
|---|---:|
| Precision | 87.65% |
| Recall | 72.45% |
| F1-score | 0.7933 |

The model correctly identified 71 of the 98 fraudulent transactions in the held-out test set, with 10 genuine transactions incorrectly flagged as fraudulent.

These results illustrate the trade-off between detecting more fraudulent transactions and reducing false alarms. Performance may vary with different data splits and threshold choices.

## How to Run
1. Open the Jupyter notebook in this repository.
2. Open it in [Google Colab](https://colab.research.google.com/).
3. Download the dataset from Kaggle using the link above.
4. Upload the CSV file to Colab or update the notebook's data-loading path.
5. Run the notebook cells in order to reproduce the analysis and model evaluation.

## Key Learnings
- Handling highly imbalanced datasets.
- Performing exploratory data analysis.
- Preventing data leakage during feature scaling.
- Training and comparing classification models.
- Evaluating precision, recall, F1-score, and confusion matrices.
- Understanding classification threshold trade-offs.

## Future Improvements
- Tune the classification threshold using validation data and evaluate it with a completely untouched test set.
- Explore additional imbalance-handling techniques.
- Compare Random Forest with other classification algorithms.
- Develop a simple web application for testing transactions.

## Author
**Amirthaa R.**

B.Tech Computer Science and Engineering

This project was developed as part of hands-on machine learning practice.
