# scripts/run_analysis.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from backend.weighted_logistic_regression import WeightedLogisticRegression
from backend.propensity_score_matching import propensity_score_matching
from backend.evaluation import evaluate_model

def run_analysis(X_train_path, X_test_path, y_train_path, y_test_path):
    """
    Run analysis on the dataset using weighted logistic regression and propensity score matching.

    Parameters:
    X_train_path (str): Path to the training features CSV file.
    X_test_path (str): Path to the test features CSV file.
    y_train_path (str): Path to the training labels CSV file.
    y_test_path (str): Path to the test labels CSV file.
    """
    # Load the datasets
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()  # Convert DataFrame to Series
    y_test = pd.read_csv(y_test_path).squeeze()  # Convert DataFrame to Series

    # Apply propensity score matching
    X_train_matched, y_train_matched = propensity_score_matching(X_train, y_train)

    # Train weighted logistic regression model
    model = WeightedLogisticRegression()
    model.fit(X_train_matched, y_train_matched)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Print evaluation metrics
    print("Classification Report:")
    print(report)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Optionally save evaluation results to a file
    with open('evaluation_results.txt', 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nROC AUC Score: {roc_auc:.4f}")

if __name__ == "__main__":
    # Example usage
    X_train_path = 'data/X_train.csv'  # Replace with the actual path to your training features file
    X_test_path = 'data/X_test.csv'  # Replace with the actual path to your test features file
    y_train_path = 'data/y_train.csv'  # Replace with the actual path to your training labels file
    y_test_path = 'data/y_test.csv'  # Replace with the actual path to your test labels file

    run_analysis(X_train_path, X_test_path, y_train_path, y_test_path)
