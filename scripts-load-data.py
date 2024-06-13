# scripts/load_data.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path, test_size=0.2, random_state=42):
    """
    Load dataset from a CSV file and split it into training and test sets.

    Parameters:
    file_path (str): Path to the CSV file.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random state for reproducibility.

    Returns:
    X_train (DataFrame): Training features.
    X_test (DataFrame): Test features.
    y_train (Series): Training labels.
    y_test (Series): Test labels.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Split the dataset into features and target variable
    X = data.drop(columns=['target'])  # Replace 'target' with the name of your target column
    y = data['target']  # Replace 'target' with the name of your target column

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    file_path = 'path/to/your/data.csv'  # Replace with the actual path to your data file
    X_train, X_test, y_train, y_test = load_data(file_path)

    # Save the datasets to CSV files (optional)
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

    print("Data loaded and split into training and test sets.")
