# Handling Imbalanced Data with Weighted Logistic Regression and Propensity Score Matching

## Objective

The objective of this project is to develop a comprehensive system for handling imbalanced datasets, with a focus on peer-to-peer (P2P) money transfer transactions. The system includes implementation of weighted logistic regression, propensity score matching for balancing datasets, and evaluation and visualization of model performance.

## Features

### 1. Weighted Logistic Regression
- Implement logistic regression with class weights to handle imbalanced data.
- Adjust weights based on class distribution to improve model performance.

### 2. Propensity Score Matching
- Implement propensity score matching (PSM) to balance datasets.
- Use matching techniques to create a balanced dataset for more accurate model evaluation.

### 3. Model Evaluation and Visualization
- Evaluate model performance using metrics such as accuracy, precision, recall, F1 score, and AUC-ROC.
- Visualize the results using Matplotlib and Seaborn for better understanding of model performance.

## Technologies

- **Programming Language**: Python
- **Libraries**: scikit-learn, pandas, Matplotlib, Seaborn

## Project Structure

```
handling-imbalanced-data/
│
├── backend/
│   ├── __init__.py
│   ├── weighted_logistic_regression.py
│   ├── propensity_score_matching.py
│   ├── evaluation.py
│
├── data/
│   ├── raw_data.csv
│   ├── processed_data.csv
│
├── scripts/
│   ├── load_data.py
│   ├── run_analysis.py
│
├── tests/
│   ├── __init__.py
│   ├── test_weighted_logistic_regression.py
│   ├── test_propensity_score_matching.py
│   ├── test_evaluation.py
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── model_training.ipynb
│   ├── model_evaluation.ipynb
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/username/handling-imbalanced-data.git
    cd handling-imbalanced-data
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Load and Process Data
- Place your raw dataset in the `data/` directory.
- Run the `load_data.py` script to preprocess the data:
    ```bash
    python scripts/load_data.py
    ```

### 2. Run Analysis
- Execute the `run_analysis.py` script to perform weighted logistic regression and propensity score matching:
    ```bash
    python scripts/run_analysis.py
    ```

### 3. Jupyter Notebooks
- Use the provided Jupyter notebooks in the `notebooks/` directory for exploratory data analysis, model training, and evaluation:
    - `exploratory_data_analysis.ipynb`
    - `model_training.ipynb`
    - `model_evaluation.ipynb`

## Backend Modules

### `backend/__init__.py`
- Initialization file for the backend package.

### `backend/weighted_logistic_regression.py`
- Implements logistic regression with class weights.
- Functions to train and evaluate the weighted logistic regression model.

### `backend/propensity_score_matching.py`
- Implements propensity score matching techniques.
- Functions to balance the dataset using propensity scores.

### `backend/evaluation.py`
- Functions to evaluate model performance using various metrics.
- Functions to visualize model performance using Matplotlib and Seaborn.

## Scripts

### `scripts/load_data.py`
- Script to load and preprocess raw data.

### `scripts/run_analysis.py`
- Script to perform weighted logistic regression and propensity score matching.
- Evaluates and visualizes model performance.

## Tests

### `tests/__init__.py`
- Initialization file for the tests package.

### `tests/test_weighted_logistic_regression.py`
- Unit tests for the weighted logistic regression module.

### `tests/test_propensity_score_matching.py`
- Unit tests for the propensity score matching module.

### `tests/test_evaluation.py`
- Unit tests for the evaluation module.

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- Matplotlib
- Seaborn

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- This project utilizes techniques from various sources and research papers on handling imbalanced data and propensity score matching.
- Special thanks to the contributors and maintainers of the libraries used in this project.
