# backend/propensity_score_matching.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

class PropensityScoreMatching:
    def __init__(self):
        """
        Initialize the PropensityScoreMatching class.
        """
        self.model = LogisticRegression()
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """
        Fit the logistic regression model to estimate propensity scores.

        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Binary treatment indicator.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.propensity_scores = self.model.predict_proba(X_scaled)[:, 1]

    def match(self, X, y, caliper=0.05):
        """
        Perform propensity score matching.

        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Binary treatment indicator.
        caliper (float): Maximum allowable difference in propensity scores for matching.

        Returns:
        pd.DataFrame: DataFrame containing matched pairs.
        """
        X_scaled = self.scaler.transform(X)
        propensity_scores = self.model.predict_proba(X_scaled)[:, 1]

        # Create a DataFrame to store the data
        data = pd.DataFrame({'X': list(X), 'y': y, 'propensity_score': propensity_scores})

        # Split the data into treated and control groups
        treated = data[data['y'] == 1]
        control = data[data['y'] == 0]

        matched_pairs = []

        for i, treated_row in treated.iterrows():
            treated_score = treated_row['propensity_score']

            # Calculate the distance between the treated unit and all control units
            distances = np.abs(control['propensity_score'] - treated_score)

            # Find the closest control unit within the caliper
            min_distance = distances.min()
            if min_distance <= caliper:
                closest_control_index = distances.idxmin()
                matched_control_row = control.loc[closest_control_index]

                # Append the matched pair to the list
                matched_pairs.append((treated_row, matched_control_row))

                # Remove the matched control unit from the pool
                control = control.drop(index=closest_control_index)

        # Create a DataFrame of matched pairs
        matched_pairs_df = pd.DataFrame({
            'treated': [pair[0]['X'] for pair in matched_pairs],
            'control': [pair[1]['X'] for pair in matched_pairs]
        })

        return matched_pairs_df

    def evaluate_balance(self, X, y):
        """
        Evaluate the balance of covariates after matching.

        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Binary treatment indicator.

        Returns:
        dict: Dictionary containing the standardized mean differences for each covariate.
        """
        X_scaled = self.scaler.transform(X)
        treated = X_scaled[y == 1]
        control = X_scaled[y == 0]

        mean_diff = np.mean(treated, axis=0) - np.mean(control, axis=0)
        pooled_std = np.sqrt((np.var(treated, axis=0) + np.var(control, axis=0)) / 2)

        standardized_mean_diff = mean_diff / pooled_std
        balance_metrics = {f'covariate_{i}': standardized_mean_diff[i] for i in range(len(standardized_mean_diff))}

        return balance_metrics
