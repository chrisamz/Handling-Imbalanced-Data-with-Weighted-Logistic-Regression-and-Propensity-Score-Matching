# backend/weighted_logistic_regression.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

class WeightedLogisticRegression:
    def __init__(self, class_weight='balanced'):
        """
        Initialize the WeightedLogisticRegression model.

        Parameters:
        class_weight (str or dict): Weights associated with classes. If 'balanced', uses the values of y to automatically adjust weights inversely proportional to class frequencies.
        """
        self.class_weight = class_weight
        self.model = LogisticRegression(class_weight=self.class_weight)

    def fit(self, X, y):
        """
        Fit the logistic regression model according to the given training data.

        Parameters:
        X (array-like): Training vector, where n_samples is the number of samples and n_features is the number of features.
        y (array-like): Target vector relative to X.
        """
        # Compute sample weights
        sample_weight = compute_sample_weight(class_weight=self.class_weight, y=y)
        
        # Fit the model with sample weights
        self.model.fit(X, y, sample_weight=sample_weight)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X (array-like): Samples.
        
        Returns:
        array: Predicted class label per sample.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Probability estimates for samples in X.

        Parameters:
        X (array-like): Samples.
        
        Returns:
        array: Returns the probability of the sample for each class in the model, where classes are ordered as they are in self.classes_.
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model using the provided test data and labels.

        Parameters:
        X (array-like): Test samples.
        y (array-like): True labels for X.

        Returns:
        float: Mean accuracy of the model on the given test data and labels.
        """
        return self.model.score(X, y)
