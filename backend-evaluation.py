# backend/evaluation.py

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluation:
    def __init__(self):
        """
        Initialize the Evaluation class.
        """
        pass

    def evaluate_classification(self, y_true, y_pred, y_proba):
        """
        Evaluate classification model performance.

        Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_proba (array-like): Predicted probabilities.

        Returns:
        dict: Dictionary containing evaluation metrics.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }
        
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        """
        Plot the confusion matrix.

        Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        normalize (bool): Whether to normalize the confusion matrix.
        title (str): Title of the plot.
        cmap (matplotlib colormap): Colormap for the plot.
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, linewidths=0.5)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def plot_roc_curve(self, y_true, y_proba):
        """
        Plot the ROC curve.

        Parameters:
        y_true (array-like): True labels.
        y_proba (array-like): Predicted probabilities.
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

    def summarize_evaluation(self, y_true, y_pred, y_proba):
        """
        Summarize and print evaluation metrics and plots.

        Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_proba (array-like): Predicted probabilities.
        """
        metrics = self.evaluate_classification(y_true, y_pred, y_proba)
        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curve(y_true, y_proba)
