"""Evaluation metrics for predictive maintenance models"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import numpy as np

class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for predictive models.
    """
    
    @staticmethod
    def compute_metrics(y_true, y_pred, y_proba=None):
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
            metrics['roc_curve'] = roc_curve(y_true, y_proba)
        
        return metrics
    
    @staticmethod
    def get_classification_report(y_true, y_pred):
        """
        Get detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        return classification_report(y_true, y_pred)
