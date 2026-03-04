Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> """
utils/metrics.py
Evaluation metrics calculation, corresponds to paper Section 6.2
Implements Precision, Recall, F1-Score, AUC for binary classification
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def calculate_classification_metrics(y_true, y_pred, y_score=None):
    """
    Calculate classification metrics for change impact detection
    Args:
        y_true: Ground truth labels (1: affected, 0: unaffected)
        y_pred: Predicted binary labels
        y_score: Predicted scores for AUC calculation (optional)
    Returns:
        metrics: Dict with precision, recall, f1, auc
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle edge case where all labels are the same
    if len(np.unique(y_true)) == 1:
        precision = 1.0 if np.all(y_true == y_pred) else 0.0
        recall = 1.0 if np.all(y_true == y_pred) else 0.0
        f1 = 1.0 if np.all(y_true == y_pred) else 0.0
        auc = 0.5
    else:
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate AUC
        if y_score is not None:
            auc = roc_auc_score(y_true, y_score)
        else:
            # Use y_pred as score if no score is provided
            auc = roc_auc_score(y_true, y_pred)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }