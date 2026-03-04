Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>># Corresponding Section: 6.2 Evaluation Metrics Calculation
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import numpy as np
from typing import Dict

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculates all evaluation metrics used in the paper.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (1 = Affected, 0 = Unaffected)
    y_pred : np.ndarray
        Binary predictions from the model
    y_pred_proba : np.ndarray, optional
        Predicted probabilities for AUC calculation

    Returns
    -------
    Dict[str, float]
        Dictionary containing precision, recall, f1, accuracy, and auc
    """
    metrics = {}
    if len(np.unique(y_true)) == 1:
        metrics["precision"] = 1.0 if np.all(y_true == y_pred) else 0.0
        metrics["recall"] = 1.0 if np.all(y_true == y_pred) else 0.0
        metrics["f1"] = 1.0 if np.all(y_true == y_pred) else 0.0
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["auc"] = 0.5
        return metrics

    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    if y_pred_proba is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics["auc"] = 0.5
    else:
        metrics["auc"] = roc_auc_score(y_true, y_pred)
    
    for key in metrics:
        metrics[key] = round(metrics[key], 3)
    
    return metrics
