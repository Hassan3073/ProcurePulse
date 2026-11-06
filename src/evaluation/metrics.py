import numpy as np
from sklearn.metrics import (accuracy_score,f1_score,precision_score,recall_score,roc_auc_score)

METRIC_NAMES = ["n","positive_rate","accuracy","precision","recall","f1","f1_macro","roc_auc"]

def compute_metrics(y_true,y_prob,threshold=0.5):
    """Return a dict of metrics for one model on one split."""
    y_true=np.asarray(y_true).astype(int)
    y_prob=np.asarray(y_prob,dtype=float)
    y_pred=(y_prob>=threshold).astype(int)
    if len (set(y_true))==2:
        auc=roc_auc_score(y_true,y_prob)
    else:
        auc=float("nan")

    return{
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(auc),
    }