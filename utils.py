# utils.py
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os

def compute_metrics(y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels)
    macro_f1 = float(np.mean(f1))
    return {
        "accuracy": float(acc),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "macro_f1": macro_f1
    }

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
