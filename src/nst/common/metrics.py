import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

def compute_metrics_from_logits(logits, labels):
    probs = softmax_np(logits)
    preds = probs.argmax(-1)
    out = {
        "accuracy": (preds == labels).mean().item(),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "balanced_acc": balanced_accuracy_score(labels, preds),
    }
    try:
        out["auroc"] = roc_auc_score(labels, probs[:,1])
    except Exception:
        out["auroc"] = float("nan")
    return out

def softmax_np(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
