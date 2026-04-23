"""
metrics.py - Classification Metrics for Imbalanced Grasp Data

This module computes all classification metrics needed for evaluating
grasp quality prediction, with special attention to class imbalance.

WHY ACCURACY ALONE IS MISLEADING:
Our dataset has 63.7% positive (good) grasps and 36.3% negative (bad) grasps.
A model that ALWAYS predicts "positive" would score 63.7% accuracy — seemingly
decent but completely useless (it never identifies bad grasps).

METRIC CATEGORIES:

1. IMBALANCE-ROBUST (Primary — these tell the real story):
   - Balanced Accuracy: average of per-class recall. A majority-class predictor
     gets exactly 50%, not 63.7%.
   - Macro F1: unweighted average F1 across both classes.
   - Specificity: TN/(TN+FP) — how well we catch bad grasps (the minority class).
   - PR-AUC: area under Precision-Recall curve — more honest than ROC-AUC
     for imbalanced data.

2. STANDARD (Report but interpret carefully):
   - Accuracy: (TP+TN)/total — dominated by the majority class.
   - Weighted F1: weighted by class support — majority class inflates it.
   - ROC-AUC: can look deceptively good on imbalanced data.

The key question to ask: "Can the model identify BOTH good and bad grasps?"
If per-class recall for negatives is very low, the model has collapsed to
predicting the majority class — high accuracy but useless in practice.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


def classification_report_full(y_true, y_pred, y_prob=None, class_names=None):
    """
    Compute ALL classification metrics for grasp quality prediction.

    This function is the single entry point for all metrics. It returns
    a comprehensive dictionary organized by category.

    Parameters:
    -----------
    y_true : list or numpy.ndarray
        True labels (0 = negative/bad, 1 = positive/good).

    y_pred : list or numpy.ndarray
        Predicted labels (0 or 1).

    y_prob : numpy.ndarray or None, shape (N, 2)
        Predicted probabilities from softmax. Column 0 = P(bad), Column 1 = P(good).
        Required for ROC curves, PR curves, and AUC scores.

    class_names : list of str or None
        Names for each class. Default: ['negative', 'positive'].

    Returns:
    --------
    metrics : dict
        Comprehensive metrics dictionary with the following structure:
        {
            'accuracy': float,
            'balanced_accuracy': float,
            'precision_macro': float,
            'precision_weighted': float,
            'recall_macro': float,
            'recall_weighted': float,
            'f1_macro': float,
            'f1_weighted': float,
            'specificity': float,
            'confusion_matrix': numpy.ndarray,
            'confusion_matrix_normalized': numpy.ndarray,
            'per_class': {
                'precision': list,
                'recall': list,
                'f1': list,
                'support': list,
            },
            'auc_roc': float or None,
            'average_precision': float or None,
            'sklearn_report': str,
        }
    """
    if class_names is None:
        class_names = ['negative', 'positive']

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ---- BASIC METRICS ----
    metrics = {}

    # Accuracy: (TP + TN) / total
    # CAVEAT: Majority baseline is 63.7% — models must beat this convincingly
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Balanced Accuracy: average of per-class recall
    # = (recall_class0 + recall_class1) / 2
    # A majority predictor gets exactly 50% here, making it much more informative
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # ---- PRECISION, RECALL, F1 (multiple averaging strategies) ----

    # Macro: compute metric for each class, then take unweighted average
    # Treats both classes equally regardless of their size
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Weighted: compute metric for each class, then take weighted average by support
    # Gives more influence to the majority class (less useful for imbalanced data)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # ---- SPECIFICITY ----
    # Specificity = TN / (TN + FP)
    # "Of all actual NEGATIVE (bad) grasps, how many did we correctly identify?"
    # This is critical for robotics safety — missing a bad grasp means the robot
    # tries a grasp that will fail, potentially dropping or damaging objects.
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        metrics['specificity'] = 0.0
        metrics['sensitivity'] = 0.0

    # ---- CONFUSION MATRIX ----
    metrics['confusion_matrix'] = cm

    # Normalized confusion matrix (rows sum to 1)
    # Shows per-class recall directly: each row shows what fraction of that
    # class was predicted as each class
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    metrics['confusion_matrix_normalized'] = cm_normalized

    # ---- PER-CLASS METRICS ----
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics['per_class'] = {
        'precision': per_class_precision.tolist(),
        'recall': per_class_recall.tolist(),
        'f1': per_class_f1.tolist(),
        'support': [int(np.sum(y_true == c)) for c in range(len(class_names))],
    }

    # ---- PROBABILITY-BASED METRICS (if probabilities provided) ----
    if y_prob is not None:
        y_prob = np.array(y_prob)

        # ROC-AUC: Area Under the ROC Curve
        # Uses the probability of the positive class (column 1)
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
        except ValueError:
            metrics['auc_roc'] = None

        # Average Precision (AP) / PR-AUC
        # Area under the Precision-Recall curve
        # More informative than ROC-AUC for imbalanced data because:
        # - ROC uses FPR which is diluted by large TN count
        # - PR focuses on the positive predictions, which is where imbalance hurts
        try:
            metrics['average_precision'] = average_precision_score(y_true, y_prob[:, 1])
        except ValueError:
            metrics['average_precision'] = None

        # ROC curve data points (for plotting)
        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob[:, 1])
            metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds}
        except ValueError:
            metrics['roc_curve'] = None

        # Precision-Recall curve data points (for plotting)
        try:
            pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_prob[:, 1])
            metrics['pr_curve'] = {
                'precision': pr_precision,
                'recall': pr_recall,
                'thresholds': pr_thresholds,
            }
        except ValueError:
            metrics['pr_curve'] = None
    else:
        metrics['auc_roc'] = None
        metrics['average_precision'] = None
        metrics['roc_curve'] = None
        metrics['pr_curve'] = None

    # ---- SKLEARN TEXT REPORT ----
    metrics['sklearn_report'] = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    return metrics


def compute_majority_baseline(y_true):
    """
    Compute what accuracy you'd get by always predicting the majority class.

    This establishes the FLOOR that any useful model must beat. If your model's
    accuracy is close to this baseline, it hasn't actually learned to classify —
    it's just predicting the most common class for everything.

    For our dataset:
    - Always predict "positive" → 63.7% accuracy, 50.0% balanced accuracy
    - This means 63.7% accuracy is the MINIMUM, not something to celebrate

    Parameters:
    -----------
    y_true : list or numpy.ndarray
        True labels.

    Returns:
    --------
    baseline : dict
        {
            'majority_class': int,
            'majority_class_name': str,
            'majority_accuracy': float,
            'majority_balanced_accuracy': float,
            'class_distribution': dict,
            'message': str,
        }
    """
    y_true = np.array(y_true)
    unique, counts = np.unique(y_true, return_counts=True)

    majority_idx = np.argmax(counts)
    majority_class = unique[majority_idx]
    majority_count = counts[majority_idx]
    total = len(y_true)
    majority_accuracy = majority_count / total

    class_names = ['negative', 'positive']

    baseline = {
        'majority_class': int(majority_class),
        'majority_class_name': class_names[int(majority_class)],
        'majority_accuracy': majority_accuracy,
        'majority_balanced_accuracy': 0.5,  # Always 50% for any majority predictor
        'class_distribution': {
            class_names[int(c)]: int(cnt)
            for c, cnt in zip(unique, counts)
        },
        'message': (
            f"Always predicting '{class_names[int(majority_class)]}' gives "
            f"{majority_accuracy:.1%} accuracy but only 50.0% balanced accuracy. "
            f"Any useful model must beat BOTH baselines."
        ),
    }

    return baseline


def print_metrics_summary(metrics, model_name="Model"):
    """
    Print a formatted summary of all classification metrics.

    Organizes metrics into Primary (imbalance-robust) and Secondary categories
    to help interpret results correctly.

    Parameters:
    -----------
    metrics : dict
        Output from classification_report_full().
    model_name : str
        Name to display in the header.
    """
    print(f"\n{'='*60}")
    print(f"  {model_name} — Classification Metrics")
    print(f"{'='*60}")

    print(f"\n--- PRIMARY METRICS (Imbalance-Robust) ---")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f} "
          f"({metrics['balanced_accuracy']*100:.1f}%)")
    print(f"  Macro F1-Score:     {metrics['f1_macro']:.4f}")
    print(f"  Specificity:        {metrics['specificity']:.4f} "
          f"(catch rate for BAD grasps)")
    print(f"  Sensitivity:        {metrics['sensitivity']:.4f} "
          f"(catch rate for GOOD grasps)")

    if metrics['average_precision'] is not None:
        print(f"  PR-AUC (Avg Prec):  {metrics['average_precision']:.4f}")

    print(f"\n--- SECONDARY METRICS (interpret with caution) ---")
    print(f"  Accuracy:           {metrics['accuracy']:.4f} "
          f"({metrics['accuracy']*100:.1f}%) "
          f"[majority baseline = 63.7%]")
    print(f"  Weighted F1:        {metrics['f1_weighted']:.4f}")

    if metrics['auc_roc'] is not None:
        print(f"  ROC-AUC:            {metrics['auc_roc']:.4f}")

    print(f"\n--- PER-CLASS BREAKDOWN ---")
    class_names = ['negative', 'positive']
    for i, name in enumerate(class_names):
        p = metrics['per_class']['precision'][i]
        r = metrics['per_class']['recall'][i]
        f = metrics['per_class']['f1'][i]
        s = metrics['per_class']['support'][i]
        print(f"  {name:10s}: precision={p:.3f}  recall={r:.3f}  "
              f"f1={f:.3f}  support={s}")

    print(f"\n--- CONFUSION MATRIX ---")
    print(metrics['sklearn_report'])
