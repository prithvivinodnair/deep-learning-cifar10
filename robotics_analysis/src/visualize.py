"""
visualize.py - Plotting and Visualization for Grasp Classification

This module provides presentation-ready plots for:
- Grasp rectangles overlaid on images (green=good, red=bad)
- Sample crop grids
- Class distribution with imbalance annotations
- Training curves
- Confusion matrices (raw + normalized)
- ROC curves
- Precision-Recall curves (critical for imbalanced data)
- Metrics comparison tables

All functions return matplotlib figure objects for flexible display/saving.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from collections import Counter


def plot_grasp_rectangles(image, pos_grasps, neg_grasps, figsize=(10, 8), title=None):
    """
    Overlay grasp rectangles on an image.

    Green rectangles = positive (successful) grasps
    Red rectangles = negative (failed) grasps

    This visualization shows WHERE on the object the robot can successfully
    grab (green) vs where it would fail (red).

    Parameters:
    -----------
    image : numpy.ndarray or PIL.Image
        The RGB image (640x480).
    pos_grasps : list of dict
        Positive grasp rectangles from parse_grasp_rectangles().
    neg_grasps : list of dict
        Negative grasp rectangles from parse_grasp_rectangles().
    """
    fig, ax = plt.subplots(figsize=figsize)

    if hasattr(image, 'convert'):
        image = np.array(image)

    ax.imshow(image)

    # Draw positive grasps in green
    for grasp in pos_grasps:
        corners = grasp['corners']
        # Close the rectangle by connecting last corner to first
        xs = [c[0] for c in corners] + [corners[0][0]]
        ys = [c[1] for c in corners] + [corners[0][1]]
        ax.plot(xs, ys, 'g-', linewidth=2, alpha=0.8)
        # Mark center
        cx, cy = grasp['center']
        ax.plot(cx, cy, 'g+', markersize=8, markeredgewidth=2)

    # Draw negative grasps in red
    for grasp in neg_grasps:
        corners = grasp['corners']
        xs = [c[0] for c in corners] + [corners[0][0]]
        ys = [c[1] for c in corners] + [corners[0][1]]
        ax.plot(xs, ys, 'r-', linewidth=2, alpha=0.8)
        cx, cy = grasp['center']
        ax.plot(cx, cy, 'r+', markersize=8, markeredgewidth=2)

    # Legend
    ax.plot([], [], 'g-', linewidth=2, label=f'Positive ({len(pos_grasps)})')
    ax.plot([], [], 'r-', linewidth=2, label=f'Negative ({len(neg_grasps)})')
    ax.legend(fontsize=11, loc='upper right')

    ax.set_title(title or 'Grasp Rectangles', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig


def plot_sample_crops(dataset, n=20, figsize=(16, 8)):
    """
    Display a grid of sample grasp crops with their labels.

    Shows what the model actually sees: 224x224 crops centered on grasp locations.

    Parameters:
    -----------
    dataset : CornellGraspClassification
        The dataset (with or without transforms).
    n : int
        Total number of crops to display (half positive, half negative).
    """
    # Collect samples by class. `dataset.labels` mirrors samples 1:1 and is
    # independent of the sample tuple format, so we use it here.
    pos_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
    neg_indices = [i for i, label in enumerate(dataset.labels) if label == 0]

    n_per_class = n // 2
    show_pos = pos_indices[:n_per_class]
    show_neg = neg_indices[:n_per_class]

    all_indices = show_pos + show_neg
    cols = 5
    rows = (len(all_indices) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, idx in enumerate(all_indices):
        crop, label = dataset[idx]

        # If crop is a tensor, convert to displayable format
        if hasattr(crop, 'numpy'):
            # Undo normalization for display
            img = crop.numpy().transpose(1, 2, 0)  # CHW → HWC
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
        else:
            img = np.array(crop) / 255.0

        axes[i].imshow(img)
        color = 'green' if label == 1 else 'red'
        title = 'GOOD grasp' if label == 1 else 'BAD grasp'
        axes[i].set_title(title, fontsize=9, color=color, fontweight='bold')
        axes[i].axis('off')

    for i in range(len(all_indices), len(axes)):
        axes[i].axis('off')

    fig.suptitle('Sample Grasp Crops (224x224)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_class_distribution(labels, figsize=(8, 5)):
    """
    Bar chart showing positive vs negative grasp counts with imbalance annotations.

    Includes:
    - Count and percentage for each class
    - A dashed line at 50% showing where perfect balance would be
    - The imbalance ratio

    Parameters:
    -----------
    labels : list of int
        All labels (0s and 1s).
    """
    labels = np.array(labels)
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    total = len(labels)

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        ['Negative\n(Bad Grasp)', 'Positive\n(Good Grasp)'],
        [n_neg, n_pos],
        color=['#e74c3c', '#2ecc71'],
        edgecolor='black',
        linewidth=0.5,
        width=0.6,
    )

    # Add count and percentage labels on bars
    for bar, count in zip(bars, [n_neg, n_pos]):
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 50% reference line
    ax.axhline(y=total / 2, color='blue', linestyle='--', linewidth=1.5,
               alpha=0.6, label=f'Perfect balance ({total//2:,} each)')

    ax.set_ylabel('Number of Grasps', fontsize=12)
    ax.set_title(f'Class Distribution — {n_pos:,} Positive vs {n_neg:,} Negative '
                 f'(Ratio: {n_pos/n_neg:.2f}:1)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(n_pos, n_neg) * 1.2)
    plt.tight_layout()
    return fig


def plot_training_curves(history, figsize=(14, 5), title_prefix=""):
    """
    Plot training and validation loss/accuracy curves side by side.

    Diagnostic guide:
    - Both curves going down = model is learning
    - Training down but validation flat = overfitting
    - Large gap between curves = overfitting
    - Both flat = model isn't learning (lr too low or model too simple)

    Parameters:
    -----------
    history : dict
        Output from train_model().
    title_prefix : str
        Prefix for plot titles (e.g., "GraspCNN" or "ResNet-18").
    """
    epochs = list(range(1, len(history['train_loss']) + 1))
    n_epochs = len(epochs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- LOSS CURVE ---
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'r-', label='Validation Loss', linewidth=2)

    # Mark best epoch (from history if available, else from min test_loss)
    best_epoch = history.get('best_epoch') or (int(np.argmin(history['test_loss'])) + 1)
    ax1.axvline(x=best_epoch, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'Best epoch ({best_epoch})')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{title_prefix} Training vs Validation Loss',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, n_epochs + 0.5)

    # --- ACCURACY CURVE ---
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'r-', label='Validation Accuracy', linewidth=2)

    # Mark majority baseline
    ax2.axhline(y=63.7, color='gray', linestyle=':', linewidth=1.5,
                alpha=0.7, label='Majority Baseline (63.7%)')

    # Vertical line at best epoch (no overlapping arrow annotation)
    ax2.axvline(x=best_epoch, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'Best epoch ({best_epoch})')

    # Put the best-accuracy number in the subtitle so nothing floats over curves.
    best_acc = max(history['test_acc'])
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(
        f'{title_prefix} Training vs Validation Accuracy\n'
        f'Best test acc: {best_acc:.1f}% @ epoch {best_epoch}',
        fontsize=12, fontweight='bold'
    )
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, n_epochs + 0.5)

    # Legend pinned to a corner that doesn't collide with high-accuracy curves
    ax2.legend(fontsize=9, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, classes, figsize=(12, 5), title=None):
    """
    Plot confusion matrix: both raw counts and normalized side by side.

    The normalized matrix is especially important for imbalanced data:
    each row shows what fraction of that class was classified correctly.
    If the negative row shows low recall, the model is ignoring the minority class.

    Parameters:
    -----------
    y_true : list/array of int
    y_pred : list/array of int
    classes : list of str
    """
    from sklearn.metrics import confusion_matrix as cm_func

    cm = cm_func(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=ax1, linewidths=0.5, linecolor='gray',
                annot_kws={'size': 14})
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('True', fontsize=12)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')

    # Normalized (row-normalized = per-class recall)
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=ax2, linewidths=0.5, linecolor='gray',
                vmin=0, vmax=1, annot_kws={'size': 14})
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('True', fontsize=12)
    ax2.set_title('Normalized (Per-Class Recall)', fontsize=13, fontweight='bold')

    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_prob, figsize=(7, 6), title=None):
    """
    Plot ROC curve for binary grasp classification.

    ROC shows the trade-off between True Positive Rate (sensitivity) and
    False Positive Rate (1 - specificity) at various classification thresholds.

    AUC = 1.0 → perfect classifier
    AUC = 0.5 → random classifier (diagonal line)

    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1).
    y_prob : numpy.ndarray, shape (N, 2)
        Predicted probabilities. Column 1 = P(positive grasp).
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, 'b-', linewidth=2,
            label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.500)')

    ax.fill_between(fpr, tpr, alpha=0.1, color='blue')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(title or f'ROC Curve (AUC = {roc_auc:.3f})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_true, y_prob, figsize=(7, 6), title=None):
    """
    Plot Precision-Recall curve — MORE INFORMATIVE than ROC for imbalanced data.

    WHY PR IS BETTER THAN ROC FOR IMBALANCED DATA:
    ROC uses FPR = FP/(FP+TN). When TN is very large (many negative samples
    correctly classified), even a lot of FPs barely move the FPR. So ROC
    can look great even when precision is poor.

    PR uses Precision = TP/(TP+FP), which directly measures how many of
    the positive predictions are correct. If the model has many false positives,
    precision drops — PR curve catches this while ROC might miss it.

    The baseline for PR is the prevalence (positive class proportion):
    - For our data: 63.7% positive → baseline AP ≈ 0.637

    Parameters:
    -----------
    y_true : array-like
        True binary labels.
    y_prob : numpy.ndarray, shape (N, 2)
        Predicted probabilities.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
    ap = average_precision_score(y_true, y_prob[:, 1])

    # Baseline: proportion of positive samples
    baseline = np.mean(np.array(y_true) == 1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, 'b-', linewidth=2,
            label=f'PR Curve (AP = {ap:.3f})')
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.6, label=f'Baseline (prevalence = {baseline:.3f})')

    ax.fill_between(recall, precision, alpha=0.1, color='blue')

    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title or f'Precision-Recall Curve (AP = {ap:.3f})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    return fig


def plot_metrics_comparison_table(results_dict, figsize=(13, 2.8)):
    """
    Render a comparison table of metrics across models as a matplotlib figure.

    Includes the majority baseline row to contextualize accuracy numbers.

    Parameters:
    -----------
    results_dict : dict
        {model_name: metrics_dict} where metrics_dict is from classification_report_full().
    """
    # Metrics to display
    metric_names = [
        'Accuracy', 'Balanced Acc', 'Macro F1', 'Weighted F1',
        'Sensitivity', 'Specificity', 'ROC-AUC', 'PR-AUC'
    ]

    # Build table data
    model_names = list(results_dict.keys())
    table_data = []

    # Add majority baseline row first
    baseline_row = ['63.7%', '50.0%', '~39.0%', '~50.0%',
                    '100.0%', '0.0%', '50.0%', '~63.7%']
    table_data.append(baseline_row)

    for name in model_names:
        m = results_dict[name]
        row = [
            f"{m['accuracy']*100:.1f}%",
            f"{m['balanced_accuracy']*100:.1f}%",
            f"{m['f1_macro']:.3f}",
            f"{m['f1_weighted']:.3f}",
            f"{m['sensitivity']:.3f}",
            f"{m['specificity']:.3f}",
            f"{m['auc_roc']:.3f}" if m['auc_roc'] else 'N/A',
            f"{m['average_precision']:.3f}" if m['average_precision'] else 'N/A',
        ]
        table_data.append(row)

    all_row_labels = ['Majority Baseline'] + model_names

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    table = ax.table(
        cellText=table_data,
        rowLabels=all_row_labels,
        colLabels=metric_names,
        cellLoc='center',
        loc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Style header row
    for j in range(len(metric_names)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Style baseline row (light gray)
    for j in range(len(metric_names)):
        table[1, j].set_facecolor('#f0f0f0')

    # Style row labels
    for i in range(len(all_row_labels)):
        table[i + 1, -1].set_facecolor('#d9e2f3')

    fig.suptitle('Model Comparison — Classification Metrics',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.82, bottom=0.05, left=0.18, right=0.98)
    return fig
