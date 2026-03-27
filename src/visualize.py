"""
visualize.py - Plotting and Visualization Utilities

This module provides functions to create presentation-ready plots for:
- Displaying sample images from the dataset
- Class distribution charts
- Training loss and accuracy curves
- Confusion matrices
- Misclassified image analysis
- CNN filter visualization

All functions use matplotlib and seaborn for consistent, publication-quality plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def plot_sample_grid(dataset, classes, n_per_class=5, figsize=(12, 7)):
    """
    Display a grid of sample images: n_per_class examples for each class.

    This is one of the most important visualizations — it gives an immediate
    sense of what the model is working with.

    Parameters:
    -----------
    dataset : torchvision dataset
        Raw CIFAR-10 dataset (no transforms). Each item is (PIL_Image, label).

    classes : list of str
        Class names, e.g., ['airplane', 'automobile', ...].

    n_per_class : int
        Number of example images to show per class.

    figsize : tuple
        Figure size in inches (width, height).
    """
    n_classes = len(classes)

    # Collect n_per_class images for each class
    # We iterate through the dataset and pick the first n_per_class images
    # for each class label.
    class_images = {i: [] for i in range(n_classes)}
    for img, label in dataset:
        if len(class_images[label]) < n_per_class:
            class_images[label].append(np.array(img))
        # Stop early once we have enough for every class
        if all(len(v) >= n_per_class for v in class_images.values()):
            break

    # Create the grid: rows = classes, columns = samples
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=figsize)
    fig.suptitle('CIFAR-10 Sample Images', fontsize=16, fontweight='bold')

    for row in range(n_classes):
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.imshow(class_images[row][col])
            ax.axis('off')  # Hide axis ticks and labels

            # Add class name on the left side (first column only)
            if col == 0:
                ax.set_title(classes[row], fontsize=10, loc='left', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_class_distribution(dataset, classes, figsize=(10, 5)):
    """
    Bar chart showing how many images are in each class.

    For CIFAR-10, this should be perfectly balanced (5,000 per class in train,
    1,000 per class in test). In real-world datasets, classes are often
    imbalanced, which requires special handling.

    Parameters:
    -----------
    dataset : torchvision dataset
        The dataset to analyze.

    classes : list of str
        Class names.
    """
    # Count how many images belong to each class
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        labels = [label for _, label in dataset]

    counts = Counter(labels)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        [classes[i] for i in range(len(classes))],
        [counts[i] for i in range(len(classes))],
        color=plt.cm.Set3(np.linspace(0, 1, len(classes))),
        edgecolor='black',
        linewidth=0.5
    )

    # Add count labels on top of each bar
    for bar, count in zip(bars, [counts[i] for i in range(len(classes))]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('CIFAR-10 Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_pixel_histograms(dataset, figsize=(12, 4)):
    """
    Plot the distribution of pixel values for each color channel (R, G, B).

    This shows us the range and distribution of pixel intensities in the
    dataset, which informs our normalization strategy.
    """
    # Get all images as a numpy array
    if hasattr(dataset, 'data'):
        data = dataset.data  # Shape: (N, 32, 32, 3), uint8
    else:
        data = np.array([np.array(img) for img, _ in dataset])

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    colors = ['red', 'green', 'blue']
    channel_names = ['Red', 'Green', 'Blue']

    for i, (ax, color, name) in enumerate(zip(axes, colors, channel_names)):
        # Flatten all pixel values for this channel
        channel_data = data[:, :, :, i].flatten()

        ax.hist(channel_data, bins=50, color=color, alpha=0.7, density=True)
        ax.set_title(f'{name} Channel', fontsize=12)
        ax.set_xlabel('Pixel Value (0-255)')
        ax.set_ylabel('Density')

        # Add mean and std as text
        mean_val = channel_data.mean()
        std_val = channel_data.std()
        ax.axvline(mean_val, color='black', linestyle='--', linewidth=1.5)
        ax.text(0.95, 0.95, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Pixel Value Distribution by Channel', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_training_curves(history, figsize=(14, 5)):
    """
    Plot training and validation loss/accuracy curves side by side.

    These curves are the primary diagnostic tool during training:
    - Both curves going down = model is learning
    - Training curve goes down but validation plateaus = OVERFITTING
    - Both curves are flat = model is not learning (lr too low? model too simple?)
    - Curves are noisy = lr too high or batch size too small

    Parameters:
    -----------
    history : dict
        Output from train_model(), containing 'train_loss', 'test_loss',
        'train_acc', 'test_acc' lists.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- LOSS CURVE ---
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # --- ACCURACY CURVE ---
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Add the best test accuracy as an annotation
    best_epoch = np.argmax(history['test_acc']) + 1
    best_acc = max(history['test_acc'])
    ax2.annotate(f'Best: {best_acc:.1f}% (epoch {best_epoch})',
                 xy=(best_epoch, best_acc),
                 xytext=(best_epoch + 2, best_acc - 5),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=10, color='green', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, classes, figsize=(10, 8), title=None):
    """
    Plot a confusion matrix as a heatmap.

    A confusion matrix shows, for each true class (rows), how many images
    were predicted as each class (columns). The diagonal = correct predictions.
    Off-diagonal = errors.

    Reading example: If row="cat" and column="dog" has a value of 50, it means
    50 cat images were incorrectly classified as dogs.

    Parameters:
    -----------
    y_true : list/array of int
        True class labels.
    y_pred : list/array of int
        Predicted class labels.
    classes : list of str
        Class names.
    """
    from sklearn.metrics import confusion_matrix as cm_func

    # Compute the confusion matrix
    cm = cm_func(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    # Use seaborn heatmap for a clean, colorful visualization
    sns.heatmap(
        cm, annot=True, fmt='d',  # 'd' = integer format
        cmap='Blues',              # Blue color scale
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_misclassified(images, true_labels, pred_labels, classes, n=20, figsize=(15, 8)):
    """
    Display a grid of misclassified images with true and predicted labels.

    This helps us understand what kinds of mistakes the model makes.
    Common patterns: cat↔dog, automobile↔truck (visually similar classes).

    Parameters:
    -----------
    images : numpy array or list of PIL images
        The test images.
    true_labels : list of int
        True class labels.
    pred_labels : list of int
        Predicted class labels.
    classes : list of str
        Class names.
    n : int
        Number of misclassified images to display.
    """
    # Find indices where prediction != true label
    wrong_indices = [i for i in range(len(true_labels))
                     if true_labels[i] != pred_labels[i]]

    # Take the first n wrong predictions
    n = min(n, len(wrong_indices))
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(n):
        idx = wrong_indices[i]
        ax = axes[i]

        # Handle both numpy arrays and PIL images
        if isinstance(images[idx], np.ndarray):
            ax.imshow(images[idx])
        else:
            ax.imshow(np.array(images[idx]))

        true_name = classes[true_labels[idx]]
        pred_name = classes[pred_labels[idx]]
        ax.set_title(f'True: {true_name}\nPred: {pred_name}',
                     fontsize=8, color='red')
        ax.axis('off')

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')

    fig.suptitle('Misclassified Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_per_class_accuracy(y_true, y_pred, classes, figsize=(10, 5)):
    """
    Bar chart showing accuracy for each individual class.

    This reveals which classes are easy vs hard for the model.
    """
    from sklearn.metrics import confusion_matrix as cm_func

    cm = cm_func(y_true, y_pred)
    # Per-class accuracy = diagonal / row sum
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

    fig, ax = plt.subplots(figsize=figsize)
    colors = ['green' if acc >= 80 else 'orange' if acc >= 60 else 'red'
              for acc in per_class_acc]

    bars = ax.bar(classes, per_class_acc, color=colors, edgecolor='black', linewidth=0.5)

    # Add percentage labels
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.axhline(y=np.mean(per_class_acc), color='blue', linestyle='--',
               label=f'Average: {np.mean(per_class_acc):.1f}%')
    ax.legend(fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def visualize_filters(model, figsize=(12, 4)):
    """
    Visualize the learned filters (kernels) of the first convolutional layer.

    The first conv layer learns basic patterns like edges, colors, and textures.
    Each filter is a 3x3x3 array (3x3 spatial, 3 color channels), which we
    can display as a small color image.

    Higher layers learn more abstract patterns that are harder to visualize directly.
    """
    # Get the first conv layer's weights
    # Shape: (out_channels, in_channels, height, width) = (32, 3, 3, 3)
    first_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv = module
            break

    if first_conv is None:
        print("No Conv2d layer found!")
        return None

    filters = first_conv.weight.data.cpu().numpy()
    n_filters = filters.shape[0]  # Number of filters (32)

    # Normalize each filter to [0, 1] for display
    cols = 8
    rows = (n_filters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_filters):
        # Each filter is (3, 3, 3) - we need to transpose to (3, 3, 3) HWC for imshow
        f = filters[i].transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
        # Normalize to [0, 1]
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        axes[i].imshow(f)
        axes[i].axis('off')
        axes[i].set_title(f'F{i+1}', fontsize=7)

    for i in range(n_filters, len(axes)):
        axes[i].axis('off')

    fig.suptitle('First Conv Layer Filters (Learned Edge/Color Detectors)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# Need torch import for visualize_filters
import torch


def plot_roc_curves(all_labels, all_probs, classes, figsize=(10, 8), title=None):
    """
    Plot per-class ROC curves using one-vs-rest approach.

    For each class, we treat it as a binary problem: "is this class or not?"
    and plot the ROC curve showing how well the model distinguishes it.

    Parameters:
    -----------
    all_labels : list/array of int
        True class indices (length N).
    all_probs : numpy.ndarray, shape (N, num_classes)
        Softmax probabilities for each sample.
    classes : list of str
        Class names.
    """
    from sklearn.metrics import roc_curve, auc

    all_labels = np.array(all_labels)
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    all_aucs = []
    for i in range(n_classes):
        # Binary: is this class i or not?
        binary_labels = (all_labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        all_aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=colors[i], linewidth=1.5,
                label=f'{classes[i]} (AUC = {roc_auc:.3f})')

    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')

    macro_auc = np.mean(all_aucs)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title or f'ROC Curves (Macro AUC = {macro_auc:.3f})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_roc_comparison(model_results, classes, figsize=(10, 7)):
    """
    Compare macro-average ROC curves across multiple models.

    Parameters:
    -----------
    model_results : dict
        {model_name: (labels, probabilities)} where labels is list of int
        and probabilities is numpy array (N, num_classes).
    classes : list of str
        Class names.
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_results)))

    for (model_name, (labels, probs)), color in zip(model_results.items(), colors):
        labels = np.array(labels)
        n_classes = len(classes)

        # Compute macro-average ROC: average across all classes
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_classes):
            binary_labels = (labels == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)

        ax.plot(all_fpr, mean_tpr, color=color, linewidth=2,
                label=f'{model_name} (AUC = {macro_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison (Macro-Average)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
