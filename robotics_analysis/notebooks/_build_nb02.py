"""
Helper script to rebuild notebooks/02_Grasp_Classification.ipynb with the
improved pipeline:
  - oriented crops
  - frozen-backbone transfer learning (ResNet-18 + EfficientNet-B0)
  - early stopping with AUC-ROC as the primary monitor
  - three-way model comparison

Run with `python _build_nb02.py` from inside notebooks/.
"""

import json
from pathlib import Path

NB = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def md(text):
    NB["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(text):
    NB["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    })


# ---------------------------------------------------------------------------
# Intro
# ---------------------------------------------------------------------------
md("""# Notebook 02: Grasp Classification (Improved Pipeline)

## Training CNN, frozen ResNet-18, and frozen EfficientNet-B0 on the Cornell Grasp Dataset

**Goal:** correctly identify *both* good and bad grasps -- not just the majority class.

### What's different from the first run

The first version of this notebook surfaced two problems:

1. **GraspCNN collapsed to the minority class** -- 45.2% accuracy, specificity 0.93 but
   sensitivity 0.16. It over-corrected when we added the weighted loss + sampler.
2. **ResNet-18 overfit aggressively** -- train acc hit 91.8% while test acc stalled at
   66.1% and test loss *grew* from 0.68 to 1.10 over 20 epochs. Classic memorisation.

This notebook applies four improvements from our research on imbalanced small-data CV:

| Improvement | Why |
|---|---|
| **Oriented crops** | Rotate each crop so the gripper axis is horizontal. The CNN no longer has to re-learn rotation invariance from scratch (Redmon & Angelova, 2015). |
| **Frozen backbone transfer learning** | Freeze the pretrained layers and only train `layer4` + `fc`. Dramatically reduces trainable parameters and regularises against overfitting. |
| **Early stopping on AUC-ROC** | Since our minority class is well above 20% (36.3%), ROC-AUC is a reliable primary metric. Stop training once AUC-ROC plateaus instead of running a fixed number of epochs. |
| **EfficientNet-B0 comparison** | Modern architecture with better accuracy-per-parameter than ResNet. Sanity-checks that ResNet's numbers aren't architecture-specific. |

### Primary metric: AUC-ROC

AUC-ROC is the *threshold-independent* summary of classifier quality. It measures the
probability that the model ranks a random positive example above a random negative
example. For our 36.3% minority class it's more stable than accuracy and avoids being
fooled by a model that collapses to the majority.

We still report balanced accuracy, macro-F1, specificity, sensitivity and PR-AUC
for full transparency. "Real accuracy" for this task means: can the model tell both
classes apart? That is precisely what AUC-ROC measures.
""")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
code("""import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Resolve src/ path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('')), 'robotics_analysis', 'src'))
sys.path.insert(0, '../src')

from data_utils import get_classification_loaders, CORNELL_CLASSES
from models import (
    GraspCNN, GraspResNet18, GraspEfficientNetB0,
    count_parameters, count_all_parameters, param_summary,
)
from train import train_model, evaluate_with_probs
from metrics import classification_report_full, compute_majority_baseline, print_metrics_summary
from visualize import (
    plot_training_curves, plot_confusion_matrix,
    plot_roc_curve, plot_precision_recall_curve,
    plot_metrics_comparison_table,
)

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Speed up fixed-size convolutions
    torch.backends.cudnn.benchmark = True
""")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
md("""## 1. Load Data with Oriented Crops + Imbalance Handling

`oriented=True` rotates each crop by `-angle` around the grasp centre before cropping,
so the gripper opening is always horizontal in the image. We then augment with horizontal
flip + colour jitter + `RandomErasing` (which fights overfitting by zeroing random
patches) but **not** random rotation -- that would undo the canonicalisation.
""")

code("""DATA_DIR = os.path.join('..', '..', 'data', 'archive (11)')

train_loader, test_loader, class_weights = get_classification_loaders(
    batch_size=32,
    data_dir=DATA_DIR,
    augment=True,
    oriented=True,   # <-- rotate crops so gripper axis is horizontal
)

class_weights = class_weights.to(device)
print(f"\\nClass weights on {device}: {class_weights}")

# Sanity check: one training batch should be ~50/50 thanks to WeightedRandomSampler
sample_batch = next(iter(train_loader))
_, labels_batch = sample_batch
pos = (labels_batch == 1).sum().item()
neg = (labels_batch == 0).sum().item()
print(f"Batch balance sanity check: {pos} positive / {neg} negative")
""")

# ---------------------------------------------------------------------------
# Majority baseline
# ---------------------------------------------------------------------------
md("""## 2. Majority-Class Baseline

A model that always predicts "positive" gets 63.7% accuracy but 50% balanced accuracy
and 50% AUC-ROC. Any useful model has to clearly beat these floors.
""")

code("""all_test_labels = []
for _, labels in test_loader:
    all_test_labels.extend(labels.numpy().tolist())

baseline = compute_majority_baseline(all_test_labels)
print(baseline['message'])
print(f"  Majority accuracy (floor): {baseline['majority_accuracy']:.3f}")
print(f"  Majority balanced accuracy: {baseline['majority_balanced_accuracy']:.3f}")
print(f"  Majority AUC-ROC:          0.500 (a constant predictor has no ranking ability)")
""")

# ---------------------------------------------------------------------------
# Model 1: GraspCNN baseline
# ---------------------------------------------------------------------------
md("""## 3. Model 1: GraspCNN (from scratch)

Our custom 3-block CNN. Now trained with **early stopping on AUC-ROC** -- we let the
scheduler run up to 40 epochs but stop as soon as AUC-ROC stops improving for 7 epochs
in a row. The best checkpoint (by AUC-ROC) is automatically restored at the end.
""")

code("""cnn_model = GraspCNN(num_classes=2)
param_summary(cnn_model, "GraspCNN")
""")

code("""cnn_criterion = nn.CrossEntropyLoss(weight=class_weights)

cnn_history = train_model(
    model=cnn_model,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=40,
    lr=0.001,
    device=device,
    criterion=cnn_criterion,
    early_stopping=True,
    patience=7,
    monitor='auc_roc',
)
""")

code("""fig = plot_training_curves(cnn_history, title_prefix="GraspCNN")
plt.savefig('../figures/training_curves_cnn.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""# Full evaluation on the restored best checkpoint
cnn_acc, cnn_preds, cnn_labels, cnn_probs = evaluate_with_probs(cnn_model, test_loader, device)
cnn_metrics = classification_report_full(cnn_labels, cnn_preds, y_prob=cnn_probs, class_names=CORNELL_CLASSES)
print_metrics_summary(cnn_metrics, model_name="GraspCNN (best checkpoint)")
""")

code("""fig = plot_confusion_matrix(cnn_labels, cnn_preds, CORNELL_CLASSES, title="GraspCNN")
plt.savefig('../figures/confusion_cnn.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""fig = plot_roc_curve(cnn_labels, cnn_probs, title="GraspCNN ROC Curve")
plt.savefig('../figures/roc_cnn.png', dpi=150, bbox_inches='tight')
plt.show()

fig = plot_precision_recall_curve(cnn_labels, cnn_probs, title="GraspCNN PR Curve")
plt.savefig('../figures/pr_cnn.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ---------------------------------------------------------------------------
# Model 2: ResNet-18 frozen
# ---------------------------------------------------------------------------
md("""## 4. Model 2: GraspResNet18 with Frozen Backbone

**Strategy:** freeze the ImageNet-pretrained features in `conv1`/`layer1`/`layer2`/`layer3`
and only train `layer4` + `fc`. This drops trainable parameters from ~11.2M to ~8.4M
(still meaningful capacity) but relies on the high-quality pretrained features for
everything else, which is the cure for the overfitting we saw previously.

The learning rate is halved to `5e-4` because the small trainable head shouldn't need
aggressive updates.
""")

code("""resnet_model = GraspResNet18(num_classes=2, freeze='partial')
param_summary(resnet_model, "GraspResNet18 (partial freeze)")
print(f"\\nFrozen layers: conv1, bn1, layer1, layer2, layer3")
print(f"Trainable layers: layer4, fc")
""")

code("""resnet_criterion = nn.CrossEntropyLoss(weight=class_weights)

resnet_history = train_model(
    model=resnet_model,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=40,
    lr=5e-4,
    device=device,
    criterion=resnet_criterion,
    early_stopping=True,
    patience=7,
    monitor='auc_roc',
)
""")

code("""fig = plot_training_curves(resnet_history, title_prefix="ResNet-18 (partial freeze)")
plt.savefig('../figures/training_curves_resnet.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""resnet_acc, resnet_preds, resnet_labels, resnet_probs = evaluate_with_probs(resnet_model, test_loader, device)
resnet_metrics = classification_report_full(resnet_labels, resnet_preds, y_prob=resnet_probs, class_names=CORNELL_CLASSES)
print_metrics_summary(resnet_metrics, model_name="ResNet-18 partial-freeze (best checkpoint)")
""")

code("""fig = plot_confusion_matrix(resnet_labels, resnet_preds, CORNELL_CLASSES, title="ResNet-18 (frozen backbone)")
plt.savefig('../figures/confusion_resnet.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""fig = plot_roc_curve(resnet_labels, resnet_probs, title="ResNet-18 ROC Curve")
plt.savefig('../figures/roc_resnet.png', dpi=150, bbox_inches='tight')
plt.show()

fig = plot_precision_recall_curve(resnet_labels, resnet_probs, title="ResNet-18 PR Curve")
plt.savefig('../figures/pr_resnet.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ---------------------------------------------------------------------------
# Model 3: EfficientNet-B0 frozen
# ---------------------------------------------------------------------------
md("""## 5. Model 3: GraspEfficientNetB0 with Frozen Backbone

EfficientNet-B0 uses MBConv (inverted residual) blocks with squeeze-and-excitation and
compound scaling. It gets better ImageNet accuracy than ResNet-18 with roughly half the
parameters. We apply the same partial-freeze strategy: freeze early stages, train the
last 3 MBConv stages + classifier.

This gives us a second, independent transfer-learning backbone -- if both ResNet and
EfficientNet land in roughly the same place, we know the signal is architecture-agnostic
and the dataset is genuinely near its performance ceiling for this crop-based setup.
""")

code("""effnet_model = GraspEfficientNetB0(num_classes=2, freeze='partial')
param_summary(effnet_model, "EfficientNet-B0 (partial freeze)")
""")

code("""effnet_criterion = nn.CrossEntropyLoss(weight=class_weights)

effnet_history = train_model(
    model=effnet_model,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=40,
    lr=5e-4,
    device=device,
    criterion=effnet_criterion,
    early_stopping=True,
    patience=7,
    monitor='auc_roc',
)
""")

code("""fig = plot_training_curves(effnet_history, title_prefix="EfficientNet-B0 (partial freeze)")
plt.savefig('../figures/training_curves_effnet.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""effnet_acc, effnet_preds, effnet_labels, effnet_probs = evaluate_with_probs(effnet_model, test_loader, device)
effnet_metrics = classification_report_full(effnet_labels, effnet_preds, y_prob=effnet_probs, class_names=CORNELL_CLASSES)
print_metrics_summary(effnet_metrics, model_name="EfficientNet-B0 partial-freeze (best checkpoint)")
""")

code("""fig = plot_confusion_matrix(effnet_labels, effnet_preds, CORNELL_CLASSES, title="EfficientNet-B0 (frozen backbone)")
plt.savefig('../figures/confusion_effnet.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""fig = plot_roc_curve(effnet_labels, effnet_probs, title="EfficientNet-B0 ROC Curve")
plt.savefig('../figures/roc_effnet.png', dpi=150, bbox_inches='tight')
plt.show()

fig = plot_precision_recall_curve(effnet_labels, effnet_probs, title="EfficientNet-B0 PR Curve")
plt.savefig('../figures/pr_effnet.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
md("""## 6. Three-Way Comparison

**Primary metric: AUC-ROC.** Accuracy is still reported but we interpret it through
the lens of the majority-class baseline (0.637) -- a number near that floor means the
model is close to predicting one class for everything.
""")

code("""results_dict = {
    'GraspCNN (scratch)': cnn_metrics,
    'ResNet-18 (partial freeze)': resnet_metrics,
    'EfficientNet-B0 (partial freeze)': effnet_metrics,
}

fig = plot_metrics_comparison_table(results_dict)
plt.savefig('../figures/model_comparison_table.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\n" + "=" * 82)
print("  MODEL COMPARISON -- primary metric is AUC-ROC")
print("=" * 82)
header = f"{'Metric':<24} {'Majority BL':>12} {'GraspCNN':>12} {'ResNet-18':>12} {'EffNet-B0':>12}"
print("\\n" + header)
print("-" * len(header))

metrics_to_compare = [
    ('AUC-ROC (primary)',    'auc_roc',             0.500),
    ('Balanced Accuracy',    'balanced_accuracy',   0.500),
    ('Macro F1',             'f1_macro',            0.390),
    ('PR-AUC',               'average_precision',   0.637),
    ('Accuracy',             'accuracy',            0.637),
    ('Sensitivity (pos)',    'sensitivity',         1.000),
    ('Specificity (neg)',    'specificity',         0.000),
]

def fmt(v):
    return f"{v:.3f}" if v is not None else "  N/A"

for name, key, bl_val in metrics_to_compare:
    c = cnn_metrics.get(key); r = resnet_metrics.get(key); e = effnet_metrics.get(key)
    print(f"  {name:<22} {bl_val:>12.3f} {fmt(c):>12} {fmt(r):>12} {fmt(e):>12}")

print(f"\\nTraining times: CNN={sum(cnn_history['epoch_times']):.0f}s, "
      f"ResNet={sum(resnet_history['epoch_times']):.0f}s, "
      f"EffNet={sum(effnet_history['epoch_times']):.0f}s")
print(f"Best epoch: CNN@{cnn_history['best_epoch']}, "
      f"ResNet@{resnet_history['best_epoch']}, "
      f"EffNet@{effnet_history['best_epoch']}")
""")

code("""# Combined ROC comparison
from sklearn.metrics import roc_curve, auc

fig, ax = plt.subplots(figsize=(8, 7))

for name, labels, probs, style in [
    ('GraspCNN',        cnn_labels,    cnn_probs,    'b-'),
    ('ResNet-18',       resnet_labels, resnet_probs, 'r-'),
    ('EfficientNet-B0', effnet_labels, effnet_probs, 'g-'),
]:
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    ax.plot(fpr, tpr, style, linewidth=2,
            label=f'{name} (AUC = {auc(fpr, tpr):.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/roc_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ---------------------------------------------------------------------------
# Imbalance analysis
# ---------------------------------------------------------------------------
md("""## 7. Imbalance Impact Analysis

Did each model learn to identify *both* classes, or did it collapse to the majority?
The gap between per-class recalls tells the story.
""")

code("""print("IMBALANCE IMPACT ANALYSIS")
print("=" * 64)
for name, m in results_dict.items():
    rec_neg = m['per_class']['recall'][0]
    rec_pos = m['per_class']['recall'][1]
    print(f"\\n{name}")
    print(f"  Recall on BAD  grasps (specificity): {rec_neg:.3f}")
    print(f"  Recall on GOOD grasps (sensitivity): {rec_pos:.3f}")
    print(f"  Gap: {abs(rec_pos - rec_neg):.3f} "
          f"({'BALANCED' if abs(rec_pos - rec_neg) < 0.15 else 'IMBALANCED -- model leans one way'})")
    print(f"  AUC-ROC: {m['auc_roc']:.3f} "
          f"({'well above' if m['auc_roc'] > 0.7 else 'near'} the 0.500 random floor)")
""")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
md("""## 8. Save Results""")

code("""os.makedirs('../saved_models', exist_ok=True)
torch.save(cnn_model.state_dict(),    '../saved_models/grasp_cnn.pth')
torch.save(resnet_model.state_dict(), '../saved_models/grasp_resnet18_frozen.pth')
torch.save(effnet_model.state_dict(), '../saved_models/grasp_efficientnet_b0_frozen.pth')
print("Model weights saved to saved_models/")

def metrics_to_serializable(metrics):
    result = {}
    for k, v in metrics.items():
        if k in ('confusion_matrix', 'confusion_matrix_normalized'):
            result[k] = v.tolist()
        elif k in ('roc_curve', 'pr_curve'):
            result[k] = {kk: vv.tolist() for kk, vv in v.items()} if v is not None else None
        else:
            result[k] = v if v is not None else None
    return result

def history_to_serializable(h):
    out = {}
    for k, v in h.items():
        if isinstance(v, list):
            out[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
        else:
            out[k] = v
    return out

results = {
    'grasp_cnn': {
        'metrics': metrics_to_serializable(cnn_metrics),
        'history': history_to_serializable(cnn_history),
        'parameters_trainable': count_parameters(cnn_model),
        'parameters_total':     count_all_parameters(cnn_model),
    },
    'grasp_resnet18_frozen': {
        'metrics': metrics_to_serializable(resnet_metrics),
        'history': history_to_serializable(resnet_history),
        'parameters_trainable': count_parameters(resnet_model),
        'parameters_total':     count_all_parameters(resnet_model),
    },
    'grasp_efficientnet_b0_frozen': {
        'metrics': metrics_to_serializable(effnet_metrics),
        'history': history_to_serializable(effnet_history),
        'parameters_trainable': count_parameters(effnet_model),
        'parameters_total':     count_all_parameters(effnet_model),
    },
}

with open('../saved_models/classification_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved to saved_models/classification_results.json")
""")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
md("""## Summary

### Models Trained
| Model | Trainable Params | Strategy | Max Epochs | Stopping |
|---|---|---|---|---|
| GraspCNN | all | from scratch | 40 | Early stop on AUC-ROC |
| ResNet-18 (partial freeze) | layer4 + fc only | ImageNet transfer | 40 | Early stop on AUC-ROC |
| EfficientNet-B0 (partial freeze) | last 3 stages + classifier | ImageNet transfer | 40 | Early stop on AUC-ROC |

### What we changed vs v1
1. **Oriented crops** encode grasp angle into pixel layout -> less for the CNN to learn.
2. **Partial freezing** keeps most of the pretrained feature extractor fixed -> fights the
   overfitting we saw on full fine-tuning.
3. **Early stopping on AUC-ROC** restores the best checkpoint automatically -> no more
   "train for 20 epochs and watch test loss climb".
4. **Three backbones** (custom CNN, ResNet-18, EfficientNet-B0) tell us whether the
   bottleneck is the model or the data.

### Why AUC-ROC is the primary metric here
With a minority class of 36.3% (well above the 20% threshold where PR-AUC typically
dominates), AUC-ROC is a reliable summary of the model's *ability to rank* good vs bad
grasps. A collapsed-to-majority model scores AUC 0.5 regardless of its raw accuracy,
which is exactly the failure mode we want to catch.

### Interpretation checklist
- **AUC-ROC well above 0.5** -> model actually distinguishes the classes (not just
  predicting majority).
- **Sensitivity ~= specificity** -> model treats both classes fairly.
- **Test loss didn't grow over training** -> early stopping worked; not overfit.
- **Train-test accuracy gap small** -> model is generalising, not memorising.
""")


# ---------------------------------------------------------------------------
# Write the notebook
# ---------------------------------------------------------------------------
out_path = Path(__file__).with_name("02_Grasp_Classification.ipynb")
out_path.write_text(json.dumps(NB, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out_path} ({len(NB['cells'])} cells)")
