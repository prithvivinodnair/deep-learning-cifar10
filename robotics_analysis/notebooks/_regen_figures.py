"""
Regenerate the figures that had overlap issues, using the saved
classification_results.json. Rebuilds:

  - figures/training_curves_cnn.png
  - figures/training_curves_resnet.png
  - figures/training_curves_effnet.png
  - figures/roc_comparison.png         (now with a zoomed inset so all 3 curves
                                        are distinguishable in the top-left)
  - figures/model_comparison_table.png (tighter layout, no dead whitespace)

No model re-training is needed because the JSON stores both per-epoch histories
and the full roc_curve data (fpr, tpr, thresholds) for each model.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from visualize import plot_training_curves, plot_metrics_comparison_table  # noqa: E402


RESULTS_JSON = HERE.parent / "saved_models" / "classification_results.json"
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

with open(RESULTS_JSON) as f:
    RESULTS = json.load(f)


# ---------------------------------------------------------------------------
# 1. Training curves (uses updated plot_training_curves with text-box + vlines)
# ---------------------------------------------------------------------------
MODEL_SPECS = [
    ("grasp_cnn",                      "GraspCNN",                       "training_curves_cnn.png"),
    ("grasp_resnet18_frozen",          "ResNet-18 (partial freeze)",     "training_curves_resnet.png"),
    ("grasp_efficientnet_b0_frozen",   "EfficientNet-B0 (partial freeze)", "training_curves_effnet.png"),
]

for key, title_prefix, filename in MODEL_SPECS:
    hist = RESULTS[key]["history"]
    fig = plot_training_curves(hist, title_prefix=title_prefix)
    out = FIG_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[training curves] wrote {out}")


# ---------------------------------------------------------------------------
# 2. ROC comparison with zoomed inset
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

style_map = {
    "grasp_cnn":                     ("GraspCNN",        "tab:blue"),
    "grasp_resnet18_frozen":         ("ResNet-18",       "tab:red"),
    "grasp_efficientnet_b0_frozen":  ("EfficientNet-B0", "tab:green"),
}

curves = {}
for key, (label, color) in style_map.items():
    m = RESULTS[key]["metrics"]
    fpr = m["roc_curve"]["fpr"]
    tpr = m["roc_curve"]["tpr"]
    auc = m["auc_roc"]
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC = {auc:.3f})")
    curves[key] = (fpr, tpr, label, color, auc)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.500)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve Comparison (with zoomed top-left)", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# Inset: zoom in on the top-left corner where the three high-AUC curves cluster.
# This is the only way to actually distinguish them visually.
axins = inset_axes(ax, width="45%", height="45%", loc="center right",
                   bbox_to_anchor=(-0.02, -0.08, 1, 1),
                   bbox_transform=ax.transAxes)
for key, (fpr, tpr, label, color, auc) in curves.items():
    axins.plot(fpr, tpr, color=color, linewidth=2)
axins.set_xlim(0.0, 0.15)
axins.set_ylim(0.80, 1.01)
axins.grid(True, alpha=0.3)
axins.set_title("Zoom: FPR ≤ 0.15, TPR ≥ 0.80", fontsize=10)
axins.tick_params(labelsize=8)

# Draw a box on the main plot showing which region the inset covers
from matplotlib.patches import Rectangle
rect = Rectangle((0.0, 0.80), 0.15, 0.20, linewidth=1.2,
                 edgecolor="black", facecolor="none", linestyle="--")
ax.add_patch(rect)

plt.tight_layout()
out = FIG_DIR / "roc_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[roc comparison] wrote {out}")


# ---------------------------------------------------------------------------
# 3. Model comparison table (tighter figure, no dead whitespace)
# ---------------------------------------------------------------------------
# plot_metrics_comparison_table expects {name: metrics_dict}
display_order = {
    "GraspCNN (scratch)":               RESULTS["grasp_cnn"]["metrics"],
    "ResNet-18 (partial freeze)":       RESULTS["grasp_resnet18_frozen"]["metrics"],
    "EfficientNet-B0 (partial freeze)": RESULTS["grasp_efficientnet_b0_frozen"]["metrics"],
}
fig = plot_metrics_comparison_table(display_order)
out = FIG_DIR / "model_comparison_table.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[comparison table] wrote {out}")

print("\nAll regenerated figures written successfully.")
