"""
models.py - Neural Network Architectures for Grasp Classification

This module defines THREE CNN architectures for classifying grasp quality:

1. GraspCNN: A custom CNN built from scratch
   - 3 convolutional blocks with increasing channels (32->64->128)
   - ~0.8M parameters -- baseline "learned from scratch" model

2. GraspResNet18: Pretrained ResNet-18 with optional layer freezing
   - Pretrained on ImageNet (1.2M images)
   - Can train all layers, only fc, or fc + layer4
   - ~11.2M total / ~1K-2.6M trainable (depending on freeze strategy)

3. GraspEfficientNetB0: Pretrained EfficientNet-B0
   - Modern efficient architecture using compound scaling + MBConv blocks
   - ~5.3M total parameters -- better accuracy-per-param than ResNet
   - Supports the same freezing strategies

WHY MULTIPLE MODELS?
- GraspCNN: "how good can we get from scratch on 8K samples?"
- GraspResNet18: "how much does ImageNet pretraining help?"
- GraspEfficientNetB0: "is a more modern / efficient backbone better?"
- Comparing them shows the impact of architecture and transfer learning.

WHY LAYER FREEZING?
Our ResNet-18 overfit badly on the first run (train 92% vs test 66%, test loss
growing epoch over epoch). When the dataset is small (8K samples) and the model
is large (11M parameters), fine-tuning every layer is overkill -- the model
memorises training images. Freezing the backbone forces the model to rely on
the already-good ImageNet features and only learn a small task-specific head,
which regularises aggressively.

FREEZE STRATEGIES supported:
    'none'       -> train every layer (original behaviour)
    'backbone'   -> freeze everything except the final classifier/fc layer
    'partial'    -> freeze everything except the last conv block + classifier
                    (layer4 for ResNet, final stages for EfficientNet)

ALL MODELS OUTPUT RAW LOGITS (no softmax):
- Shape: (batch_size, 2)
- CrossEntropyLoss applies softmax internally.
- For probabilities use torch.softmax(outputs, dim=1).
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ============================================================================
# Custom CNN (from scratch)
# ============================================================================

class GraspCNN(nn.Module):
    """
    Custom CNN for grasp classification on 224x224 image crops.
    3 conv blocks (32/64/128) + adaptive pool + 2-layer classifier.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: low-level features
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2: mid-level features
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3: high-level features
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# Freezing helpers
# ============================================================================

def _freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True


# ============================================================================
# ResNet-18 (pretrained) with optional freezing
# ============================================================================

def GraspResNet18(num_classes=2, freeze='none'):
    """
    Pretrained ResNet-18 for binary grasp classification.

    Parameters:
    -----------
    num_classes : int
        Number of output classes (default 2).
    freeze : str
        'none'     -> all layers trainable (fine-tune everything)
        'backbone' -> freeze everything except the new fc layer
        'partial'  -> freeze everything except layer4 + fc
                      (layer4 is the last residual stage, closest to the task)

    Returns:
    --------
    model : nn.Module
        ResNet-18 with replaced fc and the requested freezing strategy.
    """
    model = tv_models.resnet18(weights='IMAGENET1K_V1')

    # Replace the final fc layer first (gets fresh trainable parameters)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze == 'none':
        pass  # everything trainable
    elif freeze == 'backbone':
        _freeze_all(model)
        _unfreeze(model.fc)
    elif freeze == 'partial':
        _freeze_all(model)
        _unfreeze(model.layer4)
        _unfreeze(model.fc)
    else:
        raise ValueError(f"Unknown freeze strategy: {freeze}")

    return model


# ============================================================================
# EfficientNet-B0 (pretrained) with optional freezing
# ============================================================================

def GraspEfficientNetB0(num_classes=2, freeze='none'):
    """
    Pretrained EfficientNet-B0 for binary grasp classification.

    EfficientNet uses MBConv (inverted residual) blocks with squeeze-and-excitation
    and compound scaling. B0 is the smallest variant (~5.3M params) and typically
    matches or beats ResNet-50 on ImageNet while being much cheaper to run.

    torchvision's EfficientNet-B0 classifier is:
        Sequential(Dropout(0.2), Linear(1280, 1000))
    We replace only the final Linear to target num_classes.

    Parameters:
    -----------
    num_classes : int
        Number of output classes (default 2).
    freeze : str
        'none'     -> fine-tune everything
        'backbone' -> freeze all features, train only the classifier head
        'partial'  -> freeze features[:6], train features[6:] + classifier
                      (keeps the last 2 MBConv stages trainable)
    """
    model = tv_models.efficientnet_b0(weights='IMAGENET1K_V1')

    # classifier is Sequential(Dropout, Linear). Replace the Linear.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze == 'none':
        pass
    elif freeze == 'backbone':
        _freeze_all(model)
        _unfreeze(model.classifier)
    elif freeze == 'partial':
        _freeze_all(model)
        # features is a Sequential of 9 stages (0..8). Unfreeze the last 3 stages
        # + classifier -- these capture the most task-specific representations.
        for i in range(6, len(model.features)):
            _unfreeze(model.features[i])
        _unfreeze(model.classifier)
    else:
        raise ValueError(f"Unknown freeze strategy: {freeze}")

    return model


# ============================================================================
# Introspection helpers
# ============================================================================

def count_parameters(model):
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    """Return total parameters (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


def param_summary(model, name="Model"):
    """Print a 1-line summary of trainable vs total parameters."""
    trainable = count_parameters(model)
    total = count_all_parameters(model)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"{name}: {trainable:,} trainable / {total:,} total ({pct:.1f}%)")
