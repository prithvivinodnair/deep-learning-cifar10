"""
models.py - Neural Network Architecture Definitions

This module defines the CNN (Convolutional Neural Network) architectures
used for image classification on CIFAR-10.

KEY ARCHITECTURE CONCEPTS:
- Conv2d: A convolution layer slides small filters (3x3) across the image,
  detecting local patterns like edges, corners, and textures.
- BatchNorm: Normalizes the output of each layer to help training converge faster.
- ReLU: An activation function that outputs max(0, x). It introduces
  non-linearity — without it, stacking layers would be equivalent to
  a single linear transformation, no matter how many layers we add.
- MaxPool2d: Reduces spatial dimensions by taking the maximum value in each
  2x2 region. This makes the model somewhat invariant to small translations
  and reduces computation.
- Dropout: Randomly sets some neurons to 0 during training. This prevents
  co-adaptation (neurons relying too much on each other) and reduces overfitting.
- Linear (Fully Connected): Traditional neural network layer where every
  input connects to every output. Used at the end to map features to class scores.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models


class SimpleCNN(nn.Module):
    """
    A simple but effective CNN for CIFAR-10 classification.

    Architecture overview:
    INPUT (3x32x32) = 3-channel color image, 32x32 pixels

    BLOCK 1: Learn low-level features (edges, simple textures)
        Conv2d(3→32)  → BatchNorm → ReLU
        Conv2d(32→32) → BatchNorm → ReLU
        MaxPool2d     → size becomes 16x16
        Dropout(0.25) → randomly drop 25% of feature maps

    BLOCK 2: Learn higher-level features (parts of objects, complex patterns)
        Conv2d(32→64)  → BatchNorm → ReLU
        Conv2d(64→64)  → BatchNorm → ReLU
        MaxPool2d      → size becomes 8x8
        Dropout(0.25)

    CLASSIFIER: Map features to class predictions
        Flatten        → 64 * 8 * 8 = 4096 values
        Linear(4096→512) → ReLU → Dropout(0.5)
        Linear(512→10) → 10 class scores (one per CIFAR-10 class)

    Total parameters: ~2.2M (small enough to train on CPU)
    Expected accuracy: ~85-90% on CIFAR-10 test set
    """

    def __init__(self, num_classes=10):
        """
        Initialize the CNN layers.

        Parameters:
        -----------
        num_classes : int
            Number of output classes. Default 10 for CIFAR-10.
        """
        # super().__init__() calls the parent class (nn.Module) constructor.
        # This is required for PyTorch to properly track all layers and parameters.
        super().__init__()

        # ---- FEATURE EXTRACTION LAYERS ----
        # nn.Sequential groups layers into a pipeline: output of one feeds into the next.
        self.features = nn.Sequential(

            # === BLOCK 1: Low-level feature extraction ===

            # Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
            # - in_channels=3: input has 3 color channels (R, G, B)
            # - out_channels=32: learn 32 different 3x3 filters
            #   Each filter learns to detect a different pattern (horizontal edge,
            #   vertical edge, color blob, etc.)
            # - kernel_size=3: each filter is 3x3 pixels
            # - padding=1: add 1 pixel of zeros around the border so output size
            #   stays the same as input size (32x32 → 32x32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),

            # BatchNorm2d(32): Normalize the 32 feature maps.
            # For each of the 32 channels, it learns to shift and scale the values
            # so they have mean≈0 and std≈1. This stabilizes training and allows
            # higher learning rates.
            nn.BatchNorm2d(32),

            # ReLU (Rectified Linear Unit): f(x) = max(0, x)
            # - Negative values become 0, positive values stay the same
            # - This is the "activation function" that adds non-linearity
            # - Without activation functions, the entire network would collapse
            #   to a single linear transformation, regardless of depth
            # - inplace=True saves memory by modifying values directly instead
            #   of creating a copy
            nn.ReLU(inplace=True),

            # Second conv layer in Block 1: 32→32 channels
            # Stacking two 3x3 convolutions has the same "receptive field" as
            # one 5x5 convolution, but uses fewer parameters and adds another
            # non-linearity (more expressive).
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # MaxPool2d(2): Take the max value in each 2x2 region.
            # Input: 32x32 → Output: 16x16
            # This halves the spatial dimensions, making the model:
            # 1. Faster (fewer values to process in later layers)
            # 2. More robust to small translations (slight shifts don't change the max)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dropout2d(0.25): During training, randomly zero out 25% of entire
            # feature maps (channels). This is spatial dropout — it drops whole
            # channels rather than individual pixels, which works better for conv layers.
            # During evaluation (model.eval()), dropout is automatically disabled.
            nn.Dropout2d(0.25),


            # === BLOCK 2: Higher-level feature extraction ===
            # Now working with 16x16 feature maps. These layers learn to combine
            # the low-level patterns from Block 1 into more complex features.

            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 32→64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # 64→64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),          # 16x16 → 8x8
            nn.Dropout2d(0.25),
        )

        # ---- CLASSIFICATION LAYERS ----
        # After the conv blocks, we have 64 feature maps of size 8x8.
        # We flatten these into a 1D vector (64*8*8 = 4096 values) and use
        # fully connected (Linear) layers to map to class predictions.
        self.classifier = nn.Sequential(
            # Flatten is done in forward() below.

            # Linear(4096, 512): Fully connected layer.
            # Every one of the 4096 input values connects to every one of 512 outputs.
            # This layer learns which combinations of features correspond to which classes.
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),

            # Dropout(0.5): Drop 50% of neurons during training.
            # Higher dropout here than in conv layers because fully connected layers
            # have many more parameters and are more prone to overfitting.
            nn.Dropout(0.5),

            # Final layer: 512 → num_classes (10).
            # Outputs 10 raw scores (called "logits"). The highest score = predicted class.
            # We do NOT apply softmax here because PyTorch's CrossEntropyLoss
            # already includes softmax internally (for numerical stability).
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """
        Define the forward pass: how data flows through the network.

        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, 3, 32, 32)
            A batch of input images.

        Returns:
        --------
        torch.Tensor, shape (batch_size, num_classes)
            Raw class scores (logits) for each image in the batch.
            Higher score = model is more confident about that class.
        """
        # Pass through convolutional feature extractor
        # Input: (batch, 3, 32, 32) → Output: (batch, 64, 8, 8)
        x = self.features(x)

        # Flatten: reshape from (batch, 64, 8, 8) to (batch, 4096)
        # -1 means "figure out this dimension automatically"
        # We need to flatten because Linear layers expect 1D input per sample.
        x = x.view(x.size(0), -1)

        # Pass through classifier
        # Input: (batch, 4096) → Output: (batch, 10)
        x = self.classifier(x)

        return x


def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.

    A "parameter" is a single number that the model learns during training.
    For example, a Conv2d(3, 32, 3) has 3*32*3*3 + 32 = 896 parameters
    (weights + biases).

    More parameters = more capacity to learn complex patterns, but also
    more risk of overfitting and longer training time.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cifar_resnet18(num_classes=10):
    """
    Create a ResNet-18 modified for CIFAR-10's 32x32 images.

    Standard ResNet-18 is designed for ImageNet (224x224). For 32x32 CIFAR images
    we need two modifications:
    1. Replace the 7x7 stride-2 conv1 with a 3x3 stride-1 conv (preserves spatial info)
    2. Remove the early max pooling layer (32x32 is already small)

    Without these changes, the spatial dimensions would shrink too aggressively
    and the network would lose important spatial information.

    Parameters:
    -----------
    num_classes : int
        Number of output classes. Default 10 for CIFAR-10.

    Returns:
    --------
    model : nn.Module
        Modified ResNet-18 ready for CIFAR-10 training.
    """
    model = tv_models.resnet18(weights=None)

    # Replace first conv: 7x7 stride-2 → 3x3 stride-1 (keeps 32x32 resolution)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove max pooling (would shrink 32x32 too aggressively)
    model.maxpool = nn.Identity()

    # Replace final fully connected layer for our number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def extract_features(model, loader, device=None):
    """
    Extract feature vectors from a trained SimpleCNN's feature extractor.

    Runs all images through model.features + flatten, producing a rich
    4096-dimensional representation that captures learned visual patterns.
    These features can then be fed to traditional ML models like XGBoost.

    Parameters:
    -----------
    model : SimpleCNN
        A trained SimpleCNN model (must have a .features attribute).
    loader : DataLoader
        Data loader for the images to extract features from.
    device : torch.device or None
        If None, auto-detects GPU.

    Returns:
    --------
    features : numpy.ndarray, shape (N, feature_dim)
        Extracted feature vectors (e.g., 4096-dim for SimpleCNN).
    labels : numpy.ndarray, shape (N,)
        Corresponding class labels.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)

            # Run through feature extractor only
            feats = model.features(images)
            feats = feats.view(feats.size(0), -1)  # Flatten

            all_features.append(feats.cpu().numpy())
            all_labels.extend(batch_labels.numpy().tolist())

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)

    return features, labels
