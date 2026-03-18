"""
data_utils.py - Dataset Loading and Preprocessing Utilities

This module handles everything related to the CIFAR-10 dataset:
- Loading the dataset from torchvision (auto-downloads if not present)
- Applying transforms (normalization, augmentation)
- Creating DataLoaders for training/testing
- Providing flattened numpy arrays for traditional ML (sklearn)

KEY CONCEPTS:
- Transform: A function that modifies each image before it's fed to the model.
  Example: converting pixel values from 0-255 to 0-1 (normalization).
- DataLoader: A PyTorch utility that batches images together and shuffles them.
  Instead of feeding one image at a time, we feed 64 images at once (a "batch").
- Augmentation: Artificially creating variations of training images (flipping,
  cropping) to help the model generalize better.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# ============================================================================
# CIFAR-10 Constants
# ============================================================================

# The 10 classes in CIFAR-10. Index 0 = airplane, index 1 = automobile, etc.
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# These are the per-channel (R, G, B) mean and standard deviation values
# computed across the entire CIFAR-10 training set.
# WHY WE NEED THESE: Neural networks train better when input values are
# centered around 0 with a standard deviation of ~1. Raw pixel values (0-255)
# or even scaled values (0-1) are not centered. Subtracting the mean and
# dividing by std makes each channel have mean≈0 and std≈1.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)  # R, G, B channel means
CIFAR10_STD = (0.2470, 0.2435, 0.2616)   # R, G, B channel std devs


# ============================================================================
# PyTorch DataLoaders (for deep learning models)
# ============================================================================

def get_cifar10_loaders(batch_size=64, data_dir='../data', augment=True, num_workers=0):
    """
    Create PyTorch DataLoaders for CIFAR-10 training and test sets.

    Parameters:
    -----------
    batch_size : int (default=64)
        How many images to group together in each batch.
        - Smaller batches (32) = noisier gradients, can help escape local minima
        - Larger batches (128, 256) = smoother gradients, faster training
        - 64 is a good default that balances speed and quality

    data_dir : str
        Where to store/find the downloaded dataset files.

    augment : bool (default=True)
        Whether to apply data augmentation to training images.
        Augmentation = randomly flipping/cropping images to create variety.
        This helps prevent OVERFITTING (model memorizing training data
        instead of learning general patterns).

    num_workers : int (default=0)
        Number of parallel processes for loading data. 0 means load in
        the main process (simpler, works everywhere). Set to 2-4 on
        machines with multiple CPU cores for faster loading.

    Returns:
    --------
    train_loader : DataLoader
        Yields batches of (images, labels) from the training set.
        Images shape: (batch_size, 3, 32, 32) - 3 color channels, 32x32 pixels
        Labels shape: (batch_size,) - integer class indices (0-9)

    test_loader : DataLoader
        Same format but from the test set. No augmentation applied.
    """

    # --- TRAINING TRANSFORMS ---
    if augment:
        # When augment=True, we apply random transformations to training images.
        # This is like showing the model slightly different versions of each image
        # every time it sees it, which helps it learn more robust features.
        transform_train = transforms.Compose([
            # RandomHorizontalFlip: 50% chance of flipping image left-right.
            # A cat facing left is still a cat facing right.
            # NOTE: We don't flip vertically because upside-down objects are rare.
            transforms.RandomHorizontalFlip(),

            # RandomCrop: First pads the image by 4 pixels on each side (with zeros),
            # then randomly crops back to 32x32. This simulates the object being
            # in slightly different positions within the frame.
            transforms.RandomCrop(32, padding=4),

            # ToTensor: Converts PIL Image (H, W, C) with values 0-255
            # to PyTorch Tensor (C, H, W) with values 0.0-1.0.
            # PyTorch expects channels FIRST (C, H, W), not last (H, W, C).
            transforms.ToTensor(),

            # Normalize: Subtracts mean and divides by std for each channel.
            # After this: pixel values are roughly in range [-2, 2] with mean≈0.
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])
    else:
        # Without augmentation, we just convert and normalize.
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])

    # --- TEST TRANSFORMS ---
    # IMPORTANT: We NEVER augment test data. We want a consistent, fair evaluation.
    # The test set should reflect real-world conditions (no artificial variations).
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    # --- LOAD DATASETS ---
    # torchvision.datasets.CIFAR10 automatically downloads the dataset if
    # it's not already in data_dir. The dataset is ~170MB.
    # train=True gives us the 50,000 training images.
    # train=False gives us the 10,000 test images.
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,         # Training set
        download=True,      # Download if not already present
        transform=transform_train  # Apply our transforms to each image
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,        # Test set
        download=True,
        transform=transform_test
    )

    # --- CREATE DATALOADERS ---
    # DataLoader wraps a dataset and provides:
    # 1. Batching: groups images into batches of size batch_size
    # 2. Shuffling: randomizes order each epoch (training only!)
    # 3. Parallel loading: can use multiple CPU cores (num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Shuffle training data every epoch
        num_workers=num_workers,
        pin_memory=True     # Speeds up CPU-to-GPU transfer (harmless on CPU)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,      # Don't shuffle test data - we want reproducible results
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_raw_cifar10(data_dir='../data'):
    """
    Load CIFAR-10 WITHOUT any transforms (raw PIL images).
    Useful for visualization and exploration where we want original images.

    Returns:
    --------
    train_dataset, test_dataset : torchvision.datasets.CIFAR10
        Datasets where each item is (PIL_Image, label_int).
    """
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=None
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=None
    )
    return train_dataset, test_dataset


# ============================================================================
# NumPy Arrays (for traditional ML with scikit-learn)
# ============================================================================

def get_cifar10_numpy(data_dir='../data', subset_size=None):
    """
    Load CIFAR-10 as flattened numpy arrays for use with scikit-learn.

    Traditional ML models like Logistic Regression and SVM expect input
    as 1D feature vectors, not 2D images. So we FLATTEN each 32x32x3 image
    into a single vector of length 3072 (= 32 * 32 * 3).

    This throws away all spatial information! A pixel's position in the
    image is lost. This is exactly WHY traditional ML struggles with images
    and WHY CNNs are so much better (they preserve spatial structure).

    Parameters:
    -----------
    data_dir : str
        Path to dataset directory.

    subset_size : int or None
        If specified, only return this many training samples.
        Useful for SVM which is very slow on large datasets.
        None = use all 50,000 training samples.

    Returns:
    --------
    X_train : numpy array, shape (n_train, 3072)
        Flattened training images, pixel values scaled to [0, 1].
    y_train : numpy array, shape (n_train,)
        Training labels (integers 0-9).
    X_test : numpy array, shape (10000, 3072)
        Flattened test images.
    y_test : numpy array, shape (10000,)
        Test labels.
    """
    # Load raw datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True
    )

    # train_dataset.data is a numpy array of shape (50000, 32, 32, 3)
    # with uint8 values (0-255). We reshape to (50000, 3072) and scale to [0,1].
    X_train = train_dataset.data.reshape(-1, 3072).astype(np.float32) / 255.0
    y_train = np.array(train_dataset.targets)

    X_test = test_dataset.data.reshape(-1, 3072).astype(np.float32) / 255.0
    y_test = np.array(test_dataset.targets)

    # Optionally take a subset of training data
    if subset_size is not None and subset_size < len(X_train):
        # Use a fixed random seed for reproducibility
        rng = np.random.RandomState(42)
        indices = rng.choice(len(X_train), subset_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    return X_train, y_train, X_test, y_test
