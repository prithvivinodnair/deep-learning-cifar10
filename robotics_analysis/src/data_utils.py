"""
data_utils.py - Cornell Grasp Dataset Parsing and Loading

This module handles everything related to loading the Cornell Grasp Dataset:
- Parsing grasp rectangle annotations from .txt files
- Converting 4-corner rectangles to (center_x, center_y, angle, width, height)
- Creating cropped image patches for classification
- Building PyTorch DataLoaders with class imbalance handling

CORNELL GRASP DATASET STRUCTURE:
Each image has associated annotation files:
- pcd####r.png    → RGB image (640x480)
- pcd####d.tiff   → Depth image (640x480)
- pcd####cpos.txt → Positive (successful) grasp rectangles
- pcd####cneg.txt → Negative (failed) grasp rectangles

Each grasp rectangle is defined by 4 corner points (4 lines in the file):
    x1 y1
    x2 y2
    x3 y3
    x4 y4

These 4 corners define a rotated rectangle where the robot gripper would close.

CLASS IMBALANCE:
The dataset has 5,110 positive grasps vs 2,909 negative grasps (1.76:1 ratio).
We handle this with:
1. WeightedRandomSampler — oversamples minority class so batches are ~50/50
2. Weighted CrossEntropyLoss — penalizes minority class errors more heavily
3. Stratified splits — preserves class ratio in train/test sets
"""

import os
import math
import warnings
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms


# ============================================================================
# Constants
# ============================================================================

# Class names for binary grasp classification
# Index 0 = negative (bad grasp), Index 1 = positive (good grasp)
CORNELL_CLASSES = ['negative', 'positive']

# Using ImageNet normalization values because we use pretrained ResNet
# These values were computed from 1.2 million ImageNet images.
# Since ResNet was trained with these exact values, using them ensures
# the pretrained features work correctly on our data.
CORNELL_MEAN = (0.485, 0.456, 0.406)  # R, G, B channel means
CORNELL_STD = (0.229, 0.224, 0.225)   # R, G, B channel std devs

# Crop size for grasp patches — matches ResNet's expected input
CROP_SIZE = 224


# ============================================================================
# Grasp Rectangle Parsing
# ============================================================================

def parse_grasp_rectangles(filepath):
    """
    Parse grasp rectangle annotations from a Cornell Grasp Dataset .txt file.

    Each grasp is defined by 4 consecutive lines, each containing an (x, y)
    coordinate pair representing one corner of the grasp rectangle:

        x1 y1      ← corner 1
        x2 y2      ← corner 2
        x3 y3      ← corner 3
        x4 y4      ← corner 4

    A file can contain multiple grasps (0, 1, 2, ... rectangles).

    Data quality handling:
    - Empty files (0 bytes) → returns empty list []
    - Lines with NaN values → that grasp rectangle is skipped with a warning
    - Trailing whitespace → handled by .strip()

    Parameters:
    -----------
    filepath : str
        Path to a cpos.txt or cneg.txt file.

    Returns:
    --------
    rectangles : list of dict
        Each dict contains:
        - 'corners': list of 4 (x, y) tuples — the raw corner coordinates
        - 'center': (center_x, center_y) — midpoint of the rectangle
        - 'angle': float — rotation angle in degrees
        - 'width': float — gripper opening width
        - 'height': float — gripper plate length
    """
    rectangles = []

    # EMPTY FILE GUARD: If file doesn't exist or is empty, return []
    # This handles pcd0154cneg.txt which is 0 bytes
    if not os.path.exists(filepath):
        return rectangles

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception:
        return rectangles

    # Filter out empty lines (trailing newlines, blank lines)
    # .strip() removes the trailing whitespace that exists on every line
    clean_lines = [line.strip() for line in lines if line.strip()]

    if len(clean_lines) == 0:
        return rectangles

    # Group lines into chunks of 4 (each chunk = one grasp rectangle)
    for i in range(0, len(clean_lines) - 3, 4):
        chunk = clean_lines[i:i + 4]

        if len(chunk) < 4:
            break

        # NaN GUARD: Check if any line in this 4-line chunk contains NaN
        # This handles the corrupted grasp in pcd0165cpos.txt (lines 9 & 12)
        has_nan = False
        corners = []

        for line in chunk:
            if 'nan' in line.lower():
                has_nan = True
                break
            try:
                parts = line.split()
                x, y = float(parts[0]), float(parts[1])
                if math.isnan(x) or math.isnan(y):
                    has_nan = True
                    break
                corners.append((x, y))
            except (ValueError, IndexError):
                has_nan = True
                break

        if has_nan:
            warnings.warn(f"Skipping grasp with NaN/invalid values in {filepath}")
            continue

        # Convert 4 corners to center, angle, width, height
        # The 4 corners form a rotated rectangle:
        #   corner1 -------- corner2
        #      |                |
        #   corner4 -------- corner3
        #
        # Center = midpoint of diagonal (corner1 + corner3) / 2
        # Width = distance between corner1 and corner2 (gripper opening)
        # Height = distance between corner2 and corner3 (gripper plate length)
        # Angle = angle of the top edge (corner1 → corner2) relative to horizontal

        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = corners

        # Center: average of opposite corners (or average of all 4)
        center_x = (x1 + x2 + x3 + x4) / 4.0
        center_y = (y1 + y2 + y3 + y4) / 4.0

        # Width: distance between adjacent corners (top edge)
        width = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Height: distance between adjacent corners (side edge)
        height = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

        # Angle: angle of the top edge relative to horizontal
        # atan2 returns radians, we convert to degrees
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        rectangles.append({
            'corners': corners,
            'center': (center_x, center_y),
            'angle': angle,
            'width': width,
            'height': height,
        })

    return rectangles


# ============================================================================
# Classification Dataset
# ============================================================================

class CornellGraspClassification(Dataset):
    """
    PyTorch Dataset for Cornell Grasp binary classification.

    For each grasp rectangle in the dataset, this class:
    1. Loads the corresponding RGB image
    2. Crops a 224x224 patch centered on the grasp rectangle's center
       - If oriented=True, rotates the image so the gripper axis is horizontal
         BEFORE cropping. This makes the visual pattern consistent regardless
         of the grasp angle, letting the CNN focus on whether the geometry is
         graspable rather than re-learning rotation invariance from scratch.
    3. Applies image transforms (augmentation, normalization)
    4. Returns (crop, label) where label is 0 (bad) or 1 (good)

    The key idea: we're NOT doing full-image classification. We crop a local
    region around each grasp point and ask "would grabbing HERE succeed?"

    Parameters:
    -----------
    data_dir : str
        Path to the dataset root (contains folders 01-10).
    transform : torchvision.transforms.Compose or None
        Image transforms to apply to each crop.
    crop_size : int
        Size of the square crop around each grasp center (default 224).
    oriented : bool
        If True, rotate the image by -angle around the grasp centre before
        cropping, so the gripper opening is always horizontal in the crop.
        This encodes grasp orientation into the visual input and has been
        shown to help on Cornell-style datasets (see Redmon & Angelova, 2015).
    """

    def __init__(self, data_dir, transform=None, crop_size=CROP_SIZE, oriented=False):
        self.data_dir = data_dir
        self.transform = transform
        self.crop_size = crop_size
        self.oriented = oriented

        # These lists store every grasp sample in the dataset
        # Each sample: (image_path, center_x, center_y, angle_degrees, label)
        self.samples = []
        self.labels = []    # Just the labels — used for sampler/weight computation

        self._load_dataset()

    def _load_dataset(self):
        """
        Walk through all folders, parse all cpos.txt and cneg.txt files,
        and build the samples list.
        """
        # The dataset has folders named 01, 02, ..., 10
        folders = sorted([
            f for f in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, f))
            and f.isdigit()
        ])

        for folder in folders:
            folder_path = os.path.join(self.data_dir, folder)

            # Find all RGB images in this folder (pattern: pcd####r.png)
            image_files = sorted([
                f for f in os.listdir(folder_path)
                if f.endswith('r.png')
            ])

            for img_file in image_files:
                # Extract the base ID (e.g., "pcd0100" from "pcd0100r.png")
                base_id = img_file.replace('r.png', '')
                img_path = os.path.join(folder_path, img_file)

                # Parse positive grasps (label = 1)
                cpos_path = os.path.join(folder_path, f'{base_id}cpos.txt')
                pos_rects = parse_grasp_rectangles(cpos_path)
                for rect in pos_rects:
                    cx, cy = rect['center']
                    angle = rect['angle']
                    self.samples.append((img_path, cx, cy, angle, 1))
                    self.labels.append(1)

                # Parse negative grasps (label = 0)
                cneg_path = os.path.join(folder_path, f'{base_id}cneg.txt')
                neg_rects = parse_grasp_rectangles(cneg_path)
                for rect in neg_rects:
                    cx, cy = rect['center']
                    angle = rect['angle']
                    self.samples.append((img_path, cx, cy, angle, 0))
                    self.labels.append(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load one sample: crop the image around the grasp center and return
        (crop_tensor, label).

        Two cropping modes:
        - AXIS-ALIGNED (oriented=False): Crop a 224x224 box from the original
          image centred at (cx, cy). Simple but the gripper axis can point in
          any direction, so the CNN must learn rotation invariance implicitly.
        - ORIENTED (oriented=True): Rotate the image by -angle around (cx, cy),
          then crop. The gripper opening is always horizontal in the result,
          so orientation is baked into the pixel layout and the CNN can focus
          on graspability rather than re-learning rotation from scratch.

        Both modes clamp crop bounds to image boundaries and resize to the
        target crop_size.
        """
        img_path, cx, cy, angle, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')
        img_w, img_h = image.size  # PIL uses (width, height)

        if self.oriented:
            # PIL rotates counter-clockwise in degrees. Our angle is the top
            # edge's direction relative to horizontal, so rotating by -angle
            # aligns the gripper axis with horizontal.
            # expand=False keeps the image size fixed so (cx, cy) stays valid.
            rotated = image.rotate(
                angle,
                resample=Image.BILINEAR,
                center=(cx, cy),
                expand=False,
            )
            # Now crop axis-aligned around (cx, cy) in rotated image coords.
            half = self.crop_size // 2
            left = int(cx) - half
            top = int(cy) - half
            right = left + self.crop_size
            bottom = top + self.crop_size

            left = max(0, left)
            top = max(0, top)
            right = min(img_w, right)
            bottom = min(img_h, bottom)

            crop = rotated.crop((left, top, right, bottom))
        else:
            # Axis-aligned crop (original behaviour)
            half = self.crop_size // 2
            left = int(cx) - half
            top = int(cy) - half
            right = left + self.crop_size
            bottom = top + self.crop_size

            left = max(0, left)
            top = max(0, top)
            right = min(img_w, right)
            bottom = min(img_h, bottom)

            crop = image.crop((left, top, right, bottom))

        # Resize to exact crop_size if clamping changed dimensions
        if crop.size != (self.crop_size, self.crop_size):
            crop = crop.resize((self.crop_size, self.crop_size), Image.BILINEAR)

        # Apply transforms (augmentation + normalization)
        if self.transform:
            crop = self.transform(crop)

        return crop, label


# ============================================================================
# Class Imbalance Handling
# ============================================================================

def compute_class_weights(labels):
    """
    Compute class weights inversely proportional to class frequency.

    This is used with nn.CrossEntropyLoss(weight=...) to make the loss function
    penalize errors on the minority class (negative grasps) more heavily.

    Formula: weight_c = total_samples / (num_classes * count_c)
    - If a class has fewer samples, it gets a HIGHER weight
    - If a class has more samples, it gets a LOWER weight

    For our dataset (5,110 positive, 2,909 negative):
    - weight_negative = 8019 / (2 * 2909) = 1.378
    - weight_positive = 8019 / (2 * 5110) = 0.785

    This means misclassifying a negative grasp costs ~1.76x more than
    misclassifying a positive grasp in the loss computation.

    Parameters:
    -----------
    labels : list of int
        All labels in the dataset (0s and 1s).

    Returns:
    --------
    weights : torch.FloatTensor of shape (num_classes,)
        Weight for each class, to be passed to CrossEntropyLoss.
    """
    labels = np.array(labels)
    total = len(labels)
    num_classes = len(np.unique(labels))

    weights = []
    for c in range(num_classes):
        count_c = np.sum(labels == c)
        weight_c = total / (num_classes * count_c)
        weights.append(weight_c)

    return torch.FloatTensor(weights)


def make_weighted_sampler(labels):
    """
    Create a WeightedRandomSampler that oversamples the minority class.

    Without this sampler, each training batch would have ~64% positive and
    ~36% negative samples (matching the dataset distribution). The model
    might learn to just predict "positive" for everything because it's
    right 64% of the time.

    With the sampler, each batch has roughly 50% positive and 50% negative
    samples. The model MUST learn to distinguish both classes to do well.

    How it works:
    - Each sample gets a weight = 1 / count_of_its_class
    - Positive samples (5,110 total) each get weight 1/5110
    - Negative samples (2,909 total) each get weight 1/2909
    - PyTorch draws samples with probability proportional to their weight
    - Since negative samples have higher weight, they get drawn more often

    IMPORTANT: When using WeightedRandomSampler, do NOT set shuffle=True
    in the DataLoader — the sampler handles the ordering.

    Parameters:
    -----------
    labels : list of int
        Labels for each sample in the training set.

    Returns:
    --------
    sampler : WeightedRandomSampler
        Pass this to DataLoader(sampler=...).
    """
    labels = np.array(labels)
    class_counts = np.bincount(labels)

    # Weight for each sample = 1 / count_of_its_class
    sample_weights = np.array([1.0 / class_counts[label] for label in labels])

    # Convert to tensor
    sample_weights = torch.DoubleTensor(sample_weights)

    # replacement=True allows sampling the same item multiple times per epoch
    # This is necessary for oversampling — minority samples appear more than once
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )

    return sampler


# ============================================================================
# DataLoader Creation
# ============================================================================

def get_classification_loaders(batch_size=32, data_dir=None, augment=True,
                               num_workers=0, oriented=False):
    """
    Create training and test DataLoaders for grasp classification.

    This function:
    1. Loads the full Cornell Grasp Dataset
    2. Performs an 80/20 stratified split (preserves class ratio in both sets)
    3. Creates a WeightedRandomSampler for balanced training batches
    4. Computes class weights for the weighted loss function
    5. Returns loaders and weights ready for training

    Parameters:
    -----------
    batch_size : int
        Number of samples per batch. 32 works well for 224x224 crops on
        GPUs with 4-6GB VRAM.

    data_dir : str or None
        Path to the dataset root containing folders 01-10.
        If None, defaults to '../data/archive (11)/' relative to notebooks.

    augment : bool
        Whether to apply data augmentation to training images.
        Augmentation creates variations of each crop (flips, rotations,
        color changes) to help the model generalize.

    num_workers : int
        Number of parallel data loading workers.

    oriented : bool
        If True, rotate each crop so the gripper opening is horizontal.
        Encodes grasp orientation into the pixel layout. See
        CornellGraspClassification docstring for details.

    Returns:
    --------
    train_loader : DataLoader
        Training data with WeightedRandomSampler for balanced batches.
    test_loader : DataLoader
        Test data without resampling (natural distribution).
    class_weights : torch.FloatTensor
        Weights for CrossEntropyLoss, shape (2,).
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'archive (11)')

    # --- TRANSFORMS ---
    if augment:
        # Training transforms: augmentation + normalization.
        # Note: if oriented=True we've already canonicalised rotation, so adding
        # RandomRotation would *undo* that signal. We keep flips + colour jitter
        # + a small random erasing (helps fight the overfitting we saw previously).
        aug_list = [
            # A horizontally-flipped grasp is still the same grasp (gripper
            # opens the other way but geometry is symmetric).
            transforms.RandomHorizontalFlip(),
        ]
        if not oriented:
            # Only apply random rotation when crops aren't pre-oriented.
            aug_list.append(transforms.RandomRotation(10))

        aug_list.extend([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=CORNELL_MEAN, std=CORNELL_STD),
            # RandomErasing blanks out a random patch of the tensor. Acts as a
            # strong regulariser against memorising specific pixel patterns --
            # directly targets the overfitting we observed in earlier runs.
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
        transform_train = transforms.Compose(aug_list)
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CORNELL_MEAN, std=CORNELL_STD),
        ])

    # Test transforms: NO augmentation — we want a fair, consistent evaluation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CORNELL_MEAN, std=CORNELL_STD),
    ])

    # --- LOAD FULL DATASET ---
    # First, load with no transforms to get all samples and labels
    full_dataset = CornellGraspClassification(data_dir, transform=None, oriented=oriented)
    all_labels = full_dataset.labels

    print(f"Total samples: {len(all_labels)}")
    print(f"  Positive (good grasps): {sum(1 for l in all_labels if l == 1)}")
    print(f"  Negative (bad grasps):  {sum(1 for l in all_labels if l == 0)}")

    # --- STRATIFIED SPLIT ---
    # train_test_split with stratify=labels ensures both train and test sets
    # have the same class proportion (~63.7% positive, ~36.3% negative)
    indices = list(range(len(all_labels)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,          # 80% train, 20% test
        stratify=all_labels,    # Preserve class balance
        random_state=42         # Reproducible split
    )

    # Create separate datasets with appropriate transforms
    train_dataset = CornellGraspClassification(data_dir, transform=transform_train, oriented=oriented)
    test_dataset = CornellGraspClassification(data_dir, transform=transform_test, oriented=oriented)

    # Create subset datasets using the split indices
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    # Get training labels for sampler and weight computation
    train_labels = [all_labels[i] for i in train_indices]
    test_labels = [all_labels[i] for i in test_indices]

    print(f"\nTrain set: {len(train_labels)} samples")
    print(f"  Positive: {sum(1 for l in train_labels if l == 1)} "
          f"({100*sum(1 for l in train_labels if l == 1)/len(train_labels):.1f}%)")
    print(f"  Negative: {sum(1 for l in train_labels if l == 0)} "
          f"({100*sum(1 for l in train_labels if l == 0)/len(train_labels):.1f}%)")
    print(f"\nTest set: {len(test_labels)} samples")
    print(f"  Positive: {sum(1 for l in test_labels if l == 1)} "
          f"({100*sum(1 for l in test_labels if l == 1)/len(test_labels):.1f}%)")
    print(f"  Negative: {sum(1 for l in test_labels if l == 0)} "
          f"({100*sum(1 for l in test_labels if l == 0)/len(test_labels):.1f}%)")

    # --- CLASS WEIGHTS (for loss function) ---
    class_weights = compute_class_weights(train_labels)
    print(f"\nClass weights for loss: negative={class_weights[0]:.3f}, "
          f"positive={class_weights[1]:.3f}")

    # --- WEIGHTED SAMPLER (for balanced batches) ---
    train_sampler = make_weighted_sampler(train_labels)

    # --- CREATE DATALOADERS ---
    # Training: use sampler (NOT shuffle — they're mutually exclusive)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,      # Balances batches ~50/50
        num_workers=num_workers,
        pin_memory=True
    )

    # Test: no sampler, no shuffle — evaluate on natural distribution
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, class_weights
