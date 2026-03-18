"""
train.py - Training and Evaluation Functions

This module contains the core training loop and evaluation logic.

THE TRAINING PROCESS (simplified):
1. Feed a batch of images through the network → get predictions
2. Compare predictions to true labels → compute the LOSS (how wrong we are)
3. Compute GRADIENTS: how should each parameter change to reduce the loss?
4. UPDATE parameters in the direction that reduces the loss (gradient descent)
5. Repeat for all batches = one EPOCH
6. Repeat for many epochs until the model converges

KEY TERMS:
- Epoch: One complete pass through the entire training dataset.
- Batch: A subset of the training data processed together (e.g., 64 images).
- Loss: A number measuring how wrong the model's predictions are. Lower = better.
- Gradient: The direction and magnitude of change needed to reduce the loss.
- Learning Rate: How big of a step to take in the gradient direction.
  Too high = overshoot the optimal values. Too low = training takes forever.
- Overfitting: When the model memorizes training data but fails on new data.
  Signs: training accuracy keeps increasing but validation accuracy plateaus.
"""

import torch
import torch.nn as nn
import time


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one complete epoch (one pass through training data).

    Parameters:
    -----------
    model : nn.Module
        The neural network to train.

    loader : DataLoader
        Provides batches of (images, labels) from the training set.

    criterion : loss function (e.g., nn.CrossEntropyLoss)
        Computes how wrong the predictions are.
        CrossEntropyLoss works by:
        1. Applying softmax to convert raw scores to probabilities
        2. Computing -log(probability of the correct class)
        If the model is confident and correct, loss is low (~0).
        If the model is confident and WRONG, loss is very high.

    optimizer : torch.optim optimizer (e.g., Adam)
        Updates model parameters based on gradients.
        Adam is popular because it adapts the learning rate for each parameter.

    device : torch.device
        'cpu' or 'cuda'. Where to run computations.

    Returns:
    --------
    avg_loss : float
        Average loss across all batches in this epoch.
    accuracy : float
        Percentage of correctly classified training images (0-100).
    """
    # model.train() enables training-specific behaviors:
    # - Dropout randomly zeroes neurons (regularization)
    # - BatchNorm uses batch statistics (not running averages)
    model.train()

    running_loss = 0.0    # Sum of losses across all batches
    correct = 0           # Number of correctly classified images
    total = 0             # Total number of images processed

    for images, labels in loader:
        # Move data to the computation device (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)

        # --- FORWARD PASS ---
        # Feed images through the network to get predictions
        # outputs shape: (batch_size, 10) - raw scores for each class
        outputs = model(images)

        # Compute the loss: how wrong are we?
        loss = criterion(outputs, labels)

        # --- BACKWARD PASS ---
        # optimizer.zero_grad(): Reset all gradients to zero.
        # WHY: PyTorch ACCUMULATES gradients by default. If we don't reset,
        # gradients from the previous batch would be added to the current ones,
        # making updates incorrect.
        optimizer.zero_grad()

        # loss.backward(): Compute gradients via BACKPROPAGATION.
        # This calculates d(loss)/d(parameter) for every parameter in the model.
        # It answers: "If I increase this parameter slightly, how does the loss change?"
        loss.backward()

        # optimizer.step(): Update all parameters using the computed gradients.
        # For Adam optimizer: new_param = old_param - lr * adjusted_gradient
        # Adam adjusts the learning rate per-parameter based on historical gradients.
        optimizer.step()

        # --- TRACK METRICS ---
        running_loss += loss.item() * images.size(0)  # loss.item() gets the Python number

        # Get predictions: the class with the highest score
        # torch.max returns (max_values, indices). We want the indices.
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on a dataset (validation or test set).

    This is similar to training but:
    1. We do NOT compute gradients (saves memory and time)
    2. We do NOT update parameters
    3. Dropout is disabled (all neurons active)
    4. BatchNorm uses running averages (not batch statistics)

    Parameters & Returns: same as train_one_epoch, plus:

    Returns:
    --------
    avg_loss : float
    accuracy : float
    all_predictions : list of int
        Predicted class for every image in the dataset.
    all_labels : list of int
        True class for every image in the dataset.
        (These are useful for computing confusion matrices later.)
    """
    # model.eval() switches to evaluation mode:
    # - Dropout is disabled (all neurons active)
    # - BatchNorm uses stored running mean/var instead of batch statistics
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    # torch.no_grad() disables gradient computation.
    # WHY: During evaluation, we only need predictions, not gradients.
    # This reduces memory usage and speeds up computation significantly.
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect all predictions and labels for later analysis
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, all_predictions, all_labels


def train_model(model, train_loader, test_loader, epochs=30, lr=0.001, device=None):
    """
    Complete training pipeline: train for multiple epochs and track progress.

    Parameters:
    -----------
    model : nn.Module
        The CNN to train.

    train_loader, test_loader : DataLoader
        Training and test data loaders.

    epochs : int
        Number of complete passes through the training data.
        More epochs = more learning, but diminishing returns and risk of overfitting.

    lr : float
        Initial learning rate for the Adam optimizer.
        0.001 is a safe default for Adam. Too high (0.1) causes instability.
        Too low (0.00001) causes very slow training.

    device : torch.device or None
        If None, automatically uses GPU if available, otherwise CPU.

    Returns:
    --------
    history : dict with keys:
        'train_loss'    : list of floats (loss per epoch)
        'train_acc'     : list of floats (accuracy per epoch)
        'test_loss'     : list of floats
        'test_acc'      : list of floats
        'epoch_times'   : list of floats (seconds per epoch)

        This history dict is used to plot training curves later.
    """
    # Automatically select device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on: {device}")
    print(f"Epochs: {epochs}, Learning Rate: {lr}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)

    # Move model to device
    model = model.to(device)

    # --- LOSS FUNCTION ---
    # CrossEntropyLoss: The standard loss for classification tasks.
    # It combines LogSoftmax + NLLLoss in one step.
    # It penalizes confident wrong predictions heavily.
    criterion = nn.CrossEntropyLoss()

    # --- OPTIMIZER ---
    # Adam (Adaptive Moment Estimation): An advanced optimizer that:
    # 1. Maintains per-parameter learning rates
    # 2. Uses momentum (averages of past gradients) for smoother updates
    # 3. Adapts step sizes based on gradient history
    # Generally works well "out of the box" without much tuning.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- LEARNING RATE SCHEDULER ---
    # ReduceLROnPlateau: Monitors test loss. If it hasn't improved for
    # 'patience' epochs, reduce the learning rate by factor.
    # Intuition: Big steps help early on, but we need smaller steps
    # to fine-tune as we get close to the optimal parameters.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',       # We want to MINIMIZE loss
        factor=0.5,       # Multiply lr by 0.5 when triggered
        patience=5,       # Wait 5 epochs before reducing
    )

    # Track metrics for each epoch
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'epoch_times': []
    }

    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate on test set
        test_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device
        )

        # Step the scheduler based on test loss
        scheduler.step(test_loss)

        epoch_time = time.time() - start_time

        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)

        # Track best accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # Print progress every 5 epochs (and first/last epoch)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.1f}% | "
                  f"Time: {epoch_time:.1f}s")

    print("-" * 60)
    print(f"Best test accuracy: {best_test_acc:.1f}%")
    total_time = sum(history['epoch_times'])
    print(f"Total training time: {total_time:.0f}s ({total_time/60:.1f} min)")

    return history
