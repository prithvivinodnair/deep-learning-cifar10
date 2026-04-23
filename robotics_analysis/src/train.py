"""
train.py - Training and Evaluation Functions for Grasp Classification

This module contains the training loop and evaluation logic, following
the same patterns as our CIFAR-10 training code but adapted for:
- Weighted CrossEntropyLoss (handles class imbalance)
- Binary classification (2 classes instead of 10)
- 224x224 input crops instead of 32x32
- Early stopping to prevent overfitting
- AUC-ROC tracking as primary metric

THE TRAINING PROCESS:
1. Feed a batch of grasp crops through the network -> get predictions
2. Compare predictions to true labels -> compute the WEIGHTED loss
   (misclassifying a bad grasp costs ~1.76x more than misclassifying a good one)
3. Compute gradients via backpropagation
4. Update parameters with the optimizer
5. Repeat for all batches = one epoch
6. Evaluate on the test set (without resampling or weighting)
7. If test metric hasn't improved for `patience` epochs -> STOP and restore best weights
8. Repeat for many epochs or until early stopping fires

WHY EARLY STOPPING:
Our ResNet-18 previously showed classic overfitting: test loss rose from 0.68
to 1.10 across 20 epochs while train accuracy climbed to 91.8%. The model was
memorising the training set. Early stopping watches the TEST (validation) metric
and halts training once it stops improving, keeping the best checkpoint.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


class EarlyStopping:
    """
    Stop training early when a monitored metric stops improving.

    Maintains the best model state seen during training and restores it when
    training ends (either by patience exhaustion or normal completion).

    Parameters:
    -----------
    patience : int
        Number of epochs to wait after the last improvement before stopping.
    min_delta : float
        Minimum change in the monitored metric to qualify as an improvement.
    mode : str
        'min' for loss-like metrics (lower is better),
        'max' for accuracy/AUC-like metrics (higher is better).
    verbose : bool
        Whether to print messages when improvement happens or patience increments.
    """

    def __init__(self, patience=7, min_delta=0.0005, mode='max', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > (self.best_score + self.min_delta)
        else:
            return score < (self.best_score - self.min_delta)

    def __call__(self, score, model, epoch):
        """
        Check whether training should continue, and snapshot best weights if so.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            return

        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"    [EarlyStopping] New best {self.mode}={score:.4f} at epoch {epoch}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"    [EarlyStopping] No improvement (best={self.best_score:.4f} "
                      f"@ epoch {self.best_epoch}). Patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best(self, model):
        """Load the best-seen weights back into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one complete epoch.

    Returns:
    --------
    avg_loss : float
    accuracy : float
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device, return_probs=False):
    """
    Evaluate the model on a dataset.

    If return_probs=True, also returns softmax probabilities so we can
    compute AUC-ROC and balanced accuracy every epoch (primary metrics
    for imbalanced data).

    Returns:
    --------
    If return_probs=False: (avg_loss, accuracy, preds, labels)
    If return_probs=True:  (avg_loss, accuracy, preds, labels, probs)
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probs = [] if return_probs else None

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

            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            if return_probs:
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    if return_probs:
        all_probs = np.concatenate(all_probs, axis=0)
        return avg_loss, accuracy, all_predictions, all_labels, all_probs

    return avg_loss, accuracy, all_predictions, all_labels


def evaluate_with_probs(model, loader, device=None):
    """
    Evaluate and return predictions + class probabilities.

    Returns:
    --------
    accuracy : float
    all_predictions : list of int
    all_labels : list of int
    all_probabilities : numpy.ndarray, shape (N, 2)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    _, accuracy, preds, labels, probs = evaluate(
        model, loader, criterion, device, return_probs=True
    )
    return accuracy, preds, labels, probs


def train_model(model, train_loader, test_loader, epochs=30, lr=0.001, device=None,
                criterion=None, optimizer=None, scheduler=None,
                early_stopping=True, patience=7, monitor='auc_roc',
                verbose=True):
    """
    Complete training pipeline with early stopping and AUC-ROC tracking.

    Parameters:
    -----------
    model : nn.Module
    train_loader, test_loader : DataLoader
    epochs : int
        Maximum number of training epochs (early stopping may end sooner).
    lr : float
        Initial learning rate.
    device : torch.device or None
        Auto-detects GPU if None.
    criterion : loss function or None
    optimizer : optimizer or None
    scheduler : LR scheduler or None
    early_stopping : bool
        Whether to use early stopping.
    patience : int
        Epochs without improvement before stopping.
    monitor : str
        Metric to monitor. One of:
        - 'auc_roc'           -> higher is better (recommended, primary metric)
        - 'balanced_accuracy' -> higher is better
        - 'test_loss'         -> lower is better
        - 'test_acc'          -> higher is better
    verbose : bool
        Whether to print per-epoch summaries.

    Returns:
    --------
    history : dict
        Keys: 'train_loss', 'train_acc', 'test_loss', 'test_acc',
              'test_auc_roc', 'test_balanced_acc', 'epoch_times',
              'best_epoch', 'best_score', 'monitor', 'stopped_early'.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(f"Training on: {device}")
        print(f"Max epochs: {epochs} | LR: {lr} | "
              f"Early stopping: {early_stopping} (patience={patience}, monitor={monitor})")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,}")
        print("-" * 78)

    model = model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        # Only optimize parameters that require grad (supports frozen layers)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

    # Early stopping monitor setup
    monitor_mode = 'min' if monitor == 'test_loss' else 'max'
    es = EarlyStopping(patience=patience, mode=monitor_mode, verbose=verbose) if early_stopping else None

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'test_auc_roc': [], 'test_balanced_acc': [],
        'epoch_times': [],
        'best_epoch': 0, 'best_score': None,
        'monitor': monitor, 'stopped_early': False,
    }

    test_criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # --- TRAIN ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # --- EVAL (always compute probabilities so we can track AUC-ROC) ---
        test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
            model, test_loader, test_criterion, device, return_probs=True
        )

        # Probability-based metrics (primary for imbalanced data)
        try:
            test_auc = roc_auc_score(test_labels, test_probs[:, 1])
        except ValueError:
            test_auc = float('nan')
        test_bal_acc = balanced_accuracy_score(test_labels, test_preds)

        # --- SCHEDULER STEP ---
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_loss)
        else:
            scheduler.step()

        epoch_time = time.time() - start_time

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_auc_roc'].append(test_auc)
        history['test_balanced_acc'].append(test_bal_acc * 100)  # store as %
        history['epoch_times'].append(epoch_time)

        if verbose and (epoch == 1 or epoch % 2 == 0 or epoch == epochs):
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Tr Loss {train_loss:.3f} Acc {train_acc:5.1f}% | "
                  f"Te Loss {test_loss:.3f} Acc {test_acc:5.1f}% | "
                  f"AUC {test_auc:.3f} | BalAcc {test_bal_acc*100:5.1f}% | "
                  f"{epoch_time:.1f}s")

        # --- EARLY STOPPING CHECK ---
        if es is not None:
            if monitor == 'auc_roc':
                score = test_auc
            elif monitor == 'balanced_accuracy':
                score = test_bal_acc
            elif monitor == 'test_loss':
                score = test_loss
            elif monitor == 'test_acc':
                score = test_acc
            else:
                raise ValueError(f"Unknown monitor: {monitor}")

            es(score, model, epoch)
            if es.early_stop:
                if verbose:
                    print(f"\n[EarlyStopping] Stopped at epoch {epoch}. "
                          f"Best {monitor}={es.best_score:.4f} at epoch {es.best_epoch}.")
                history['stopped_early'] = True
                break

    # Restore best model weights if early stopping was used
    if es is not None:
        model = es.restore_best(model)
        history['best_epoch'] = es.best_epoch
        history['best_score'] = es.best_score
        if verbose:
            print(f"[EarlyStopping] Restored best weights from epoch {es.best_epoch} "
                  f"({monitor}={es.best_score:.4f})")
    else:
        # Without early stopping, best epoch = the one with best monitor metric
        best_idx = int(np.argmax(history['test_auc_roc']))
        history['best_epoch'] = best_idx + 1
        history['best_score'] = history['test_auc_roc'][best_idx]

    if verbose:
        print("-" * 78)
        total_time = sum(history['epoch_times'])
        print(f"Total training time: {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"Best epoch: {history['best_epoch']} | "
              f"Best {monitor}: {history['best_score']:.4f}")

    return history
