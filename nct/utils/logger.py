"""Logging utilities for NCT training."""

import time
from typing import Optional


class NCTLogger:
    """Logger for NCT training with different verbosity levels."""

    def __init__(self, verbose: int = 1):
        """
        Initialize logger.

        Args:
            verbose: 0=silent, 1=progress, 2=detailed
        """
        self.verbose = verbose
        self.start_time = None
        self.iteration_start_time = None

    def training_start(self, config: dict):
        """Print training start message."""
        if self.verbose == 0:
            return

        self.start_time = time.time()

        if self.verbose >= 1:
            print("\nNCT Classifier Training")
            print("=" * 50)
            print(f"Iterations: {config['num_iterations']}  |  "
                  f"Backbone: {config['backbone']}  |  "
                  f"Device: {config.get('device', 'auto')}")
            print()

    def iteration_start(self, iteration: int, total_iterations: int, num_nodes: int, epochs: int, class_weights=None):
        """Print iteration start message."""
        if self.verbose == 0:
            return

        self.iteration_start_time = time.time()

        if self.verbose == 1:
            print(f"Iteration {iteration}/{total_iterations} [{num_nodes} nodes, {epochs} epochs]")
        elif self.verbose >= 2:
            print(f"\n--- Iteration {iteration}/{total_iterations}: Training with {num_nodes} nodes for {epochs} epochs ---")
            if class_weights is not None:
                import numpy as np
                # Convert to numpy safely
                if hasattr(class_weights, 'cpu'):
                    weights_array = class_weights.cpu().numpy()
                elif hasattr(class_weights, 'numpy'):
                    weights_array = class_weights.numpy()
                else:
                    weights_array = np.array(class_weights)
                print(f"  Class Weights: {np.round(weights_array, 2)}")

    def phase_start(self, phase_name: str, phase_num: int, epochs: int):
        """Print phase start message."""
        if self.verbose == 0:
            return

        if self.verbose == 1:
            phase_label = "frozen backbone" if phase_num == 1 else "full model"
            print(f"  Phase {phase_num} ({phase_label}): {epochs} epochs", end='', flush=True)
        elif self.verbose >= 2:
            print(f"  {phase_name}")

    def epoch_update(self, epoch: int, total_epochs: int, train_loss: float, val_loss: float):
        """Print epoch update (only for verbose=2)."""
        if self.verbose >= 2:
            print(f"    Epoch {epoch}/{total_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    def early_stop(self, phase_name: str, phase_num: int):
        """Print early stopping message."""
        if self.verbose == 0:
            return

        if self.verbose == 1:
            print(" → early stop", end='', flush=True)
        elif self.verbose >= 2:
            print(f"    Early stopping triggered in {phase_name}.")

    def phase_end(self, phase_num: int, completed_all: bool = False):
        """Print phase end message (only for verbose=1)."""
        if self.verbose == 1:
            if completed_all:
                print(" → completed all epochs")
            else:
                print()  # New line after early stop or normal completion

    def weight_transfer(self):
        """Print weight transfer message."""
        if self.verbose >= 1:
            print("  Transferring weights from previous iteration")

    def iteration_end(self):
        """Print iteration end message."""
        if self.verbose >= 1 and self.iteration_start_time:
            duration = time.time() - self.iteration_start_time
            if self.verbose == 1:
                print(f"  ✓ Completed in {duration/60:.1f} min\n")
            # For verbose=2, just add extra spacing
            elif self.verbose >= 2:
                print()

    def training_end(self, test_accuracy: Optional[float] = None, worst_group_acc: Optional[float] = None):
        """Print training end message."""
        if self.verbose == 0:
            return

        if self.start_time:
            total_duration = time.time() - self.start_time

            if self.verbose >= 1:
                print(f"\nTraining Complete ({total_duration/60:.1f} min total)")
                if test_accuracy is not None:
                    metrics_str = f"Final Test Accuracy: {test_accuracy:.2%}"
                    if worst_group_acc is not None:
                        metrics_str += f" | Worst Group: {worst_group_acc:.2%}"
                    print(metrics_str)
                print()

    def log(self, message: str, level: int = 1):
        """
        Generic log message.

        Args:
            message: Message to print
            level: Minimum verbosity level to print (default 1)
        """
        if self.verbose >= level:
            print(message)
