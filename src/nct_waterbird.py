# --- IMPORTS & GLOBAL CONFIGURATION ---
import argparse
import copy
import os
import random
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# Set the device for PyTorch operations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- ARGUMENT PARSING & SEEDING ---

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run iterative hierarchical training on Waterbird data.")
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory where the dataset is stored.')
    parser.add_argument('--num_iterations', type=int, default=3, help='Number of training iterations.')
    parser.add_argument('--epochs_per_iteration', type=int, nargs='+', default=[20, 20, 20], help='List of epochs for each iteration.')
    parser.add_argument('--initial_base_lr', type=float, default=1e-4, help='Initial learning rate for the CNN base.')
    parser.add_argument('--initial_head_lr', type=float, default=1e-3, help='Initial learning rate for the output heads.')
    parser.add_argument('--lr_decay_factor', type=float, default=2.0, help='Factor to DIVIDE base LR by between iterations.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for Adam optimizer.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    return parser.parse_args()


def set_seed(seed: int):
    """Sets the seed for all relevant random number generators to ensure reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Using deterministic algorithms can aid reproducibility but may impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- DATA HANDLING ---

def find_waterbirds_dataset(data_dir: str) -> str | None:
    """Checks for and returns the path to the local Waterbird dataset directory."""
    dataset_path = os.path.join(data_dir, "waterbird")
    if os.path.exists(dataset_path):
        print(f"Dataset found at: {dataset_path}")
        return dataset_path
    else:
        print(f"--- ERROR: Dataset not found at the expected path: {dataset_path} ---")
        return None


class WaterbirdDataset(Dataset):
    """Custom PyTorch Dataset for loading images and labels from the Waterbird dataset."""
    def __init__(self, metadata_df: pd.DataFrame, root_dir: str, transform=None):
        self.metadata = metadata_df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = os.path.join(self.root_dir, self.metadata.iloc[idx]['img_filename'])
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image not found at {img_path}")
            return torch.zeros(3, 224, 224), -1  # Return a placeholder for error handling

        label = self.metadata.iloc[idx]['y']
        if self.transform:
            image = self.transform(image)
        return image, label


class IterativeDataset(Dataset):
    """A wrapper dataset for iterative training, combining images with dynamic labels and weights."""
    def __init__(self, base_dataset: WaterbirdDataset, hier_labels: torch.Tensor):
        self.base_dataset = base_dataset
        self.hier_labels = hier_labels
        assert len(self.base_dataset) == len(self.hier_labels), "Base dataset and labels must have the same length."

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, _ = self.base_dataset[idx]
        hier_label = self.hier_labels[idx]
        return image, hier_label


def validate_dataset_integrity(data_bundle: Dict[str, Any]):
    """Validates that datasets respect split boundaries and report any integrity issues."""
    print("--- Dataset Integrity Validation ---")

    train_dataset = data_bundle["base_train_dataset"]
    val_dataset = data_bundle["base_val_dataset"]
    test_dataset = data_bundle["base_test_dataset"]

    # Check a few samples from each dataset to ensure they're accessing correct splits
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Verify split integrity by checking metadata
    train_splits = train_dataset.metadata['split'].unique()
    val_splits = val_dataset.metadata['split'].unique()
    test_splits = test_dataset.metadata['split'].unique()

    print(f"Train dataset contains splits: {train_splits} (should be [0])")
    print(f"Val dataset contains splits: {val_splits} (should be [1])")
    print(f"Test dataset contains splits: {test_splits} (should be [2])")

    # Validation passed if each dataset only contains its designated split
    integrity_check = (
        len(train_splits) == 1 and train_splits[0] == 0 and
        len(val_splits) == 1 and val_splits[0] == 1 and
        len(test_splits) == 1 and test_splits[0] == 2
    )

    if integrity_check:
        print("✓ Dataset integrity check PASSED - No cross-split contamination detected")
    else:
        print("✗ Dataset integrity check FAILED - Cross-split contamination detected!")

    print("--- End Validation ---\n")
    return integrity_check


def prepare_data(data_dir: str) -> Dict[str, Any]:
    """Loads metadata, defines transformations, and creates base datasets."""
    dataset_path = find_waterbirds_dataset(data_dir)
    if not dataset_path:
        raise FileNotFoundError("Waterbird dataset directory not found.")

    metadata_df = pd.read_csv(os.path.join(dataset_path, 'metadata.csv'))

    # Define standard transformations for training and validation/testing
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets for each split
    train_df = metadata_df[metadata_df['split'] == 0]
    val_df = metadata_df[metadata_df['split'] == 1]
    test_df = metadata_df[metadata_df['split'] == 2]

    data_bundle = {
        "base_train_dataset": WaterbirdDataset(train_df, dataset_path, train_transform),
        "base_val_dataset": WaterbirdDataset(val_df, dataset_path, val_test_transform),
        "base_test_dataset": WaterbirdDataset(test_df, dataset_path, val_test_transform),
        "y_train_t": torch.tensor(train_df['y'].values, dtype=torch.long),
        "y_val_t": torch.tensor(val_df['y'].values, dtype=torch.long),
        "y_test_t": torch.tensor(test_df['y'].values, dtype=torch.long),
    }

    # Create group labels for final evaluation (y*2 + place)
    place_test_t = torch.tensor(test_df['place'].values, dtype=torch.long)
    data_bundle["group_test_t"] = data_bundle["y_test_t"] * 2 + place_test_t

    # Validate dataset integrity to ensure no data leakage
    validate_dataset_integrity(data_bundle)

    return data_bundle


# --- MODEL ARCHITECTURE ---

class HierarchicalClassifier(nn.Module):
    """A neural network with a shared ResNet-50 feature extractor and multiple output heads."""
    def __init__(self, num_heads: int, freeze_base: bool = False):
        super().__init__()
        # Use a pre-trained ResNet-50 as the shared base
        self.base = models.resnet50(weights='IMAGENET1K_V1')
        num_features = self.base.fc.in_features
        self.base.fc = nn.Identity()  # Remove the original classifier

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        # Create a list of separate classifiers (heads), one for each hierarchical node
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 1)
            ) for _ in range(num_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base(x)
        outputs = [head(features) for head in self.output_heads]
        return torch.cat(outputs, dim=1)


def initialize_or_update_model(num_nodes: int, prev_model: nn.Module | None) -> nn.Module:
    """Initializes a new model or transfers weights from a previous one."""
    new_model = HierarchicalClassifier(num_heads=num_nodes).to(DEVICE)
    if prev_model:
        # Transfer the learned base weights
        new_model.base.load_state_dict(prev_model.base.state_dict())
        print("  Transferring weights from previous output heads...")
        # Transfer head weights to their new "easy" children
        for old_head_idx, old_head in enumerate(prev_model.output_heads):
            new_head_idx = old_head_idx * 2
            new_model.output_heads[new_head_idx].load_state_dict(old_head.state_dict())
    return new_model


# --- TRAINING & LOSS ---

def weighted_bce_loss(outputs: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """Calculates BCE loss with class-specific weights."""
    target_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).float()
    unreduced_loss = F.binary_cross_entropy_with_logits(outputs, target_one_hot, reduction='none')
    # Apply weights corresponding to each sample's target class
    weights_for_batch = class_weights[targets].unsqueeze(1)
    return (unreduced_loss * weights_for_batch).mean()


def _run_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, scaler: GradScaler, 
               class_weights: torch.Tensor, is_training: bool) -> float:
    """Runs a single epoch of training or validation."""
    model.train(is_training)
    total_loss = 0.0
    
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            if is_training:
                optimizer.zero_grad()

            with autocast(): # Mixed precision
                outputs = model(images)
                loss = weighted_bce_loss(outputs, labels, class_weights)
            
            if is_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            total_loss += loss.item()

    return total_loss / len(loader)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int, base_lr: float, head_lr: float, weight_decay: float,
                class_weights: torch.Tensor):
    """Trains the model using a two-phase approach (frozen base, then full fine-tuning)."""
    best_val_loss = float('inf')
    patience, patience_counter = 10, 0
    best_model_weights = copy.deepcopy(model.state_dict())
    scaler = GradScaler()
    
    # Validation uses un-weighted loss for fair comparison across iterations
    val_class_weights = torch.ones_like(class_weights)

    # --- Phase 1: Train with Frozen Base ---
    print("    Phase 1: Training output heads (frozen base).")
    for param in model.base.parameters(): param.requires_grad = False
    optimizer1 = optim.AdamW(model.output_heads.parameters(), lr=head_lr, weight_decay=weight_decay)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'min', factor=0.5, patience=3)
    
    phase1_epochs = epochs // 2
    for epoch in range(phase1_epochs):
        train_loss = _run_epoch(model, train_loader, optimizer1, scaler, class_weights, is_training=True)
        val_loss = _run_epoch(model, val_loader, None, scaler, val_class_weights, is_training=False)
        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler1.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("    Early stopping triggered in Phase 1.")
            model.load_state_dict(best_model_weights)
            break

    # --- Phase 2: Fine-tune Full Model ---
    print("    Phase 2: Fine-tuning the full model.")
    for param in model.base.parameters(): param.requires_grad = True
    optimizer2 = optim.AdamW([
        {'params': model.base.parameters(), 'lr': base_lr},
        {'params': model.output_heads.parameters(), 'lr': head_lr}
    ], weight_decay=weight_decay)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'min', factor=0.5, patience=3)


    for epoch in range(phase1_epochs, epochs):
        train_loss = _run_epoch(model, train_loader, optimizer2, scaler, class_weights, is_training=True)
        val_loss = _run_epoch(model, val_loader, None, scaler, val_class_weights, is_training=False)
        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler2.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("    Early stopping triggered in Phase 2.")
            break
            
    if best_model_weights:
        model.load_state_dict(best_model_weights)


# --- EVALUATION & ANALYSIS ---
@torch.no_grad()
def get_all_predictions(model: nn.Module, dataloader: DataLoader) -> torch.Tensor:
    """Helper function to get model predictions for an entire dataset."""
    model.eval()
    all_preds = []
    for images, _ in dataloader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
    return torch.cat(all_preds)



def analyze_final_nodes(predicted_hier_classes: torch.Tensor, true_binary_labels: torch.Tensor, 
                        group_labels: torch.Tensor, num_iterations: int, group_names: Dict[int, str]):
    """Provides a detailed performance breakdown for each final node in the hierarchy."""
    
    print("\n--- Detailed Analysis of Final Nodes ---")
    final_num_nodes = 2**num_iterations

    def get_path_description(node_index: int, num_iter: int) -> str:
        # The first split is always Landbird vs. Waterbird
        path = ["Land Bird" if node_index < (2**(num_iter - 1)) else "Water Bird"]
        # Subsequent splits are Easy vs. Difficult based on correctness
        for i in range(num_iter - 2, -1, -1):
            decision_bit = (node_index >> i) & 1
            path.append("Difficult" if decision_bit == 1 else "Easy")
        return " -> ".join(path)

    for node_idx in range(final_num_nodes):
        mask = (predicted_hier_classes == node_idx)
        population = mask.sum().item()

        path_str = get_path_description(node_idx, num_iterations)
        node_predicts_binary = 1 if node_idx >= final_num_nodes / 2 else 0
        prediction_str = "Water Bird" if node_predicts_binary == 1 else "Land Bird"
        
        print(f"\n────────────────────────────────────────\nNode {node_idx}: ({path_str})")
        print(f"  Final Prediction: '{prediction_str}'")

        if population == 0:
            print("  Population: 0 samples. Node is unused.")
            continue

        true_labels_in_node = true_binary_labels[mask]
        groups_in_node = group_labels[mask]
        accuracy = (true_labels_in_node == node_predicts_binary).float().mean().item()
        
        print(f"  Population: {population} samples")
        print(f"  Node Overall Accuracy: {accuracy * 100:.2f}%")
        
        print("  Ground Truth Breakdown by Group:")
        for group_id, group_name in group_names.items():
            group_mask = (groups_in_node == group_id)
            group_population = group_mask.sum().item()
            if group_population > 0:
                expected_label = 1 if group_id >= 2 else 0
                correctness = "CORRECTLY" if node_predicts_binary == expected_label else "INCORRECTLY"
                print(f"    - {group_name:<28}: {group_population:<4} samples. (Predicts {correctness})")



def evaluate_model_at_iteration(model: nn.Module, data_bundle: Dict, current_iteration_index: int, batch_size: int):
    """Performs evaluation on the test set and prints comprehensive results for a given iteration."""
    print(f"\n--- Evaluation on Test Set after Iteration {current_iteration_index + 1} ---")
    test_dataset = IterativeDataset(data_bundle["base_test_dataset"], torch.zeros_like(data_bundle["y_test_t"]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    predicted_hier_classes = get_all_predictions(model, test_loader)
    num_nodes_at_iteration = 2**(current_iteration_index + 1)
    predicted_binary = (predicted_hier_classes >= num_nodes_at_iteration / 2).long()

    y_test_t = data_bundle["y_test_t"]
    group_test_t = data_bundle["group_test_t"]
    
    # Overall Metrics
    final_accuracy = (predicted_binary == y_test_t).float().mean().item()
    final_f1_score = f1_score(y_test_t.numpy(), predicted_binary.numpy())
    print(f"Test Accuracy: {final_accuracy * 100:.2f}%")
    print(f"Test F1 Score: {final_f1_score:.4f}")

    # Group-Specific Performance
    print("\n--- Group-Specific Performance ---")
    group_names = {
        0: "Land Bird on Land (Easy)",
        1: "Land Bird on Water (Hard)",
        2: "Water Bird on Land (Hard)",
        3: "Water Bird on Water (Easy)"
    }
    group_accuracies = []
    for group_id in range(4):
        mask = (group_test_t == group_id)
        if mask.sum() > 0:
            acc = (predicted_binary[mask] == y_test_t[mask]).float().mean().item()
            group_accuracies.append(acc)
            print(f"  Accuracy for group '{group_names[group_id]}': {acc * 100:.2f}%")
    
    if group_accuracies:
        print(f"\nWorst-Group Accuracy: {min(group_accuracies) * 100:.2f}%")
    
    analyze_final_nodes(predicted_hier_classes, y_test_t, group_test_t, current_iteration_index + 1, group_names)


# --- MAIN ORCHESTRATION ---

def main():
    """Main function to run the entire iterative training and evaluation pipeline."""
    args = parse_args()
    set_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    print("\n--- Experiment Configuration ---")
    for key, value in sorted(vars(args).items()):
        print(f"  - {key:<25}: {value}")
    print(f"  - {'device':<25}: {DEVICE}")
    print("--------------------------------\n")

    # --- Data Loading and Initial Setup ---
    data_bundle = prepare_data(args.data_dir)
    current_train_labels = data_bundle["y_train_t"]
    current_val_labels = data_bundle["y_val_t"]
    current_base_lr = args.initial_base_lr
    model = None

    # --- Main Iterative Training Loop ---
    for i in range(args.num_iterations):
        num_nodes = 2**(i + 1)
        epochs_for_iter = args.epochs_per_iteration[i]
        print(f"\n--- Iteration {i+1}/{args.num_iterations}: Training with {num_nodes} nodes for {epochs_for_iter} epochs ---")
        
        # Calculate dynamic, inverse frequency class weights
        class_counts = torch.bincount(current_train_labels, minlength=num_nodes).float().clamp(min=1)
        class_weights = len(current_train_labels) / (num_nodes * class_counts)
        print(f"  Class Weights: {np.round(class_weights.numpy(), 2)}")
        
        # Initialize model and transfer weights from the previous iteration
        model = initialize_or_update_model(num_nodes, model)
        torch.compile(model)
        
        # Create datasets and loaders for the current iteration
        train_dataset = IterativeDataset(data_bundle["base_train_dataset"], current_train_labels)
        val_dataset = IterativeDataset(data_bundle["base_val_dataset"], current_val_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        # Train the model for one iteration
        train_model(
            model, train_loader, val_loader, 
            epochs=epochs_for_iter, 
            base_lr=current_base_lr, 
            head_lr=args.initial_head_lr, 
            weight_decay=args.weight_decay,
            class_weights=class_weights.to(DEVICE)
        )

        # Evaluate the model on the test set after the current iteration is complete.
        evaluate_model_at_iteration(model, data_bundle, current_iteration_index=i, batch_size=args.batch_size)


        # --- Update Labels for the next iteration ---
        if i < args.num_iterations - 1:
            print("\n  Updating hierarchical labels for the next iteration...")

            # Create a new, non-shuffled loader for evaluation to ensure prediction order matches label order.
            train_eval_loader = DataLoader(
                IterativeDataset(data_bundle["base_train_dataset"], current_train_labels),
                batch_size=args.batch_size, 
                shuffle=False,
                num_workers=8, 
                pin_memory=True
            )

            # Update training labels using the correct loader
            train_preds = get_all_predictions(model, train_eval_loader)
            is_correct_train = (train_preds == current_train_labels)
            current_train_labels = current_train_labels * 2 + (1 - is_correct_train.long())
            
            # Update validation labels
            val_preds = get_all_predictions(model, val_loader)
            is_correct_val = (val_preds == current_val_labels)
            current_val_labels = current_val_labels * 2 + (1 - is_correct_val.long())
            
            # Decay the learning rate for the base model
            current_base_lr /= args.lr_decay_factor

            
    print("\nExperiment complete.")


if __name__ == '__main__':
    main()
