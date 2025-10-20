# NCT: Neural Classification Tree

A hierarchical classification library implementing iterative training and refinement for image datasets.

## Overview

Neural Classification Tree (NCT) is a hierarchical approach to binary image classification. The method iteratively trains a tree structure where difficult-to-classify samples are routed to specialized nodes in deeper levels of the hierarchy. This approach can improve performance on challenging subgroups within datasets.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd NCT

# Install dependencies
pip install torch torchvision scikit-learn pandas numpy pillow
```

## Quick Start

```python
from nct import NCTClassifier
from torch.utils.data import Dataset

# Initialize the classifier
classifier = NCTClassifier(
    num_iterations=3,
    epochs_per_iteration=[20, 20, 20],
    backbone='resnet50',
    batch_size=32
)

# Train on PyTorch Dataset objects
classifier.fit(
    X_train=train_dataset,
    X_val=val_dataset
)

# Evaluate
results = classifier.evaluate(
    X_test=test_dataset,
    group_labels=test_groups
)
```

## Data Format

NCT requires PyTorch Dataset objects as input. Your dataset should implement:
- `__len__()`: Returns the number of samples
- `__getitem__(idx)`: Returns a tuple of (image_tensor, label)

Example:

```python
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = CustomDataset(image_paths, labels, transform=transform)
```

## Configuration

Customize training with configuration parameters:

```python
from nct import NCTClassifier

classifier = NCTClassifier(
    num_iterations=3,
    epochs_per_iteration=[30, 20, 15],
    batch_size=64,
    initial_base_lr=1e-5,
    initial_head_lr=1e-5,
    lr_decay_factor=2.0,
    backbone='resnet50',
    pretrained=True,
    weight_decay=1e-4,
    optimizer_type='adamw',
    use_class_weights=True,
    image_size=224,
    patience=15,
    phase1_epochs_ratio=0.5,
    seed=42,
    verbose=2
)
```

### Key Parameters

- `num_iterations`: Number of hierarchical levels (default: 3)
- `epochs_per_iteration`: List of training epochs per iteration (default: [20, 20, 20])
- `batch_size`: Batch size for training and inference (default: 32)
- `initial_base_lr`: Learning rate for backbone network (default: 1e-5)
- `initial_head_lr`: Learning rate for classification heads (default: 1e-5)
- `lr_decay_factor`: Factor to divide learning rate by each iteration (default: 2.0)
- `backbone`: CNN architecture ('resnet50', 'resnet18', etc.)
- `optimizer_type`: Optimizer type - 'adamw', 'adam', 'sgd', 'rmsprop' (default: 'adamw')
- `optimizer_config`: Optimizer-specific parameters as a dictionary (default: {})
- `use_class_weights`: Enable balanced class weighting for imbalanced data (default: True)
- `scheduler_type`: Learning rate scheduler - 'plateau', 'step', 'cosine', 'cosine_warmup', or 'none' (default: 'plateau')
- `scheduler_config`: Scheduler-specific parameters as a dictionary (default: {})
- `phase1_epochs_ratio`: Fraction of epochs for phase 1 frozen backbone training (default: 0.5)
- `patience`: Early stopping patience epochs (default: 15)
- `verbose`: Logging verbosity (0=silent, 1=basic, 2=detailed)

## Training

The `fit()` method trains the hierarchical classifier:

```python
classifier.fit(
    X_train=train_dataset,      # PyTorch Dataset
    X_val=val_dataset,          # PyTorch Dataset
    X_test=test_dataset,        # Optional: for per-iteration evaluation
    group_labels=group_labels,  # Optional: group labels for test set
    dataset_name="waterbird"    # Optional: dataset name for logging
)
```

### Two-Phase Training

Each iteration uses a two-phase training strategy:
1. **Phase 1**: Train classification heads with frozen backbone
2. **Phase 2**: Fine-tune the entire model (backbone + heads)

The split between phases is controlled by `phase1_epochs_ratio` (default: 0.5 for 50-50 split).

### Class Weighting

NCT supports balanced class weighting to handle imbalanced datasets. By default, classes are weighted using the formula: `total_samples / (num_classes * class_counts)`, which gives minority classes higher weight proportional to their rarity.

```python
# Use balanced class weighting (default behavior)
classifier = NCTClassifier(
    use_class_weights=True
)

# Disable class weighting (standard cross-entropy loss)
classifier = NCTClassifier(
    use_class_weights=False
)
```

When enabled, the weighting ensures that minority classes contribute more to the loss, helping the model learn from imbalanced data.

### Optimizer Configuration

NCT supports multiple optimizers with customizable parameters:

```python
# Default AdamW optimizer
classifier = NCTClassifier(
    optimizer_type='adamw'
)

# SGD with custom momentum and Nesterov acceleration
classifier = NCTClassifier(
    optimizer_type='sgd',
    optimizer_config={'momentum': 0.95, 'nesterov': True}
)

# Adam with custom betas
classifier = NCTClassifier(
    optimizer_type='adam',
    optimizer_config={'betas': (0.9, 0.999), 'eps': 1e-8}
)

# AdamW with custom parameters
classifier = NCTClassifier(
    optimizer_type='adamw',
    optimizer_config={'betas': (0.9, 0.999), 'amsgrad': True}
)
```

Available optimizers:
- `'adamw'` (default): AdamW optimizer (recommended for most cases)
- `'adam'`: Adam optimizer
- `'sgd'`: Stochastic Gradient Descent (supports momentum, nesterov)
- `'rmsprop'`: RMSprop optimizer

### Learning Rate Scheduling

NCT supports multiple learning rate schedulers to optimize training convergence:

```python
# Default plateau scheduler (reduces LR when validation loss plateaus)
classifier = NCTClassifier(
    scheduler_type='plateau',
    scheduler_config={'factor': 0.5, 'patience': 3}
)

# Cosine annealing with warmup (requires transformers library)
classifier = NCTClassifier(
    scheduler_type='cosine_warmup',
    scheduler_config={'num_warmup_steps': 200, 'num_training_steps': 5000}
)

# Step decay scheduler
classifier = NCTClassifier(
    scheduler_type='step',
    scheduler_config={'step_size': 10, 'gamma': 0.1}
)

# Cosine annealing
classifier = NCTClassifier(
    scheduler_type='cosine',
    scheduler_config={'T_max': 50, 'eta_min': 0}
)

# No scheduler
classifier = NCTClassifier(
    scheduler_type='none'
)
```

Available schedulers:
- `'plateau'` (default): Reduces LR when validation loss stops improving
- `'cosine_warmup'`: Cosine annealing with linear warmup (requires `pip install transformers`)
- `'step'`: Step decay at fixed intervals
- `'cosine'`: Cosine annealing
- `'none'`: No learning rate scheduling

## Evaluation

The `evaluate()` method provides comprehensive metrics:

```python
results = classifier.evaluate(
    X_test=test_dataset,
    group_labels=group_labels,
    dataset_name="waterbird"
)

# Access results
print(results['basic_metrics'])      # Accuracy, precision, recall, F1
print(results['group_metrics'])      # Per-group performance
print(results['node_analysis'])      # Hierarchical structure info
print(results['node_utilization'])   # Node usage statistics
```

### Metrics

**Basic Metrics:**
- Accuracy
- Precision, Recall, F1 Score
- Balanced Accuracy

**Group Metrics** (when group_labels provided):
- Per-group accuracy
- Worst-group accuracy
- Average group accuracy
- Group-wise confusion matrices

## How It Works

NCT builds a hierarchical classifier by iteratively expanding the number of classification heads that operate on shared backbone features:

1. **Iteration 1**: Train a model with **2 parallel heads** on the original binary labels
   - Head 0: predicts class 0
   - Head 1: predicts class 1

2. **Label Update**: Route samples to refined labels based on prediction correctness
   - For each sample, determine which head it was assigned to and whether the prediction was correct
   - Correctly classified samples → "easy" sub-node (label = old_label × 2)
   - Incorrectly classified samples → "hard" sub-node (label = old_label × 2 + 1)

3. **Iteration 2**: Train a **new model** with **4 parallel heads**
   - All 4 heads operate on the same backbone features (not stacked on previous heads)
   - Previous head weights are transferred to "easy" child heads (head 0→0, head 1→2)
   - New "hard" heads (1, 3) are initialized fresh to handle difficult samples

4. **Iteration 3+**: Continue doubling the number of heads (8, 16, ...)
   - Each iteration creates a new model with 2^(iteration+1) heads
   - All heads share the same backbone and operate in parallel

5. **Prediction**: Input passes through backbone once, then all heads predict in parallel
   - Each head outputs a score
   - The head with highest score determines the hierarchical node assignment
   - Final binary prediction: nodes 0-3 → class 0, nodes 4-7 → class 1 (for 8-head model)

**Architecture Evolution:**

```
Iteration 1: 2 heads
  Shared Backbone → [Head 0, Head 1]

Iteration 2: 4 heads
  Shared Backbone → [Head 0, Head 1, Head 2, Head 3]
                     (easy 0) (hard 0) (easy 1) (hard 1)

Iteration 3: 8 heads
  Shared Backbone → [Head 0, Head 1, ..., Head 7]
```


## Running an Example

The `src/nct_package_main.py` script demonstrates usage on the Waterbirds dataset:

```bash
python src/nct_package_main.py \
    --data_dir ./data \
    --num_iterations 3 \
    --epochs_per_iteration 20 20 20 \
    --initial_base_lr 1e-4 \
    --initial_head_lr 1e-3 \
    --batch_size 32 \
    --seed 42
```

### Command-Line Arguments

- `--data_dir`: Directory containing the dataset
- `--num_iterations`: Number of training iterations
- `--epochs_per_iteration`: Epochs for each iteration (space-separated)
- `--initial_base_lr`: Initial learning rate for backbone
- `--initial_head_lr`: Initial learning rate for heads
- `--lr_decay_factor`: LR decay factor between iterations
- `--batch_size`: Training batch size
- `--weight_decay`: Weight decay for optimizer
- `--seed`: Random seed for reproducibility
- `--phase1_epochs_ratio`: Ratio of epochs for frozen backbone phase

## Package Structure

```
nct/
├── core/
│   ├── classifier.py      # Main NCTClassifier API
│   └── config.py          # NCTConfig dataclass
├── data/
│   ├── datasets.py        # IterativeDataset implementation
│   └── loaders.py         # DataLoader creation utilities
├── models/
│   ├── hierarchical.py    # HierarchicalClassifier model
│   └── utils.py           # Model initialization utilities
├── training/
│   ├── trainer.py         # NCTTrainer training loops
│   ├── optimizer.py       # Optimizer and scheduler creation
│   └── loss.py            # Loss functions and class weights
├── evaluation/
│   ├── metrics.py         # Metric calculation functions
│   └── analysis.py        # Hierarchical analysis utilities
└── utils/
    ├── seed.py            # Random seed management
    ├── device.py          # Device detection
    └── logger.py          # Logging utilities
```

## API Reference

### NCTClassifier

**Main Methods:**

- `fit(X_train, X_val, X_test=None, group_labels=None, dataset_name="generic")`: Train the model
- `evaluate(X_test, group_labels=None, dataset_name="generic")`: Evaluate on test set
- `save_checkpoint(filepath)`: Save trained model to disk
- `from_checkpoint(filepath, device=None)`: Load trained model from disk (class method)

### NCTConfig

Configuration dataclass with validation and serialization support. All parameters can be passed directly to `NCTClassifier.__init__()` or accessed via `classifier.config`.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- scikit-learn
- pandas
- numpy
- Pillow

## License

-

## Citation

-
