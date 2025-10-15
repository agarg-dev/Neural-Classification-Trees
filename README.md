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

1. **Iteration 1**: Train a binary classifier on original labels
2. **Label Update**: Route samples based on prediction correctness
   - Correctly classified → "easy" branch (label 0)
   - Incorrectly classified → "hard" branch (label 1)
3. **Iteration 2+**: Train new classifiers on each branch
4. **Prediction**: Route test samples through the tree hierarchy

Example hierarchy structure:

```
Iteration 1:           [Root: 2 nodes]
                      /              \
               Class 0 (node 0)   Class 1 (node 1)

Iteration 2:           [Root: 4 nodes]
                      /              \
            [Class 0 Node]      [Class 1 Node]
               /      \            /      \
           Easy 0   Hard 0     Easy 1   Hard 1
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
- `save_model(filepath)`: Save trained model to disk
- `load_model(filepath)`: Load trained model from disk

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
