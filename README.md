# NCT: Neural Classification Tree Library

A hierarchical classification library for image datasets using iterative training and refinement.

## Overview

The Neural Classification Tree (NCT) library provides a hierarchical approach to image classification that iteratively refines predictions by building a tree-like structure. This approach is particularly useful for:

- **Handling challenging samples**: Difficult-to-classify samples get specialized attention in deeper parts of the hierarchy
- **Improving fairness**: Better performance across different subgroups/demographics
- **Interpretable predictions**: Clear decision paths through the hierarchical structure

## Key Features

- **🎯 Generic**: Works with any image classification dataset
- **🔧 Easy to use**: Scikit-learn-like API (`fit`, `predict`, `evaluate`)
- **📊 Comprehensive evaluation**: Built-in fairness and group-wise metrics
- **🏗️ Modular design**: Clean, extensible architecture
- **⚡ Efficient**: Two-phase training with backbone freezing and fine-tuning
- **🎨 Flexible**: Support for multiple data formats and backbones

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

# Initialize the classifier
nct = NCTClassifier(
    num_iterations=3,
    epochs_per_iteration=[20, 20, 20],
    backbone='resnet50',
    batch_size=32
)

# Train on your data
nct.fit(train_images, train_labels, val_images, val_labels)

# Make predictions
predictions = nct.predict(test_images)
probabilities = nct.predict_proba(test_images)

# Evaluate performance
results = nct.evaluate(test_images, test_labels,
                      group_labels=test_groups)
```

## Supported Data Formats

The NCT library supports multiple input formats:

### 1. Directory Structure
```
data/
├── class_0/
│   ├── image1.jpg
│   └── image2.jpg
└── class_1/
    ├── image3.jpg
    └── image4.jpg
```

```python
nct.fit("path/to/data")
```

### 2. Pandas DataFrame
```python
import pandas as pd

df = pd.DataFrame({
    'image_path': ['img1.jpg', 'img2.jpg', ...],
    'label': [0, 1, ...]
})
nct.fit(df)
```

### 3. Arrays/Lists
```python
image_paths = ['img1.jpg', 'img2.jpg', ...]
labels = [0, 1, ...]
nct.fit(image_paths, labels)
```

### 4. PyTorch Tensors
```python
images = torch.randn(1000, 3, 224, 224)  # Example tensor
labels = torch.randint(0, 2, (1000,))
nct.fit(images, labels)
```

## Configuration

Customize the model with `NCTConfig`:

```python
from nct import NCTClassifier, NCTConfig

config = NCTConfig(
    num_iterations=3,
    epochs_per_iteration=[30, 20, 15],
    batch_size=64,
    initial_base_lr=1e-4,
    initial_head_lr=1e-3,
    lr_decay_factor=2.0,
    backbone='resnet50',
    pretrained=True,
    weight_decay=1e-4,
    optimizer_type='adamw',
    image_size=224,
    patience=15,
    verbose=True
)

nct = NCTClassifier(**config.to_dict())
```

## Architecture

The NCT library is organized into several modules:

```
nct/
├── core/           # Main API classes
├── data/           # Data handling and loading
├── models/         # Neural network architectures
├── training/       # Training loops and optimization
├── evaluation/     # Metrics and analysis
└── utils/          # Utilities (seeding, device management)
```

### Key Components

- **NCTClassifier**: Main API class with sklearn-like interface
- **HierarchicalClassifier**: Neural network with shared backbone + multiple heads
- **NCTTrainer**: Two-phase training (frozen backbone → full fine-tuning)
- **Evaluation suite**: Comprehensive metrics including fairness analysis

## How It Works

1. **Initial Training**: Train a 2-class classifier (binary split)
2. **Label Update**: Samples are routed to "easy" or "hard" branches based on correctness
3. **Hierarchical Expansion**: Each node splits into 2 children (easy/hard)
4. **Iterative Refinement**: Repeat for specified number of iterations
5. **Final Prediction**: Route through the tree to get final classification

```
Iteration 1:     [Root]
                /      \
         Class 0          Class 1

Iteration 2:     [Root]
                /      \
        [Class 0]      [Class 1]
         /     \        /     \
    C0-Easy  C0-Hard  C1-Easy  C1-Hard

... and so on
```

## Evaluation

The library provides comprehensive evaluation metrics:

```python
results = nct.evaluate(test_data, test_labels,
                      group_labels=group_labels,
                      group_names={0: "Group A", 1: "Group B", ...})

# Access different metrics
print(results['basic_metrics'])      # Accuracy, F1, etc.
print(results['group_metrics'])      # Per-group performance
print(results['node_analysis'])      # Hierarchical structure analysis
print(results['node_utilization'])   # How many nodes are used
```

## Examples

See `example_usage.py` for comprehensive examples including:

- Basic usage with dummy data
- Custom configuration
- Different data formats
- Model saving/loading
- Evaluation and analysis

Run the examples:
```bash
python example_usage.py
```

## API Reference

### NCTClassifier

**Methods:**
- `fit(X_train, y_train, X_val=None, y_val=None)`: Train the model
- `predict(X)`: Make binary predictions
- `predict_proba(X)`: Get prediction probabilities
- `evaluate(X_test, y_test, group_labels=None)`: Comprehensive evaluation
- `save_model(filepath)`: Save trained model
- `load_model(filepath)`: Load trained model

**Parameters:**
- `num_iterations`: Number of hierarchical levels (default: 3)
- `epochs_per_iteration`: Training epochs per iteration (default: [20, 20, 20])
- `backbone`: CNN backbone ('resnet50', 'resnet18', etc.)
- `batch_size`: Training batch size (default: 32)
- `initial_base_lr`: Learning rate for backbone (default: 1e-4)
- `initial_head_lr`: Learning rate for classification heads (default: 1e-3)
- `device`: Training device ('auto', 'cuda', 'cpu')

### NCTConfig

Configuration dataclass with validation and serialization support.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Citation

If you use this library in your research, please cite:

```bibtex
@software{nct_library,
  title={NCT: Neural Classification Tree Library},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```