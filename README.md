# Neural Classification Trees: Interpretable Hierarchical Classification for Robustness to Spurious Correlations

## Abstract

Deep neural networks trained with Empirical Risk Minimization (ERM) often exploit **spurious correlations**—statistical associations between input features and labels that hold in training data but fail to generalize. This leads to high average accuracy but poor performance on minority subgroups where the spurious correlation is absent. We propose **Neural Classification Trees (NCT)**, an interpretable hierarchical classification framework that iteratively partitions samples into **Easy** and **Difficult** nodes based on prediction correctness. Samples the model struggles with are routed to specialized classification heads that learn to handle them. The resulting tree structure provides natural interpretability: a sample's path through the hierarchy (e.g., "Waterbird → Difficult → Easy") reveals its classification difficulty across training iterations. Our tree architecture employs **auxiliary losses** on parent nodes to prevent catastrophic forgetting while children specialize, and **two-phase training** to balance representation learning with classifier adaptation. Experiments on spurious correlation benchmarks (Waterbirds, CelebA, ISIC, Colored MNIST) demonstrate improved worst-group accuracy while maintaining interpretability.

---

## 1. Introduction

### 1.1 The Problem of Spurious Correlations

Consider a bird classification dataset where 95% of waterbirds appear on water backgrounds and 95% of landbirds appear on land backgrounds. A standard neural network learns to use the background as a predictive shortcut—it's statistically valid on the training distribution. However, this model catastrophically fails on the 5% of waterbirds photographed on land, achieving near-zero accuracy on this minority subgroup.

This phenomenon generalizes across domains:
- **Medical imaging**: Benign skin lesions often have colored ruler markings in training data
- **Facial recognition**: Hair color correlates with gender in celebrity datasets
- **Object recognition**: Object class correlates with typical background or context

The core issue is that ERM optimizes for average accuracy, which can be achieved by exploiting spurious features. The model has no incentive to learn the true causal features when shortcuts exist.

### 1.2 Worst-Group Accuracy as the Robustness Metric

Let $\mathcal{G} = \{g_1, g_2, ..., g_K\}$ denote subgroups defined by combinations of the true label $y$ and a spurious attribute $a$. For binary classification with a binary spurious attribute, we have four groups:

| Group | Label ($y$) | Spurious Attribute ($a$) | Example (Waterbirds) |
|-------|-------------|--------------------------|----------------------|
| $g_0$ | 0 | 0 | Landbird on Land (majority) |
| $g_1$ | 0 | 1 | Landbird on Water (minority) |
| $g_2$ | 1 | 0 | Waterbird on Land (minority) |
| $g_3$ | 1 | 1 | Waterbird on Water (majority) |

**Worst-group accuracy** is defined as:

$$\text{WGA} = \min_{g \in \mathcal{G}} \mathbb{E}_{(x,y) \sim P_g}[\mathbf{1}[f(x) = y]]$$

This metric captures robustness to distribution shift: a model with high WGA correctly classifies samples regardless of spurious attributes.

### 1.3 Our Approach: Hierarchical Specialization

NCT addresses spurious correlations through **hierarchical specialization**. Rather than training a single classifier that must handle all samples equally, NCT identifies difficult samples and routes them to specialized classification heads. The key insight is:

> Samples that are misclassified likely rely on features the model hasn't learned. By creating dedicated classifiers for these samples, we can learn complementary feature representations.

The resulting tree structure provides interpretability: examining which samples end up in "Difficult" nodes reveals patterns the model struggles with—often corresponding to minority subgroups affected by spurious correlations.

---

## 2. Method

### 2.1 Overview

NCT builds a hierarchical tree of classifiers over $T$ iterations. At each iteration $t$:

1. Train a classifier with $C \cdot 2^{t-1}$ heads (where $C$ is the number of classes)
2. Evaluate predictions on training data
3. Route samples to **Easy** or **Difficult** child nodes based on correctness
4. Repeat with the expanded label space

After $T$ iterations, each original class has $2^{T-1}$ specialized nodes, forming a binary tree of depth $T-1$ per class.

### 2.2 Formal Problem Setup

**Input**: Training set $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^N$ where $y_i \in \{0, 1, ..., C-1\}$

**Goal**: Learn a hierarchical classifier $f_T: \mathcal{X} \rightarrow \{0, 1, ..., C \cdot 2^{T-1} - 1\}$ that maps inputs to leaf nodes, where leaf nodes can be decoded to class predictions.

**Notation**:
- $h_t$: Hierarchical label at iteration $t$, where $h_t \in \{0, 1, ..., C \cdot 2^{t-1} - 1\}$
- $\hat{h}_t(x)$: Predicted hierarchical label at iteration $t$
- $\text{class}(h_t)$: Function mapping hierarchical label to class: $\text{class}(h_t) = \lfloor h_t / 2^{t-1} \rfloor$

### 2.3 Iterative Training with Correctness-Based Routing

**Iteration 1**: Initialize hierarchical labels as class labels: $h_1^{(i)} = y_i$

Train classifier $f_1$ with $C$ heads using weighted binary cross-entropy:

$$\mathcal{L}_1 = \frac{1}{N} \sum_{i=1}^N w_{h_1^{(i)}} \cdot \text{BCE}(f_1(x_i), \text{onehot}(h_1^{(i)}))$$

where class weights $w_c = \frac{N}{C \cdot N_c}$ balance imbalanced classes.

**Label Update (Routing)**: After training iteration $t$, compute new hierarchical labels for iteration $t+1$:

$$h_{t+1}^{(i)} = 2 \cdot h_t^{(i)} + \mathbf{1}[\hat{h}_t(x_i) \neq h_t^{(i)}]$$

This formula routes:
- **Correctly classified samples** to the **Easy child**: $h_{t+1} = 2h_t$ (even)
- **Incorrectly classified samples** to the **Difficult child**: $h_{t+1} = 2h_t + 1$ (odd)

**Iteration $t > 1$**: Train classifier $f_t$ with $C \cdot 2^{t-1}$ heads on the updated labels $\{h_t^{(i)}\}$.

### 2.4 Node Structure and Interpretation

The hierarchical label encodes the complete classification history. For a 3-iteration binary classifier:

| Node Index | Binary | Path | Interpretation |
|------------|--------|------|----------------|
| 0 | 000 | Class 0 → Easy → Easy | Consistently correct |
| 1 | 001 | Class 0 → Easy → Difficult | Correct initially, then struggled |
| 2 | 010 | Class 0 → Difficult → Easy | Initially wrong, then corrected |
| 3 | 011 | Class 0 → Difficult → Difficult | Consistently difficult |
| 4 | 100 | Class 1 → Easy → Easy | Consistently correct |
| 5 | 101 | Class 1 → Easy → Difficult | Correct initially, then struggled |
| 6 | 110 | Class 1 → Difficult → Easy | Initially wrong, then corrected |
| 7 | 111 | Class 1 → Difficult → Difficult | Consistently difficult |

**Interpretability**: A sample landing in "Waterbird → Difficult → Difficult" was misclassified in both iterations 1 and 2. This pattern often corresponds to minority subgroup samples (e.g., waterbirds on land backgrounds) where spurious features dominate.

---

## 3. Tree Architecture with Auxiliary Loss

### 3.1 The Challenge of Hierarchical Weight Transfer

A naive approach would train independent classifiers at each iteration. However, this wastes learned representations and leads to inconsistent feature spaces. We propose a **tree architecture** where parent classifiers provide features for their children.

### 3.2 Parent Layer Architecture

Each classification head consists of four layers:

$$\text{Head}(z) = W_4 \cdot \text{Dropout}(\text{ReLU}(W_3 \cdot z))$$

where $z$ is the input features (backbone output or parent features), $W_3 \in \mathbb{R}^{d \times d_{\text{hidden}}}$, and $W_4 \in \mathbb{R}^{d_{\text{hidden}} \times 1}$.

When transitioning from iteration $t$ to $t+1$:
1. **Previous output heads become parent layers**: The complete 4-layer head is preserved
2. **First 3 layers provide features**: $\phi_{\text{parent}}(z) = \text{Dropout}(\text{ReLU}(W_3 \cdot z))$
3. **Fourth layer used for auxiliary loss**: $\hat{y}_{\text{parent}} = W_4 \cdot \phi_{\text{parent}}(z)$
4. **Children connect to parent features**: New heads take $\phi_{\text{parent}}(z)$ as input, not backbone features

This creates a tree where information flows: **Backbone → Parent Layer → Child Heads**

### 3.3 Auxiliary Loss: Preventing Catastrophic Forgetting

A critical challenge in hierarchical learning is **catastrophic forgetting**: as we train children to specialize, parent representations may drift, destabilizing the features that children depend on.

**Auxiliary loss** addresses this by maintaining parent classification accuracy during child training:

$$\mathcal{L}_{\text{aux}} = \text{BCE}(\hat{y}_{\text{parent}}, h_{t-1})$$

where $h_{t-1} = \lfloor h_t / 2 \rfloor$ is the parent's label (undoing the Easy/Difficult split).

**Why auxiliary loss is essential**:

1. **Representation stability**: Parent layers provide input features to children. Without auxiliary loss, gradient updates from children can corrupt these features.

2. **Hierarchical consistency**: A sample should be classified correctly at all levels of the hierarchy. Auxiliary loss enforces that parent decisions remain valid while children refine them.

3. **Regularization effect**: Auxiliary loss acts as implicit regularization, preventing children from overfitting to spurious patterns that contradict parent-level predictions.

4. **Immediate parent only**: We use only the immediate parent layer (not all ancestors) to avoid over-constraining the optimization. This balances stability with flexibility for specialization.

**Combined loss function**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + \lambda_{\text{aux}} \cdot \mathcal{L}_{\text{aux}} + \lambda_{\text{reg}} \cdot \mathcal{R}(\theta)$$

where $\lambda_{\text{aux}}$ controls auxiliary loss strength (default 0.3, recommended 0.3-0.6) and $\mathcal{R}(\theta)$ is weight regularization.

---

## 4. Two-Phase Training

### 4.1 Motivation

Transfer learning typically involves two stages: feature extraction (frozen backbone) and fine-tuning. NCT extends this to each iteration with **two-phase training**:

**Phase 1 (Head Training)**: Freeze backbone, train only classification heads
- Allows heads to adapt to the current label space
- Prevents early gradient noise from corrupting backbone representations
- Parent layers are also trained (they're part of the classification path)

**Phase 2 (Full Fine-tuning)**: Unfreeze backbone, train entire model
- Enables backbone to adapt features for the current iteration's task
- Lower learning rate for backbone (typically $10\times$ lower than heads)

### 4.2 Phase Ratio Schedule

The fraction of training spent in Phase 1 varies by iteration:

$$\text{phase1\_ratio} = [r_1, r_2, ..., r_T]$$

**Default schedule**: `[0.5, 0.5, 0.5]` (50% Phase 1, 50% Phase 2)

**Recommended for Waterbirds**: `[1.0, 0.5, 0.5]`
- **Iteration 1**: 100% Phase 1 (heads adapt to pretrained features)
- **Iteration 2+**: 50% Phase 1, 50% Phase 2 (balance adaptation and refinement)

### 4.3 Model Selection Criterion

Each phase has independent early stopping. Final model selection uses:

$$\text{criterion} = \begin{cases}
\text{validation loss} & \text{if } t = 1 \\
\text{worst-group accuracy} & \text{if } t > 1
\end{cases}$$

**Rationale**: Iteration 1 establishes baseline representations (optimize for overall fit). Subsequent iterations focus on minority subgroups (optimize for worst-group performance).

This behavior is controlled by `model_selection_criterion: "auto"`. Can be overridden to `"wga"` or `"val_loss"`.

---

## 5. Routing Strategies

### 5.1 Correctness-Based Routing (Default)

The simplest strategy routes purely based on prediction correctness:

$$\text{is\_difficult}(x_i) = \mathbf{1}[\hat{h}_t(x_i) \neq h_t^{(i)}]$$

**Advantages**: Simple, deterministic, directly targets misclassified samples.

### 5.2 Correctness-Certainty Routing (GMM-Based)

Some correctly classified samples may rely on spurious features and would fail under distribution shift. We identify these **uncertain correct** samples using prediction entropy:

$$H(x) = -\sum_j p_j(x) \log p_j(x)$$

where $p_j(x) = \sigma(f_t(x))_j$ are sigmoid probabilities.

**GMM Clustering on Correct Samples**:
1. Compute entropy for all correctly classified samples
2. Fit 1-component and 2-component Gaussian Mixture Models to entropy values
3. Use BIC (Bayesian Information Criterion) to determine if 2 clusters fit better
4. If 2 clusters exist, identify high-entropy cluster as "uncertain"
5. Route uncertain correct samples to Difficult nodes

$$\text{is\_difficult}(x_i) = \mathbf{1}[\hat{h}_t(x_i) \neq h_t^{(i)}] \lor (\text{correct}(x_i) \land \text{uncertain}(x_i))$$

**Intuition**: Correct predictions made with low confidence may be exploiting spurious correlations. By routing them to Difficult nodes, we force the model to learn more robust features.

---

## 6. Interpretability Analysis

### 6.1 Node Path Interpretation

Each leaf node's path provides a natural explanation:

```
Sample: Waterbird on Land Background
Path: Waterbird → Difficult → Easy
Interpretation:
  - Iteration 1: Misclassified (likely predicted Landbird due to land background)
  - Iteration 2: Correctly classified (specialized Difficult node learned to ignore background)
```

### 6.2 Attribution Visualization

We generate **Grad-CAM** and other attribution maps for samples at each node, revealing:
- **Easy nodes**: Typically focus on semantic features (bird shape, texture)
- **Difficult nodes**: May initially focus on spurious features, then shift to semantic features

Supported attribution methods:
- Layer-based: `layer_gradcam`, `layer_conductance`
- Pixel-based: `saliency`, `deeplift`, `guided_gradcam`, `occlusion`

### 6.3 Group Distribution Analysis

Analyzing which subgroups populate each node reveals spurious correlation patterns:

```
Node: Waterbird → Difficult → Difficult
Population: 127 samples
  - Waterbird on Water: 12 samples (9.4%)
  - Waterbird on Land: 115 samples (90.6%)  ← Minority subgroup dominates
```

This concentration of minority subgroup samples in Difficult nodes confirms that NCT successfully identifies samples affected by spurious correlations.

---

## 7. Datasets and Spurious Correlations

| Dataset | Task | Spurious Attribute | Correlation | Minority Groups |
|---------|------|-------------------|-------------|-----------------|
| **Waterbirds** | Bird type | Background | 95% | Landbird-on-Water, Waterbird-on-Land |
| **CelebA** | Hair color | Gender | 94% | Blond Male, Non-Blond Female |
| **ISIC** | Lesion type | Colored patches | Variable | Benign without patch |
| **Colored MNIST** | Digit class | Color | 99.5% | Color-mismatched samples |
| **UMNIST** | Digit group | Digit 8 | Imbalanced | Digit 8 in 5-9 class |

### 7.1 Waterbirds

Constructed from CUB-200-2011 birds on Places backgrounds. Groups:
- $g_0$: Landbird on Land (majority, ~3,500 training samples)
- $g_1$: Landbird on Water (minority, ~180 training samples)
- $g_2$: Waterbird on Land (minority, ~60 training samples)
- $g_3$: Waterbird on Water (majority, ~1,000 training samples)

### 7.2 CelebA

Celebrity face attributes with Blond_Hair as label and Male as spurious attribute:
- Blond males are extremely rare (~1% of blond samples)
- Models learn "blond → female" shortcut

### 7.3 ISIC

Skin lesion classification where benign lesions often have colored ruler patches:
- Models learn "colored patch → benign" shortcut
- Critical for medical applications where robustness is essential

### 7.4 Colored MNIST (CMNIST)

5-class MNIST with synthetic color correlation (99.5% in training, random in test):
- Tests robustness to strong spurious signals
- 5 classes: digit pairs (0,1), (2,3), (4,5), (6,7), (8,9)
- 5 colors as spurious attribute

---

## 8. Implementation Details

### 8.1 Architecture

**Backbone**: ResNet-50 (ImageNet pretrained) for natural images, LeNet-5 for MNIST variants

**Classification Heads**:
- Hidden dimension: 256 (default)
- Dropout: 0.3 (default)
- Architecture: `Linear(in_features, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, 1)`

### 8.2 Configuration

NCT uses YAML configuration files. Below are default values and recommended Waterbirds settings:

**Default Configuration** (`NCTConfig` defaults):
```yaml
# Architecture
num_iterations: 3
hierarchical_architecture: correct_label_routing
aux_loss_weight: 0.3

# Training
epochs_per_iteration: [50, 50, 50]  # Auto-generated if not specified
phase1_epochs_ratio: [0.5, 0.5, 0.5]  # Auto-generated if not specified
batch_size: 32

# Optimization
optimizer:
  type: adamw
  weight_decay: 1e-4
initial_base_lr: 1e-5
initial_head_lr: 1e-5
lr_decay_factor: 2.0

# Loss
use_class_weights: true
max_class_weight: null  # No clamping by default

# Early Stopping
phase1_patience: 10
phase2_patience: 10
model_selection_criterion: auto
```

**Recommended Waterbirds Configuration** (`script/configs/waterbird/waterbird_nct.yaml`):
```yaml
# Architecture
num_iterations: 3
hierarchical_architecture: correct_label_routing
aux_loss_weight: 0.6  # Higher for better parent stability

# Training
epochs_per_iteration: [100, 100, 100]
phase1_epochs_ratio: [1.0, 0.5, 0.5]  # Full Phase 1 for iteration 1
batch_size: 64

# Optimization
optimizer:
  type: adamw
  weight_decay: 1.0  # Strong regularization
initial_base_lr: 8e-6
initial_head_lr: 8e-6

# Loss
use_class_weights: true
max_class_weight: 20.0  # Prevent extreme weights

# Early Stopping
phase1_patience: 15
phase2_patience: 5
```

### 8.3 Class Weighting

To handle class imbalance from Easy/Difficult splits:

$$w_c = \min\left(\frac{N}{C \cdot N_c}, w_{\max}\right)$$

where $w_{\max}$ (recommended 20.0) prevents extreme weights for very rare classes.

---

## 9. Key Findings

1. **Auxiliary loss is critical**: Without auxiliary loss, child nodes lose stable parent features, degrading performance. Recommended $\lambda_{\text{aux}} \in [0.3, 0.6]$.

2. **Two-phase training improves stability**: Full fine-tuning from scratch leads to training instability; frozen backbone Phase 1 provides reliable initialization.

3. **Difficult nodes capture minority subgroups**: Analysis shows 80-90% of minority subgroup samples end up in Difficult nodes, confirming NCT's ability to identify challenging samples.

4. **Interpretability enables debugging**: Node paths reveal model failure modes, enabling targeted improvements.

5. **Higher aux_loss_weight for complex datasets**: Datasets with stronger spurious correlations benefit from higher auxiliary loss (0.5-0.6).

---

## 10. Usage

### Training

```bash
python src/train.py \
    --dataset waterbird \
    --preset script/configs/waterbird/waterbird_nct.yaml \
    --data_dir ./data \
    --checkpoint_dir outputs/checkpoints \
    --generate_attributions
```

### Python API

```python
from nct import NCTClassifier, NCTConfig

config = NCTConfig.from_yaml("script/configs/waterbird/waterbird_nct.yaml")
classifier = NCTClassifier(config=config)

# Train with group labels for WGA tracking
results = classifier.fit(
    X_train=train_dataset,
    X_val=val_dataset,
    X_test=test_dataset,
    group_labels=test_groups,
    dataset_name="waterbird",
    class_names=("Landbird", "Waterbird"),
)

# Evaluate
eval_results = classifier.evaluate(X_test=test_dataset, group_labels=test_groups)
print(f"WGA: {eval_results['group_metrics']['summary']['worst_group_accuracy']:.2%}")
```

### Attribution Visualization

```bash
python src/generate_attributions.py \
    --checkpoint outputs/checkpoints/final_model.pt \
    --data_dir ./data \
    --methods all_layer
```

### Hyperparameter Tuning

```bash
python src/tune/tune_broad.py \
    --dataset waterbird \
    --n_trials 100 \
    --data_dir ./data
```

---

## 11. Project Structure

```
nct/                           # Core package
├── core/
│   ├── classifier.py          # NCTClassifier: main API (fit, predict, evaluate)
│   └── config.py              # NCTConfig: all hyperparameters
├── models/
│   ├── hierarchical.py        # HierarchicalClassifier: backbone + heads + parent layers
│   └── utils.py               # initialize_or_update_model: weight transfer
├── training/
│   ├── trainer.py             # NCTTrainer: two-phase training loop
│   ├── loss.py                # weighted_bce_loss, compute_auxiliary_loss
│   └── optimizer.py           # Optimizer/scheduler creation
├── evaluation/
│   ├── metrics.py             # calculate_group_metrics, WGA, AUROC
│   └── analysis.py            # get_node_label, analyze_hierarchical_nodes
├── visualization/
│   └── attribution_captum.py  # CaptumAttribution: Grad-CAM, saliency
├── data/
│   ├── configs.py             # DATASET_REGISTRY: dataset configurations
│   ├── datasets.py            # IterativeDataset, GenericImageDataset
│   └── preprocessing.py       # load_waterbird_metadata, load_celeba_metadata, etc.
└── utils/
    ├── checkpoints.py         # CheckpointManager: save/load models
    ├── hierarchical.py        # get_num_nodes, node_to_class, get_node_label
    └── logger.py              # NCTLogger: training logging

src/                           # Entry points
├── train.py                   # Main training script
├── generate_attributions.py   # Standalone attribution generation
└── tune/                      # Optuna hyperparameter search
    ├── tune_broad.py          # Broad search
    ├── tune_narrow.py         # Narrow search around good configs
    └── tune_erm.py            # ERM baseline tuning

script/                        # Configs and job scripts
├── configs/                   # YAML configs per dataset
│   ├── waterbird/
│   ├── celeba/
│   ├── isic/
│   └── ...
└── *.sh                       # SLURM job scripts
```

---

## 12. Citation

```bibtex
@inproceedings{nct2025,
  title={Neural Classification Trees: Interpretable Hierarchical Classification for Robustness to Spurious Correlations},
  author={},
  booktitle={},
  year={2025}
}
```

---

## 13. Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- torchvision
- scikit-learn
- captum (attribution visualization)
- wandb (experiment tracking, optional)
- optuna (hyperparameter tuning, optional)

```bash
pip install -r requirements.txt
```

For Compute Canada:
```bash
./script/setup_env.sh
module load python/3.11.5
source ~/envs/nct_env/bin/activate
```
