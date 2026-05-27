# Neural Classification Trees (NCT)

An interpretable hierarchical classifier that improves worst-group accuracy on datasets with spurious correlations.

Standard models trained with ERM exploit spurious shortcuts (for example, classifying a waterbird by its water background) and fail on minority subgroups where the shortcut breaks. NCT trains iteratively. After each round it routes correctly classified samples to an *Easy* child node and misclassified samples to a *Difficult* child node, then trains specialized heads on each. The Difficult nodes collect the hard, minority-subgroup samples, which improves worst-group accuracy and makes each prediction interpretable by the path it took through the tree.

## How it works

NCT runs for `T` iterations. Each iteration trains one head per node, predicts the training set, and splits every node into two children based on prediction correctness:

- correct goes to the *Easy* child (`new_label = 2 * label`)
- wrong goes to the *Difficult* child (`new_label = 2 * label + 1`)

After `T` iterations each original class owns `2^(T-1)` leaf nodes, and a sample's leaf encodes its full history:

| Leaf path | Meaning |
|-----------|---------|
| Waterbird, Easy, Easy | correct in both iterations |
| Waterbird, Easy, Difficult | correct first, then struggled |
| Waterbird, Difficult, Easy | wrong first, later corrected |
| Waterbird, Difficult, Difficult | consistently hard (usually a minority subgroup) |

In the default tree mode (`hierarchical_architecture: correct_label_routing`) each iteration's heads become parent layers that feed features to their children, with an auxiliary loss keeping parent predictions stable while children specialize. Training is two-phase per iteration: frozen backbone first to train the heads, then full fine-tuning.

## Installation

```bash
git clone <repo-url>
cd Neural-Classification-Trees
pip install -r requirements.txt
```

Core dependencies: PyTorch, torchvision, scikit-learn, captum (attributions), wandb (optional logging).

## Datasets

| Dataset | Task | Spurious attribute | Minority groups |
|---------|------|--------------------|-----------------|
| `waterbird` | Landbird vs. Waterbird | Background (land/water) | Landbird-on-Water, Waterbird-on-Land |
| `celeba` | Blond vs. Not-Blond | Gender | Blond Male |
| `isic` | Benign vs. Malignant | Colored patches | Benign without patch |
| `cmnist` | 5 digit-pair classes | Color (99.5% correlated) | Color-mismatched samples |
| `umnist` | Digit 0-4 vs. 5-9 | Digit 8 | Digit 8 in the 5-9 class |

Each dataset lives in its own subdirectory under `--data_dir` (default `./data`). CMNIST, UMNIST, and ISIC can be generated or downloaded with the provided scripts:

```bash
python dataset_download/download_cmnist.py --output_dir ./data/cmnist
python dataset_download/download_umnist.py --output_dir ./data/umnist
python dataset_download/download_isic_dataset.py --output_dir ./data/isic
```

Waterbirds and CelebA use the standard public releases. The loaders in [nct/data/preprocessing.py](nct/data/preprocessing.py) show the exact files each dataset expects.

## Quick start

Train on Waterbirds and generate attribution maps:

```bash
python src/train.py \
    --dataset waterbird \
    --preset script/configs/waterbird/waterbird.yaml \
    --data_dir ./data \
    --checkpoint_dir outputs/checkpoints \
    --generate_attributions
```

Or use the wrapper, which fills in the preset and output paths:

```bash
bash script/train.sh waterbird            # bash script/train.sh <dataset> [seed] [data_dir]
```

Each dataset has a single tuned preset at `script/configs/<dataset>/<dataset>.yaml` (`waterbird`, `celeba`, `isic`, `cmnist`, `umnist`).

## Python API

The package exposes two public classes: `NCTClassifier` (the model) and `NCTConfig` (hyperparameters).

`NCTClassifier` methods:

| Method | Description |
|--------|-------------|
| `fit(X_train, X_val, X_test=None, group_labels=None, dataset_name=..., class_names=...)` | Train the tree across all iterations. The `X_*` arguments are torch `Dataset` objects. |
| `predict(X)` | Class labels as a NumPy array of shape `(n,)`. |
| `predict_proba(X)` | Class probabilities, shape `(n, num_classes)`. |
| `evaluate(X_test, group_labels=None, ...)` | Metrics dict: accuracy, F1, AUROC, worst-group accuracy, per-node analysis. |
| `generate_attribution_visualizations(X_test, save_dir, methods=...)` | Save Grad-CAM and other attribution maps per node. |
| `save_checkpoint(path)` | Save model, config, and per-iteration results to one file. |
| `NCTClassifier.from_checkpoint(path, device=None)` | Load a fitted classifier back. |

`NCTConfig` holds the hyperparameters: build it with `NCTConfig.from_yaml(path)` or `NCTConfig(**kwargs)`, and inspect it with `config.to_dict()`.

```python
from nct import NCTClassifier, NCTConfig

config = NCTConfig.from_yaml("script/configs/waterbird/waterbird.yaml")
clf = NCTClassifier(config=config)

clf.fit(
    X_train=train_dataset,
    X_val=val_dataset,
    X_test=test_dataset,
    group_labels=test_groups,         # enables worst-group accuracy
    dataset_name="waterbird",
    class_names=("Landbird", "Waterbird"),
)

results = clf.evaluate(X_test=test_dataset, group_labels=test_groups)
print(f"WGA: {results['group_metrics']['summary']['worst_group_accuracy']:.2%}")

clf.save_checkpoint("outputs/checkpoints/final_model.pt")
```

`src/train.py` wires this together with the dataset loaders in `nct/data`, and is a good reference for the full training and evaluation flow.

## Using your own dataset

`fit`, `predict`, and `evaluate` accept any PyTorch `Dataset` whose `__getitem__` returns an `(image, label)` pair, with integer labels in `0 .. num_classes-1`. A few things to handle yourself:

- **Preprocessing.** NCT does not transform inputs. Your `Dataset` must return model-ready tensors of shape `(C, H, W)`, already resized and normalized (apply a transform inside the `Dataset`). See [nct/data/transforms.py](nct/data/transforms.py) for the transforms used by the built-in datasets.
- **Config to match your data.** Set `num_classes`, `num_channels` (3 for RGB, 1 for grayscale), `image_size`, and `backbone` (`resnet50`/`resnet18`/`resnet34` for natural images, `lenet5`/`lenet4` for small grayscale).
- **Worst-group accuracy.** Pass `group_labels`, an array aligned to the test set giving each sample a subgroup id (used only for evaluation).

```python
clf = NCTClassifier(num_classes=2, backbone="resnet50", image_size=224, num_channels=3)
clf.fit(X_train=my_train, X_val=my_val, X_test=my_test, group_labels=my_test_groups)
```

## Interpretability and attributions

Generate Grad-CAM and other attribution maps for each node from a saved checkpoint:

```bash
python src/generate_attributions.py \
    --checkpoint outputs/checkpoints/final_model.pt \
    --data_dir ./data \
    --methods all_layer        # or: layer_gradcam, saliency, deeplift, all, all_pixel
```

Comparing Easy and Difficult node attributions shows which features each specialized head relies on.

## Configuration

All hyperparameters live in `NCTConfig` ([nct/core/config.py](nct/core/config.py)) and load from YAML. Key fields:

```yaml
hierarchical_architecture: correct_label_routing   # tree mode ("none" = parallel heads)
num_iterations: 3
aux_loss_weight: 0.6                                # parent-stability auxiliary loss (0 disables)
phase1_epochs_ratio: [1.0, 0.5, 0.5]                # frozen-backbone fraction per iteration
model_selection_criterion: auto                     # val_loss for iter1, worst-group acc after
use_class_weights: true
max_class_weight: 20.0
sampling_strategy: none                             # none | upsample | downsample | geomean
min_samples_per_node: 20                            # merge tiny Difficult nodes into the Easy sibling
```

CLI flags on `src/train.py` can override individual fields for ablations (for example, `--aux_loss_weight`, `--sampling_strategy`).

## Repository structure

```
nct/                        Core package
  core/
    classifier.py           NCTClassifier: main API (fit / predict / evaluate)
    config.py               NCTConfig: all hyperparameters (YAML-loadable)
  models/
    hierarchical.py         HierarchicalClassifier: backbone + heads + parent layers
    utils.py                Weight transfer between iterations
  training/
    trainer.py              NCTTrainer: two-phase training loop
    loss.py                 Weighted BCE, class weights, auxiliary loss
    optimizer.py            Optimizer and scheduler construction
  evaluation/
    metrics.py              Worst-group accuracy, AUROC, pseudo-label metrics
    analysis.py             Node labelling and per-node analysis
  visualization/
    attribution_captum.py   Grad-CAM and other Captum attributions
  data/
    configs.py              DATASET_REGISTRY: per-dataset config and group definitions
    datasets.py             IterativeDataset, GenericImageDataset
    preprocessing.py        Per-dataset metadata loaders
  utils/                    Checkpoints, device, seeding, logging

src/
  train.py                  Main training entry point
  generate_attributions.py  Standalone attribution generation

script/
  configs/                  YAML presets per dataset
  train.sh                  Convenience training wrapper
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{garg2026discovering,
  title     = {Discovering Latent Groups for Robust Classification},
  author    = {Garg, Ankur and A\"ivodji, Ulrich and Ebrahimi Kahou, Samira and Michalski, Vincent},
  year      = {2026}
}
```
