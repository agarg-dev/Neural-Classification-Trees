# Neural Classification Trees (NCT)

Interpretable hierarchical classifier for image datasets with spurious correlations. NCT iteratively builds a tree of classifiers, routing difficult samples to specialized nodes to improve worst-group accuracy.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python src/train.py \
    --dataset waterbird \
    --preset configs/waterbird.yaml \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints \
    --generate_attributions
```

Supported datasets: `waterbird`, `celeba`, `isic`, `umnist`.

## Attribution Generation

Generate attribution visualizations from a saved checkpoint:

```bash
python src/generate_attributions.py \
    --checkpoint ./checkpoints/final_model.pt \
    --data_dir ./data \
    --methods all_layer
```

Available methods: `layer_gradcam`, `saliency`, `deeplift`, `guided_gradcam`, `occlusion`, or keywords `all`, `all_layer`, `all_pixel`.

## Configuration

Each dataset has a YAML config in `configs/`:

| File | Dataset | Backbone | Description |
|------|---------|----------|-------------|
| `waterbird.yaml` | Waterbird | ResNet-50 | Bird type vs. background |
| `celeba.yaml` | CelebA | ResNet-50 | Blond hair vs. gender |
| `isic.yaml` | ISIC | ResNet-50 | Skin lesion with colored patches |
| `umnist.yaml` | UMNIST | LeNet-5 | Digit classification with spurious digit 8 |

Key hyperparameters are documented in each YAML file. See `nct/core/config.py` for the full list of options.

## Dataset Preparation

Datasets should be placed in the `--data_dir` directory:

- **Waterbird**: Download from [Waterbirds dataset](https://github.com/kohpangwei/group_DRO#waterbirds)
- **CelebA**: Download from [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **ISIC**: Download from [ISIC Archive](https://www.isic-archive.com/)
- **UMNIST**: Generated programmatically from MNIST (handled by data loading code)
