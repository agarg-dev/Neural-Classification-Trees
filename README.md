# Neural Classification Trees (NCT)

Interpretable hierarchical classifier for image datasets with spurious correlations. NCT iteratively builds a tree of classifiers, routing difficult samples to specialized child nodes to improve worst-group accuracy.

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

Supported datasets: `waterbird`, `celeba`, `isic`, `umnist`, `cmnist`, `urbancars`.

## Attribution Generation

Regenerate attribution visualizations from a saved checkpoint:

```bash
python src/generate_attributions.py \
    --checkpoint ./checkpoints/final_model.pt \
    --dataset waterbird \
    --data_dir ./data \
    --methods all_layer
```

Available methods: `layer_gradcam`, `guided_gradcam`, `layer_conductance`, `layer_integrated_gradients`, `layer_gradient_x_activation`, `saliency`, `input_x_gradient`, `deeplift`, `guided_backprop`, `occlusion`, or the keywords `all`, `all_layer`, `all_pixel`.

## Configuration

Each dataset has a YAML config in `configs/`:

| File | Dataset | Backbone | Description |
|------|---------|----------|-------------|
| `waterbird.yaml` | Waterbirds | ResNet-50 | Bird type vs. background |
| `celeba.yaml` | CelebA | ResNet-50 | Blond hair vs. gender |
| `isic.yaml` | ISIC | ResNet-50 | Skin lesion vs. colored patches |
| `umnist.yaml` | UMNIST | LeNet-4 | Binary digit vs. digit-8 minority |
| `cmnist.yaml` | CMNIST | LeNet-5 | 5-class digits vs. color (5 colors) |
| `urbancars.yaml` | UrbanCars | ResNet-50 | Car type vs. background and co-occurring object |

See `nct/core/config.py` for the full list of options.

## Dataset Preparation

Place datasets under `--data_dir`:

- **Waterbirds** &mdash; <https://github.com/kohpangwei/group_DRO#waterbirds>
- **CelebA** &mdash; <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
- **ISIC** &mdash; <https://www.isic-archive.com/>
- **UMNIST** &mdash; prepared undersampled MNIST images and `metadata.csv` under `data/umnist/`.
- **CMNIST** &mdash; prepared 5-class colored MNIST images and `metadata.csv` under `data/cmnist/`.
- **UrbanCars** &mdash; use the [official preparation script](https://github.com/facebookresearch/Whac-A-Mole/blob/main/scripts/prepare_dataset_models/create_urbancars.sh) and place its output under `data/urbancars/`.

## Method Summary

Each iteration trains one classification head per node and routes samples to easy/difficult children based on prediction correctness:
- Iteration 1: `num_classes` heads (one per class).
- Iteration `t`: `num_classes * 2^(t-1)` heads.
- A sample correct at iteration `t` is routed to its parent's easy child; an incorrect sample is routed to the difficult child.

Specialized heads on the difficult path concentrate on minority subgroups (the spurious-conflicting samples), improving worst-group accuracy without group labels.
