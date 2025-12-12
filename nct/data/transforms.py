"""Image transformations and augmentations."""

from torchvision import transforms

from .configs import DatasetConfig


def create_transforms(config: DatasetConfig, is_training: bool = True):
    """Create image transforms based on dataset configuration."""

    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                config.train_crop_size,
                scale=config.train_scale,
                ratio=config.train_ratio,
                interpolation=2
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform_list = []

        if config.eval_centercrop_first is not None:
            transform_list.append(transforms.CenterCrop(config.eval_centercrop_first))
            transform_list.append(transforms.Resize(config.target_resolution))
        elif config.eval_resize is not None:
            transform_list.append(transforms.Resize((config.eval_resize, config.eval_resize)))
            transform_list.append(transforms.CenterCrop((config.eval_centercrop, config.eval_centercrop)))
        else:
            transform_list.append(transforms.Resize(config.target_resolution))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transforms.Compose(transform_list)
