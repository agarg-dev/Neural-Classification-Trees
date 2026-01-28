"""Image transformations and augmentations."""

from torchvision import transforms

from .configs import DatasetConfig


def create_transforms(config: DatasetConfig, is_training: bool = True):
    """Create image transforms based on dataset configuration."""

    if is_training:
        transform_list = [
            transforms.RandomResizedCrop(
                config.train_crop_size,
                scale=config.train_scale,
                ratio=config.train_ratio,
                interpolation=2,
            ),
            transforms.RandomHorizontalFlip(),
        ]

        if config.use_vertical_flip:
            transform_list.append(transforms.RandomVerticalFlip())

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.normalize_mean, std=config.normalize_std
                ),
            ]
        )

        return transforms.Compose(transform_list)
    else:
        transform_list = []

        if config.eval_centercrop_first is not None:
            transform_list.append(transforms.CenterCrop(config.eval_centercrop_first))
            transform_list.append(transforms.Resize(config.target_resolution))
        elif config.eval_resize is not None:
            transform_list.append(
                transforms.Resize((config.eval_resize, config.eval_resize))
            )
            transform_list.append(
                transforms.CenterCrop((config.eval_centercrop, config.eval_centercrop))
            )
        else:
            transform_list.append(transforms.Resize(config.target_resolution))

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.normalize_mean, std=config.normalize_std
                ),
            ]
        )

        return transforms.Compose(transform_list)
