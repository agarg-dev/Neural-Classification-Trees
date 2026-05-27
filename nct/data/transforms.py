"""Image transformations and augmentations."""

from torchvision import transforms

from .configs import DatasetConfig


def create_transforms(config: DatasetConfig, is_training: bool = True):
    """Create image transforms based on dataset configuration."""

    if is_training:
        if config.train_padding is not None:
            transform_list = []
            if config.train_resize is not None:
                transform_list.append(transforms.Resize(config.train_resize))
            transform_list.append(
                transforms.RandomCrop(config.train_crop_size, padding=config.train_padding)
            )
        else:
            transform_list = [
                transforms.RandomResizedCrop(
                    config.train_crop_size,
                    scale=config.train_scale,
                    ratio=config.train_ratio,
                    interpolation=2,
                ),
            ]

        if config.use_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if config.use_vertical_flip:
            transform_list.append(transforms.RandomVerticalFlip())

        if config.train_padding is not None:
            target_h = (
                config.target_resolution[0]
                if isinstance(config.target_resolution, tuple)
                else config.target_resolution
            )
            if config.train_crop_size != target_h:
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
