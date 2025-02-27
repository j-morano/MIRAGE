from typing import Dict

import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

from mutils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mutils.dataset_folder import MultiTaskImageFolder



class ToRGB(A.ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p)

    def apply(self, img, **_params):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        return img

    def get_transform_init_args_names(self):
        return ()

class ToRange(A.ImageOnlyTransform):
    def __init__(self, p=1.0, range=(0, 255)):
        super(ToRange, self).__init__(p)
        self.range = range

    def apply(self, img, **_params):
        min_val, max_val = np.min(img), np.max(img)
        img = img * (self.range[1] - self.range[0]) / (max_val - min_val) + self.range[0]
        return img

    def get_transform_init_args_names(self):
        return ()


def simple_transform(
    train: bool,
    additional_targets: Dict[str, str],
    input_size: int = 512,
    norm: str = 'minmax',
):
    """Default transform for semantic segmentation, applied on all
    modalities.
    """

    norm_list = []
    if norm == 'imagenet':
        print("Using imagenet normalization")
        norm_list += [
            ToRGB(p=1),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    elif norm == 'sam':
        print("Using SAM normalization")
        # Rescale everything to [0, 255]
        norm_list += [
            ToRGB(p=1),
            ToRange(range=(0, 255), p=1),
        ]
    elif norm == 'z-score':
        print("Using z-score normalization")
        norm_list += [
            ToRGB(p=1),
            A.Normalize(mean=0, std=1),
        ]
    else:
        # No extra operations needed
        pass

    if train:
        init_size = input_size + (int(input_size * 0.1))
        transform_list = [
            A.HorizontalFlip(p=0.5),
            A.Resize(height=init_size, width=init_size, p=1),
            A.RandomCrop(height=input_size, width=input_size, p=1),
        ]
        transform_list += norm_list
        transform_list += [
            ToTensorV2(),
        ]
        transform = A.Compose(transform_list, additional_targets=additional_targets)
    else:
        transform_list = [
            A.Resize(height=input_size, width=input_size, p=1),
        ]
        transform_list += norm_list
        transform_list += [
            ToTensorV2(),
        ]
        transform = A.Compose(transform_list, additional_targets=additional_targets)  # type: ignore

    print(
        f'[Train: {train}]'
        f' [Input size: {input_size}]'
        f' [Normalization: {norm}]'
        f' [Additional targets: {additional_targets}]'
        f' [Transform list: {transform_list}]'
    )
    return transform


class DataAugmentationForSemSeg(object):
    """Data transform / augmentation for semantic segmentation
    downstream tasks.
    """

    def __init__(
        self,
        transform,
        seg_num_classes,
        key_to_replace='bscan'
    ):
        self.transform = transform
        self.seg_num_classes = seg_num_classes
        self.key_to_replace = key_to_replace

    def __call__(self, task_dict):
        # Need to replace rgb key to image
        key_to_replace = self.key_to_replace
        task_dict['image'] = task_dict.pop(key_to_replace)
        # Convert to np.array
        task_dict = {k: np.array(v) for k, v in task_dict.items()}

        task_dict = self.transform(**task_dict)

        # And then replace it back to rgb
        task_dict[key_to_replace] = task_dict.pop('image')

        for task in task_dict:
            if task in [key_to_replace]:
                task_dict[task] = task_dict[task].to(torch.float)
            elif task in ['semseg']:
                img = task_dict[task].to(torch.long)
                task_dict[task] = img

        return task_dict


def build_semseg_dataset(args, data_path, transform, max_images=None):
    transform = DataAugmentationForSemSeg(
        transform=transform,
        seg_num_classes=args.num_classes,
        key_to_replace=args.in_domains[0]
    )
    return MultiTaskImageFolder(
        data_path,
        args.all_domains,
        args=args,
        transform=transform,
        prefixes=None,
        max_images=max_images
    )
