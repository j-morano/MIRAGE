import random
from typing import Callable, Dict, List, Optional, Tuple
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
from skimage import io
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

from mutils.dataset_folder import make_nonclass_dataset



class DataAugmentationForMIRAGE:
    def __init__(self, args):
        self.input_size = args.input_size
        self.hflip = args.hflip
        self.intensity_shift = args.intensity_shift
        self.random_crop = args.random_crop
        self.affine = transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        )
        self.input_size = args.input_size
        self.args = args

    def __call__(self, task_dict):
        flip = random.random() < self.hflip  # Stores whether to flip all images or not
        affine = self.affine
        affine_params = affine.get_params(
            affine.degrees, affine.translate, affine.scale, affine.shear,
            [512, 512]
        )
        for task in task_dict.keys():
            if flip:
                task_dict[task] = np.flip(task_dict[task], axis=-1)
            if self.intensity_shift > 0 and task not in ['layermaps', 'bscanlayermap']:
                # IMPORTANT: It assumes that the image is in [0, 1] range
                shift = np.random.normal(0, self.intensity_shift, 1).astype(np.float32)
                if random.random() < 0.5:
                    shift = -shift
                task_dict[task] = np.clip(task_dict[task] + shift, 0, 1)

            img = torch.from_numpy(task_dict[task].copy()).contiguous().unsqueeze(0)

            if task in ['bscan', 'bscanlayermap']:
                # All transformations
                c_params = affine_params
            else:
                # Only translation in x direction
                c_params = 0, (affine_params[1][0],0), affine_params[2], 0
            if self.args.affine:
                img = TF.affine(  # type: ignore
                    img,
                    *c_params,
                    interpolation=affine.interpolation,
                    fill=0,  # type: ignore
                    center=affine.center,
                )
            if task in ['layermaps', 'bscanlayermap']:
                interpolation = TF.InterpolationMode.NEAREST
            else:
                interpolation = TF.InterpolationMode.BILINEAR

            if img.shape[1:] != tuple(self.input_size[task]):
                img = TF.resize(
                    img,
                    self.input_size[task],
                    interpolation=interpolation,
                )
            if task in ['layermaps', 'bscanlayermap']:
                img = img.squeeze(0)
            task_dict[task] = img

        return task_dict


class MultiTaskPretDatasetFolder(VisionDataset):
    """A generic multi-task dataset loader where the samples are arranged in this way: ::
        root/task_a/class_x/xxx.ext
        root/task_a/class_y/xxy.ext

        root/task_b/class_x/xxx.ext
        root/task_b/class_y/xxy.ext

    Args:
        root (string): Root directory path.
        tasks (list): List of tasks as strings
        extensions (tuple[string]): A list of allowed extensions.
            extensions should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

    Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        tasks: List[str],
        args,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_cache: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.tasks = tasks
        self.args = args
        self.use_cache = use_cache
        assert args is not None

        samples = {
            task: make_nonclass_dataset(os.path.join(self.root, task), extensions)
            for task in self.tasks
        }

        for task, task_samples in samples.items():
            if len(task_samples) == 0:
                msg = "Found 0 logs in subfolders of: {}\n".format(os.path.join(self.root, task))
                if extensions is not None:
                    msg += "Supported extensions are: {}".format(",".join(extensions))
                raise RuntimeError(msg)

        self.extensions = extensions

        self.samples = samples
        # self.targets = [s[1] for s in list(samples.values())[0]]

        self.cache = {}
        self.ids = {}

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        target = None
        if self.use_cache and index in self.cache:
            sample_dict, target = deepcopy(self.cache[index])
        else:
            sample_dict = {}
            for task in self.tasks:
                sample = None
                path, target = self.samples[task][index]
                if path.endswith('.npy') or path.endswith('.npz'):
                    if task == 'layermaps':
                        sample = np.load(path)['layer_maps'].astype(int)
                    elif task == 'bscanlayermap':
                        sample = np.load(path).astype(int)
                    else:
                        sample = np.load(path).astype(np.float32) / 255.0
                else:
                    sample = io.imread(path) / 255.0
                sample_dict[task] = sample
                if index not in self.ids:
                    self.ids[index] = Path(path).stem
            if self.use_cache:
                self.cache[index] = deepcopy((sample_dict, target))

        if self.transform is not None:
            sample_dict = self.transform(sample_dict)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # for task in sample_dict:
        #     print(task, sample_dict[task].shape, sample_dict[task].min(), sample_dict[task].max())

        return sample_dict, target, self.ids[index]

    def __len__(self) -> int:
        return len(list(self.samples.values())[0])


def build_mirage_pretraining_dataset(args):
    transform = DataAugmentationForMIRAGE(args)
    return MultiTaskPretDatasetFolder(
        args.data_path,
        args.all_domains,
        args=args,
        transform=transform,
    )
