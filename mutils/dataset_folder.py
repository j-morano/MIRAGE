import os
import os.path
from pathlib import Path
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
from skimage import io
from torchvision.datasets.vision import VisionDataset

from mutils.data_constants import IMG_EXTENSIONS



def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_valid_file(x: str, extensions: Tuple[str, ...]) -> bool:
    return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))


def make_nonclass_dataset(
    directory: str,
    extensions: Optional[Tuple[str, ...]] = None,
) -> List[Tuple[str, int]]:
    print(f"Making non-class dataset from {directory}")
    instances = []
    directory = os.path.expanduser(directory)
    if extensions is not None:
        c_is_valid_file = partial(is_valid_file, extensions=extensions)
    else:
        c_is_valid_file = partial(is_valid_file, extensions=IMG_EXTENSIONS)
    target_dir = directory
    assert os.path.isdir(target_dir), target_dir
    for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if c_is_valid_file(path):
                item = path, 0
                instances.append(item)
    return instances


def normalize_to_0_1(sample: np.ndarray) -> np.ndarray:
    """Normalize to 0-1 range data with any range (positive values)."""
    return (sample - np.min(sample)) / (np.max(sample) - np.min(sample))


class MultiTaskDatasetFolder(VisionDataset):
    """A generic multi-task dataset loader where the samples are arranged in this way: ::

        root/task_a/class_x/xxx.ext
        root/task_a/class_y/xxy.ext
        root/task_a/class_z/xxz.ext

        root/task_b/class_x/xxx.ext
        root/task_b/class_y/xxy.ext
        root/task_b/class_z/xxz.ext

    Args:
        root (string): Root directory path.
        tasks (list): List of tasks as strings
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt logs)
            both extensions and is_valid_file should not be passed.

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
            prefixes: Optional[Dict[str,str]] = None,
            max_images: Optional[int] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.tasks = tasks
        self.args = args
        assert args is not None

        prefixes = {} if prefixes is None else prefixes
        prefixes.update({task: '' for task in tasks if task not in prefixes})

        samples = {
            task: make_nonclass_dataset(
                os.path.join(self.root, f'{prefixes[task]}{task}'),
                extensions,
            )
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

        # Select random subset of dataset if so specified
        if isinstance(max_images, int):
            total_samples = len(list(self.samples.values())[0])
            np.random.seed(0)
            permutation = np.random.permutation(total_samples)
            for task in samples:
                self.samples[task] = [self.samples[task][i] for i in permutation][:max_images]

        self.cache = {}
        self.ids = {}

    def __getitem__(self, index: int) -> Tuple:
        """
        Returns:
            tuple: (sample, target, id)
        """
        target = None
        if index in self.cache:
            sample_dict, target = deepcopy(self.cache[index])
        else:
            sample_dict = {}
            for task in self.tasks:
                path, target = self.samples[task][index]
                sample = io.imread(path)
                if 'semseg' in task:
                    # Convert to 0-indexed labels
                    sample = np.vectorize(self.args.mapping.get)(sample)
                else:
                    sample = normalize_to_0_1(sample)
                sample_dict[task] = sample
                if index not in self.ids:
                    self.ids[index] = Path(path).stem
            # self.cache[index] = deepcopy((sample_dict, target))

        if self.transform is not None:
            sample_dict = self.transform(sample_dict)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # for k, v in sample_dict.items():
        #     print(f"Task: {k}, shape: {v.shape}")

        return sample_dict, target, self.ids[index]

    def __len__(self) -> int:
        return len(list(self.samples.values())[0])


class MultiTaskImageFolder(MultiTaskDatasetFolder):
    def __init__(
            self,
            root: str,
            tasks: List[str],
            args,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            prefixes: Optional[Dict[str,str]] = None,
            max_images: Optional[int] = None,
    ):
        super().__init__(
            root,
            tasks,
            args=args,
            extensions=IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
            prefixes=prefixes,
            max_images=max_images,
        )
        self.imgs = self.samples
