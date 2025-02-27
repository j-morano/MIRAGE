import random

import torch
from torch import Tensor
from torch import nn
import torchvision.transforms.functional as VF
import torchvision.transforms as tvtr



class MinMaxNorm:
    """
    Transforms each channel to the range [0, 1].
    """
    def __call__(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())


class MinMaxNormChannel:
    def __init__(self) -> None:
        super().__init__()
        self.scaler = MinMaxNorm()

    def __call__(self, tensor):
        for i in range(tensor.shape[0]):
            if tensor[i].max() > 0:
                tensor[i] = self.scaler(tensor[i:i+1].clone())
        return tensor


class NaiveNorm:
    """
    Transforms each channel to the range [0, 1], if it is not already.
    """
    def __call__(self, tensor):
        if tensor.min() < 0:
            raise ValueError("Tensor contains negative values")
        elif tensor.max() > 1 and tensor.max() <= 255:
            tensor = tensor / 255.0
        elif tensor.max() > 255:
            tensor = tensor / 65535.0
        return tensor


class NaiveNormChannel:
    """
    Transforms each channel to the range [0, 1], if it is not already.
    """
    def __init__(self) -> None:
        super().__init__()
        self.scaler = NaiveNorm()

    def __call__(self, tensor):
        for i in range(tensor.shape[0]):
            tensor[i] = self.scaler(tensor[i:i+1].clone())
        return tensor


class Identity:
    def __call__(self, img):
        return img


class ToRGB:
    def __call__(self, img: torch.Tensor):
        return img.repeat(3, 1, 1)


class RandomIntensity(nn.Module):
    def __init__(self, intensity_range=(0.8, 1.2)):
        super().__init__()
        self.intensity_range = intensity_range

    @staticmethod
    def get_abs_max(tensor):
        if tensor.max() <= 1:
            abs_max = 1
        elif tensor.max() > 1 and tensor.max() <= 255:
            abs_max = 255
        elif tensor.max() > 255:
            abs_max = 65535
        else:
            raise ValueError(
                "Image values are not in the expected range:"
                f" [{tensor.max()}, {tensor.min()}], {torch.unique(tensor)}"
            )
        return abs_max

    def forward(self, img):
        intensity = torch.empty(1).uniform_(*self.intensity_range).item()
        return torch.clamp(img * intensity, 0, self.get_abs_max(img))


class RandomIntensityChannel(nn.Module):
    def __init__(self, intensity_range=(0.8, 1.2)):
        super().__init__()
        self.intensity_range = intensity_range
        self.intensity = RandomIntensity(intensity_range)

    def forward(self, img):
        for i in range(img.shape[0]):
            if img[i].max() > 0:
                img[i] = self.intensity(img[i:i+1].clone())
        return img


class RandomAffineChannel(tvtr.RandomAffine):
    """Same as RandomAffine but with a random rotation for every
    channel.
    """
    def __init__(self, p=1.0, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        if random.random() < (1 - self.p):
            return img

        if self.fill == 0.5:
            fill = random.uniform(img.min().item(), img.max().item())
        else:
            fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)]
            else:
                fill = [float(f) for f in fill]  # type: ignore

        img_size = VF.get_image_size(img)

        for i in range(img.shape[0]):
            # Apply a transformation only in 90% of the cases
            if random.random() < 0.9:
                ret = self.get_params(
                    self.degrees, self.translate, self.scale, self.shear,
                    img_size
                )
                img[i] = VF.affine(
                    img[i:i+1].clone(), *ret, interpolation=self.interpolation,
                    fill=fill, center=self.center  # type: ignore
                )
        return img
