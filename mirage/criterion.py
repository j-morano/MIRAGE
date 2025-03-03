import unittest
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param label_smoothing: Amount of smoothing in the loss (default is 0.0)
    """

    def __init__(
        self,
        patch_size: Union[tuple, list],
        stride: int = 1,
        label_smoothing : float = 0.0
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size[0] // stride
        self.label_smoothing = label_smoothing
        self.epoch = 0

    def forward(self, input, target, mask=None):

        loss = F.cross_entropy(input, target, reduction='none', label_smoothing=self.label_smoothing)

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            H, W = input.shape[-2:]
            nh, nw = H // self.scale_factor, W // self.scale_factor
            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
            loss = loss.nanmean()  # Account for zero masks
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training

        return loss


class SpatialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epoch = 0

    def patchify(self, imgs, nh, nw):
        p = self.scale_factor
        x = rearrange(imgs, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p)
        return x

    def unpatchify(self, x, nh, nw):
        p = self.scale_factor
        imgs = rearrange(x, "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)", nh=nh, nw=nw, p1=p, p2=p)
        return imgs


class MaskedMSELoss(SpatialLoss):
    """L2 loss with masking.
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param norm_pix: Normalized pixel loss
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        if isinstance(patch_size, int):
            self.scale_factor = patch_size // stride
        elif isinstance(patch_size, (list, tuple)):
            self.scale_factor = patch_size[0] // stride
        self.norm_pix = norm_pix

    def forward(self, input, target, mask=None):

        H, W = input.shape[-2:]
        nh, nw = H // self.scale_factor, W // self.scale_factor

        if self.norm_pix:
            target = self.patchify(target, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nh, nw)

        loss = F.mse_loss(input, target, reduction='none')

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            loss = loss.mean(dim=1)  # B, C, H, W -> B, H, W
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
            loss = loss.nanmean() # Account for zero masks
        else:
            loss = loss.mean() # If this is ever nan, we want it to stop training

        return loss


