# From:  https://github.com/Project-MONAI/MONAI/tree/0.3.0
# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import warnings
from typing import Any, Tuple
import collections.abc



def issequenceiterable(obj: Any) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str)


def ensure_tuple(vals: Any) -> Tuple[Any, ...]:
    """
    Returns a tuple of `vals`.
    """
    if not issequenceiterable(vals):
        vals = (vals,)

    return tuple(vals)


def ensure_tuple_size(tup: Any, dim: int, pad_val: Any = 0) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or padded with `pad_val` as necessary.
    """
    tup = ensure_tuple(tup) + (pad_val,) * dim
    return tuple(tup[:dim])


def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    For a tensor `labels` of dimensions B1[spatial_dims], return a tensor of dimensions `BN[spatial_dims]`
    for `num_classes` N number of classes.

    Example:

        For every value v = labels[b,1,h,w], the value in the result at [b,v,h,w] will be 1 and all others 0.
        Note that this will include the background label, thus a binary mask should be treated as having 2 classes.
    """
    assert labels.dim() > 0, "labels should have dim of 1 or more."

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndimension() < dim + 1:
        shape = ensure_tuple_size(labels.shape, dim + 1, 1)
        labels = labels.reshape(*shape)

    sh = list(labels.shape)

    assert sh[dim] == 1, "labels should have a channel with length equals to one."
    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


class GeneralizedDiceLoss(torch.nn.modules.loss._Loss):
    """
    Compute the generalised Dice loss defined in:

        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L279
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        reduction: str = 'mean',
    ) -> None:
        """
        Args:
            include_background: If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            w_type: {``"square"``, ``"simple"``, ``"uniform"``}
                Type of function to transform ground truth volume to a weight factor. Defaults to ``"square"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super(GeneralizedDiceLoss, self).__init__()

        if int(sigmoid) + int(softmax)> 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.w_func = lambda x: torch.reciprocal(x * x)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            smooth: a small constant to avoid nan.

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        assert (
            target.shape == input.shape
        ), f"ground truth has differing shape ({target.shape}) from input ({input.shape})"

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, len(input.shape)))
        intersection = torch.sum(target * input, reduce_axis)

        ground_o = torch.sum(target, reduce_axis)
        pred_o = torch.sum(input, reduce_axis)

        denominator = ground_o + pred_o

        w = self.w_func(ground_o.float())
        for b in w:
            infs = torch.isinf(b)
            b[infs] = 0.0
            b[infs] = torch.max(b)

        f: torch.Tensor = 1.0 - (2.0 * (intersection * w).sum(1) + smooth) / ((denominator * w).sum(1) + smooth)


        if self.reduction == 'mean':
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == 'sum':
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            pass  # returns [N, n_classes] losses
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


class CEGDiceLoss(nn.Module):
    """Combined Cross-entropy and Generalized Dice Loss."""

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = True,
        sigmoid: bool = False,
        softmax: bool = True,
        reduction: str = 'mean',
        ce_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight

        self.ce = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.dice = GeneralizedDiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            reduction=reduction,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input shape: torch.Size([batch_size, num_classes, height, width])
        # target shape: torch.Size([batch_size, height, width])
        ce_loss = self.ce(input, target)
        target = target.unsqueeze(1)
        dice_loss = self.dice(input, target)

        return self.ce_weight * ce_loss + (1 - self.ce_weight) * dice_loss
