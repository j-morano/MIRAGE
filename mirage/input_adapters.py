from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mirage.utils import build_2d_sincos_posemb, pair, trunc_normal_



class PatchedInputAdapter(nn.Module):
    """Adapter for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.

    Args:
        num_channels: Number of input channels of the image/feature map
        stride_level: Stride level compared to the full-sized image.
            E.g. 4 for 1/4th the size of the image.
        patch_size_full: Int or tuple of the patch size over the full image size.
            Patch size for smaller inputs will be computed accordingly.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        learnable_pos_emb: Set to True to learn positional embeddings instead
        image_size: Default image size. Used to initialize size of positional embeddings.
    """
    def __init__(
        self,
        num_channels: int,
        stride_level: int,
        patch_size_full: Union[int, Tuple[int,int]],
        dim_tokens: Optional[int] = None,
        sincos_pos_emb: bool = True,
        learnable_pos_emb: bool = False,
        image_size: Union[int, Tuple[int]] = 224
    ):

        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        print(f'Image size: {self.image_size}, Patch size: {self.patch_size_full}')
        self.num_patches = (self.image_size[0] // patch_size_full[0]) * (self.image_size[1] // patch_size_full[1])

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up MIRAGE.

        Args:
            dim_tokens: Dimension of tokens
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if self.sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=self.learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, h_posemb, w_posemb))
            trunc_normal_(self.pos_emb, std=0.02)

        # Image -> tokens projection
        self.proj = nn.Conv2d(
            in_channels=self.num_channels, out_channels=self.dim_tokens,
            kernel_size=(self.P_H, self.P_W), stride=(self.P_H, self.P_W)
        )

    @torch.jit.ignore  # type: ignore
    def no_weight_decay(self):
        return {'pos_emb'}

    def forward(self, x):
        """
        Forward pass through input adapter, transforming image to sequence of tokens.
        Adds task and positional encodings.

        Args:
            x: Input image tensor
        """
        _B, _C, H, W = x.shape
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        assert (H % self.P_H == 0) and (W % self.P_W == 0), f'Image sizes {H}x{W} must be divisible by patch sizes {self.P_H}x{self.P_W}'
        N_H, N_W = H // self.P_H, W // self.P_W # Number of patches in height and width

        # Create patches [B, C, H, W] -> [B, (H*W), C]
        x_patch = rearrange(self.proj(x), 'b d nh nw -> b (nh nw) d')

        # Create positional embedding
        x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode='bicubic', align_corners=False)
        x_pos_emb = rearrange(x_pos_emb, 'b d nh nw -> b (nh nw) d')

        # Add patches and positional embeddings
        x = x_patch + x_pos_emb

        return x


class SemSegInputAdapter(nn.Module):
    """
    Adapter for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.

    Args:
        num_classes: Number of input semantic classes
        stride_level: Stride level compared to the full-sized image.
            E.g. 4 for 1/4th the size of the image.
        patch_size_full: Int or tuple of the patch size over the full image size.
            Patch size for smaller inputs will be computed accordingly.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        learnable_pos_emb: Set to True to learn positional embeddings instead
        image_size: Default image size. Used to initialize size of positional embeddings.
        dim_class_emb: Dimension of learned class embedding
        interpolate_class_emb: Set to True to average pool class embeddings of each patch
        emb_padding_idx: Padding index (e.g. image border), default is None
    """

    def __init__(
        self,
        num_classes: int,
        stride_level: int,
        patch_size_full: Union[int, Tuple[int, int]],
        dim_tokens: Optional[int] = None,
        sincos_pos_emb: int = True,
        learnable_pos_emb: int = False,
        image_size: Union[int, Tuple[int]] = 224,
        dim_class_emb: int = 64,
        interpolate_class_emb: bool = False,
        emb_padding_idx: int = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        self.dim_class_emb = dim_class_emb
        self.interpolate_class_emb = interpolate_class_emb
        self.emb_padding_idx = emb_padding_idx
        if self.emb_padding_idx is not None:
            self.num_classes += 1

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)

        print(f'Image size: {self.image_size}, Patch size: ({self.P_H}, {self.P_W})')

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768):
        '''
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up MIRAGE.

        Args:
            dim_tokens: Dimension of tokens
        '''
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if self.sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=self.learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, h_posemb, w_posemb))
            trunc_normal_(self.pos_emb, std=0.02)

        # Image -> tokens projection
        self.class_emb = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.dim_class_emb, padding_idx=self.emb_padding_idx)
        trunc_normal_(self.class_emb.weight, std=0.02)

        if self.interpolate_class_emb:
            self.proj = nn.Sequential(
                nn.Upsample(scale_factor=(1 / self.P_H, 1 / self.P_W),
                            mode='bilinear'),  # Actually a downsample operation
                nn.Conv2d(in_channels=self.dim_class_emb, out_channels=self.dim_tokens,
                          kernel_size=1, stride=1),
            )
        else:
            self.proj = nn.Conv2d(
                in_channels=self.dim_class_emb, out_channels=self.dim_tokens,
                kernel_size=(self.P_H, self.P_W), stride=(self.P_H, self.P_W)
            )

    @torch.jit.ignore  # type: ignore
    def no_weight_decay(self):
        return {'pos_emb', 'class_emb'}

    def forward(self, x):
        '''
        Forward pass through input adapter, transforming image to sequence of tokens.
        Adds task and positional encodings.

        Args:
            x: Input image tensor
        '''
        _B, H, W = x.shape
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        assert (H % self.P_H == 0) and (
                    W % self.P_W == 0), f'Image sizes {H}x{W} must be divisible by patch sizes {self.P_H}x{self.P_W}'
        N_H, N_W = H // self.P_H, W // self.P_W  # Number of patches in height and width

        # Map to embedding
        x = rearrange(self.class_emb(x), 'b nh nw c -> b c nh nw')

        # Create patches [B, C, H, W] -> [B, (H*W), C]
        x_patch = rearrange(self.proj(x), 'b d nh nw -> b (nh nw) d')

        # Create positional embedding
        x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode='bilinear')
        x_pos_emb = rearrange(x_pos_emb, 'b d nh nw -> b (nh nw) d')

        # Add patches and positional embeddings
        x = x_patch + x_pos_emb

        return x

