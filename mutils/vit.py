from functools import partial

import torch
import torch.nn as nn
import timm.models.vision_transformer as vit



class GlobalPooler:
    """Global avg. pooling without CLS token."""
    def __call__(self, x):
        return x[:, 1:, :].mean(dim=1)

class TokenMixPooler:
    """Token mixing: CLS token + global avg. pooling of patch tokens."""
    def __call__(self, x):
        cls_token = x[:, 0]
        patch_token = x[:, 1:].mean(dim=1)
        return torch.cat((cls_token, patch_token), dim=1)

class CLSPooler:
    """CLS token."""
    def __call__(self, x):
        return x[:, 0]


class VisionTransformer(vit.VisionTransformer):
    """Vision Transformer with support for different pooling methods."""

    def __init__(self, pool='global', **kwargs):
        super().__init__(**kwargs)

        self.pool = pool
        if self.pool == 'global':
            print("ViT: Using global pool")
            self.pooler = GlobalPooler()
        elif self.pool == 'token_mix':
            print("ViT: Using token mix")
            self.head = nn.Linear(
                in_features=kwargs["embed_dim"] * 2,
                out_features=kwargs["num_classes"],
                bias=True
            )
            self.pooler = TokenMixPooler()
        else:
            self.pooler = CLSPooler()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.pooler(x)


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

