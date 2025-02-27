from typing import List, Callable

import torch
from torch import nn
import torchvision.transforms as tvtr
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from mirage_wrapper import MIRAGECls
from mutils.transforms import (
    RandomIntensityChannel,
    RandomAffineChannel,
    Identity,
    MinMaxNormChannel,
    ToRGB,
    NaiveNormChannel
)
from mutils.factory import get_factory_adder
import mutils.lr_utils as lru
from mutils.vit import vit_large_patch16, vit_base_patch16



add_config, fm_config_factory = get_factory_adder()



class FoundModel:
    def __init__(self, args):
        args.weight_decay = 1e-2
        if args.fill is not None:
            if args.fill < 0:
                args.fill = None
        if args.linear_probing:
            print('> Linear probing')
            args.lr = 1e-3
        else:
            print('> Full finetune')
            args.lr = 1e-5
        self.args = args
        self.model: nn.Module

    def get_optimizer(self, model):
        return torch.optim.AdamW(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

    def build_transform(self, subset, augment):
        print(f'>>> Building transform "{subset}"')
        intensity_msg = 'Random intensity shift'
        intensity = RandomIntensityChannel()
        if self.args.fill is None:
            if 'kermany' in self.args.data_set.lower():
                fill = 1
            else:
                fill = 0
        else:
            fill = self.args.fill
        affine_msg = f'Random affine (fill={fill})'
        affine = RandomAffineChannel(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5,
            interpolation=tvtr.InterpolationMode.BILINEAR,
            fill=fill,
        )
        if not self.args.affine:
            affine_msg = 'No random affine'
            affine = Identity()
        grayscale = Identity()
        grayscale = tvtr.Grayscale(num_output_channels=1)
        norm_list = [ NaiveNormChannel() ]
        min_max = self.get_min_max()
        norm_list += self.get_model_norm()

        transforms_list = [
            tvtr.Resize(
                size=(self.args.input_size, self.args.input_size),
                interpolation=tvtr.InterpolationMode.BILINEAR,
            ),
            grayscale,
            tvtr.ToTensor(),
            tvtr.ConvertImageDtype(torch.float32),
            min_max,
        ]
        if augment:
            print('Random horizontal flip (0.5)')
            print(intensity_msg)
            print(affine_msg)
            transforms_list += [
                tvtr.RandomHorizontalFlip(p=0.5),
                intensity,
                affine,
            ]
        print('Norm list:', norm_list)
        transforms_list += norm_list
        transforms = tvtr.Compose(transforms_list)

        return transforms

    def get_model_norm(self) -> List[Callable]:
        # By default, convert to RGB and apply ImageNet normalization
        return [
            ToRGB(),
            tvtr.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]

    def get_min_max(self) -> Callable:
        return Identity()

    def set_requires_grad(self):
        if self.args.linear_probing:
            # print(model.state_dict().keys())
            print('Freezing encoder layers for linear probing')
            # freeze encoder layers for linear probing
            print('Tuned parameters:')
            for name, param in self.model.named_parameters():
                if 'head.' not in name:  # and 'norm.' not in name:
                    param.requires_grad = False
                else:
                    print('\t', name)
        else:
            for param in self.model.parameters():
                param.requires_grad = True


class FoundSOTAModel(FoundModel):
    def __init__(self, args):
        super().__init__(args)
        if self.args.input_size is None:
            self.args.input_size = 224


class MIRAGEFM(FoundModel):
    def __init__(self, args):
        super().__init__(args)
        if args.input_size is None:
            args.input_size = 512

        self.model = MIRAGECls(
            input_size=args.input_size,
            patch_size=32,
            num_classes=args.num_classes,
            modalities='bscan',
            # NOTE: weights are loaded in the model
            weights=args.weights,
        )
        self.args = args

    def get_model_norm(self) -> List[Callable]:
        return [ MinMaxNormChannel() ]

    def get_min_max(self) -> Callable:
        return MinMaxNormChannel()


@add_config('mirage-large')
class MIRAGELargeFM(MIRAGEFM):
    pass


@add_config('mirage-base')
class MIRAGEBaseFM(MIRAGEFM):
    pass


class ViT21kFoundModel(FoundSOTAModel):
    def load_model(self, model_fun, args, state_dict_fn):
        model = model_fun(
            img_size=args.input_size,
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path,
            pool=args.pool,
        )
        state_dict = torch.load(state_dict_fn, weights_only=False)
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        assert set(msg.missing_keys) == {
            'head.weight',
            'head.bias',
        }
        return model


@add_config('imagenet21k-base')
class ViTB16_21kFM(ViT21kFoundModel):
    # Weights: timm--vit_base_patch16_224.orig_in21k.bin
    def __init__(self, args):
        super().__init__(args)
        self.model = self.load_model(
            vit_base_patch16,
            args,
            args.weights
        )


@add_config('imagenet21k-large')
class ViTL16_21kFM(ViT21kFoundModel):
    # Weights: timm--vit_large_patch16_224.orig_in21k.bin
    def __init__(self, args):
        super().__init__(args)
        self.model = self.load_model(
            vit_large_patch16,
            args,
            args.weights
        )


class DINOv2TokenMix(nn.Module):
    def __init__(self, model, embed_dim, num_classes):
        super().__init__()
        self.model = model
        self.head = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x):
        out = self.model.forward_features(x)
        out = torch.cat([
            out['x_norm_clstoken'],
            out['x_norm_patchtokens'].mean(dim=1)
        ], dim=1)
        out = self.head(out)
        return out


class DINOv2GlobalPool(nn.Module):
    def __init__(self, model, embed_dim, num_classes):
        super().__init__()
        self.model = model
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        out = self.model.forward_features(x)
        out = out['x_norm_patchtokens'].mean(dim=1)
        out = self.head(out)
        return out


class DINOv2CLSToken(DINOv2GlobalPool):
    def forward(self, x):
        out = self.model.forward_features(x)
        out = out['x_norm_clstoken']
        out = self.head(out)
        return out


class DINOv2FM(FoundSOTAModel):
    def load_model(self, model_name, embed_dim):
        model = torch.hub.load('facebookresearch/dinov2', model_name)
        assert isinstance(model, nn.Module)
        print(f'Loaded ImageNet weights for {model_name}')
        if self.args.pool == 'token_mix':
            print('ViT: Using token mix')
            model = DINOv2TokenMix(model, embed_dim, self.args.num_classes)
        elif self.args.pool == 'global':
            model = DINOv2GlobalPool(model, embed_dim, self.args.num_classes)
        elif self.args.pool == 'cls':
            model = DINOv2CLSToken(model, embed_dim, self.args.num_classes)
        return model


@add_config('dinov2-base')
class DINOv2ViTB14FM(DINOv2FM):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.load_model('dinov2_vitb14', 768)


@add_config('dinov2-large')
class DINOv2ViTL14FM(DINOv2FM):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.load_model('dinov2_vitl14', 1024)


@add_config('retfound')
class RETFoundFM(FoundSOTAModel):
    # Weights: RETFound_mae_natureOCT.pth
    def __init__(self, args):
        super().__init__(args)
        self.model = self.load_model()
        if not args.linear_probing:
            if 'octdl' in args.data_set.lower():
                self.args.lr = 1e-2
            else:
                self.args.lr = 1e-4

    def load_model(self):
        model = vit_large_patch16(
            img_size=self.args.input_size,
            num_classes=self.args.num_classes,
            drop_path_rate=self.args.drop_path,
            pool=self.args.pool,
        )

        # load pre-trained weights if path provided
        checkpoint_model = torch.load(
            self.args.weights,
            map_location='cpu',
            weights_only=False
        )['model']

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        assert set(msg.missing_keys) == {
            'head.weight',
            'head.bias',
        }
        return model

    def get_optimizer(self, model):
        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lru.param_groups_lrd(
            model,
            self.args.weight_decay,
            no_weight_decay_list=model.no_weight_decay(),
            layer_decay=self.args.layer_decay,
        )
        return torch.optim.AdamW(param_groups, lr=self.args.lr)
