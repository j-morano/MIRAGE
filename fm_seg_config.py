import copy
from typing import Dict
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from mutils.pos_embed import interpolate_pos_embed
from mutils.factory import get_factory_adder
from mirage.input_adapters import PatchedInputAdapter, SemSegInputAdapter



DOMAIN_CONF = {
    'bscan': {
        'channels': 1,
        'stride_level': 1,
        'aug_type': 'image',
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
    },
    'slo': {
        'channels': 1,
        'stride_level': 1,
        'aug_type': 'image',
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
    },
    'semseg': {
        'stride_level': 4,
        'aug_type': 'mask',
        'input_adapter': partial(SemSegInputAdapter,
            num_classes=4,
            dim_class_emb=64,
            interpolate_class_emb=False,
            emb_padding_idx=4),
    },
}


# Foundation model config factory
add_fm, fm_factory = get_factory_adder()

# IMPORTANT: the name of the model used in add_fm should be contained
#   in the checkpoint file name. This is used to determine which FM
#   to use.


class FoundModel:
    def __init__(self, norm: str, model: str):
        self.norm = norm
        self.model = model
        self.domain_conf: Dict[str, Dict]

    def __call__(self, model, checkpoint):
        print(f'>> Using {self.__class__.__name__} to load model')
        checkpoint_model = self.loader(checkpoint)

        # Interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # Load pre-trained model
        _msg = model.load_state_dict(checkpoint_model, strict=False)
        # print(_msg)
        return model

    def build_domain_conf(self):
        domain_conf = copy.deepcopy(DOMAIN_CONF)
        if self.norm != 'minmax':
            print('>>> Using 3 channels instead of 1')
            domain_conf['bscan']['channels'] = 3
            domain_conf['bscan']['input_adapter'] = partial(PatchedInputAdapter, num_channels=3)
            domain_conf['slo']['channels'] = 3
            domain_conf['slo']['input_adapter'] = partial(PatchedInputAdapter, num_channels=3)
        self.domain_conf = domain_conf

    @staticmethod
    def loader(_checkpoint):
        raise NotImplementedError


@add_fm('dinov2')
class DINOv2FM(FoundModel):
    def __init__(self, norm='imagenet', model='miragelight_large'):
        super().__init__(norm, model)

    def __call__(self, model, _checkpoint):
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        model.encoder = nn.Sequential(*dino.blocks)  # type: ignore
        print('>> Loaded DINOv2 model directly from torch hub')
        checkpoint_model = model.state_dict()
        interpolate_pos_embed(model, checkpoint_model)
        return model


@add_fm('retfound')
class RETFoundFM(FoundModel):
    def __init__(self, norm='imagenet', model='miragelight_large'):
        super().__init__(norm, model)

    @staticmethod
    def loader(checkpoint):
        checkpoint_model = checkpoint['model']
        print('Changing model keys')
        def do_replacements(key):
            if key == 'cls_token':
                return 'global_tokens'
            if key.startswith('blocks.'):
                key = key.replace('blocks.', 'encoder.')
            return key
        # Remove all the keys 'decoder_encoder.'
        new_checkpoint_model = {}
        for k, v in checkpoint_model.items():
            if not k.startswith('decoder_blocks.'):
                new_checkpoint_model[do_replacements(k)] = v
        checkpoint_model = new_checkpoint_model
        return checkpoint_model


@add_fm('medsam')
class MedSAMFM(FoundModel):
    def __init__(self, norm='sam', model='miragelight_base'):
        super().__init__(norm, model)

    @staticmethod
    def loader(checkpoint):
        print('Changing checkpoints keys for MedSAM model')
        checkpoint_model = checkpoint
        # Remove all weights starting with 'mask_decoder'
        for k in list(checkpoint_model.keys()):
            if 'mask_decoder' in k or 'prompt_encoder' in k:
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if 'image_encoder' in k:
                checkpoint_model[
                    k.replace('image_encoder.blocks', 'encoder')
                    .replace('lin1', 'fc1')
                    .replace('lin2', 'fc2')
                ] = checkpoint_model.pop(k)
        return checkpoint_model


@add_fm('mirage-large')
class MIRAGELargeFM(FoundModel):
    def __init__(self, norm='minmax', model='miragelight_large'):
        super().__init__(norm, model)

    @staticmethod
    def loader(checkpoint):
        # This is for MIRAGE models
        checkpoint_model = checkpoint['model']
        # Replace all 'bscanlayermap' with 'semseg'
        print("Replacing bscanlayermap with semseg")
        for k in list(checkpoint_model.keys()):
            if 'bscanlayermap' in k:
                checkpoint_model[k.replace('bscanlayermap', 'semseg')] = checkpoint_model.pop(k)

        class_emb_key = 'input_adapters.semseg.class_emb.weight'
        if class_emb_key in checkpoint_model:
            checkpoint_model[class_emb_key] = F.pad(checkpoint_model[class_emb_key], (0, 0, 0, 1))

        # Remove output adapters
        for k in list(checkpoint_model.keys()):
            if "output_adapters" in k:
                del checkpoint_model[k]
        return checkpoint_model


@add_fm('mirage-base')
class MIRAGEBaseFM(MIRAGELargeFM):
    def __init__(self, norm='minmax', model='miragelight_base'):
        super().__init__(norm, model)
