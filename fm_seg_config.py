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
