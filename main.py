from functools import partial
from pathlib import Path
import copy
import argparse

import numpy as np
import torch
from torch import nn
from skimage import io
from skimage.transform import resize
from torchvision.utils import save_image

from mirage.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from mirage.output_adapters import SpatialOutputAdapter
from mirage.model import MIRAGEModel



DEFAULT_CONF = {
    'channels': 1,
    'stride_level': 1,
    'input_adapter': partial(PatchedInputAdapter, num_channels=1),
    'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
}


DOMAIN_CONF = {
    'bscan': copy.deepcopy(DEFAULT_CONF),
    'slo': copy.deepcopy(DEFAULT_CONF),
    "bscanlayermap": {
        "num_classes": 13,
        "stride_level": 1,
        "input_adapter": partial(
            SemSegInputAdapter,
            num_classes=13,
            dim_class_emb=64,
            interpolate_class_emb=False,
        ),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=13),
    },
}


def get_model(args):
    """Creates and returns model from arguments."""
    print(
        f"Creating model: {args.model} for inputs {args.in_domains}"
        f" and outputs {args.out_domains}"
    )
    all_domains = set(args.in_domains + args.out_domains)
    if isinstance(args.patch_size, int):
        args.patch_size = {
            domain: (args.patch_size, args.patch_size)
            for domain in all_domains
        }

    input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=tuple(args.patch_size[domain]),
            image_size=args.input_size[domain],
        )
        for domain in args.in_domains
    }

    output_adapters = {
        domain: DOMAIN_CONF[domain]['output_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=tuple(args.patch_size[domain]),
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task=domain,
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn,
            image_size=args.input_size[domain],
        )
        for domain in args.out_domains
    }

    if 'large' in args.model:
        model = MIRAGEModel(
            args=args,
            input_adapters=input_adapters,
            output_adapters=output_adapters,
            num_global_tokens=args.num_global_tokens,
            drop_path_rate=args.drop_path,
            dim_tokens=1024,
            depth=24,
            num_heads=16,
        )
    else:
        model = MIRAGEModel(
            args=args,
            input_adapters=input_adapters,
            output_adapters=output_adapters,
            num_global_tokens=args.num_global_tokens,
            drop_path_rate=args.drop_path,
        )

    return model


class MIRAGEWrapper(nn.Module):
    def __init__(
        self,
        input_size=512,
        patch_size=32,
        all_tokens=False,
        modalities="bscan",
        weights=None,
        map_location="cpu",
    ):
        super().__init__()

        assert weights is not None
        state_dict = torch.load(weights, map_location=map_location, weights_only=False)
        model_state_dict = state_dict["model"]

        self.args = state_dict["args"]

        modalities = self.args.in_domains
        self.args.in_domains = modalities
        self.args.input_size = {}
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        for domain in modalities:
            if domain != "bscanlayermap":
                self.args.input_size[domain] = input_size
            else:
                self.args.input_size[domain] = (128, 128)
        self.args.patch_size = {}
        for domain in modalities:
            if domain != "bscanlayermap":
                self.args.patch_size[domain] = patch_size
            else:
                self.args.patch_size[domain] = (8, 8)
        self.args.grid_size = {}
        for domain in modalities:
            self.args.grid_size[domain] = []
            for i in range(len(input_size)):
                self.args.grid_size[domain].append(input_size[i] // patch_size[i])

        self.model = get_model(self.args)
        self.all_tokens = all_tokens
        print('>> Loading weights from:', weights)
        self.model.load_state_dict(model_state_dict, strict=True)

    def forward(self, x: dict):
        """
        Args:
            x: (B, C, H, W) tensor. H and W are determined by the
            input_size parameter in the constructor. It expects a tensor
            in the range [0, 1].
        Returns:
            (B, C, H, W) tensor
        """
        for k, v in x.items():
            x[k] = v.to(self.device)
        masks = {}
        for k in self.args.in_domains:
            if k not in x:
                if k == 'bscanlayermap':
                    x[k] = torch.zeros((1, *self.args.input_size[k])).long()
                else:
                    x[k] = torch.zeros((1, 1, *self.args.input_size[k]))
                fill_v = 1
            else:
                fill_v = 0
            print('Input:', k, x[k].shape, x[k].min(), x[k].max())
            mask = np.full(self.args.grid_size[k], fill_v)
            masks[k] = torch.LongTensor(mask).flatten()[None].to(self.device)
        preds, _masks = self.model(
            x,
            mask_inputs=False,
            task_masks=masks,
        )
        return preds

    @property
    def device(self):
        return next(self.parameters()).device


def to_tensor(fn):
    fn = str(fn)
    if fn.endswith('.jpeg') or fn.endswith('.jpg') or fn.endswith('.png'):
        img = io.imread(fn)
        img = img[..., 0]
    elif fn.endswith('.npy'):
        img = np.load(fn)
    else:
        raise ValueError('Unsupported file format:', fn.split('.')[-1])
    if 'layermap' in fn:
        img = resize(img, (128, 128), order=0, preserve_range=True, anti_aliasing=False)
        img = torch.tensor(img).unsqueeze(0).long()
    else:
        img = resize(img, (512, 512), order=1, preserve_range=True, anti_aliasing=True)
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
        # Normalize to [0, 1]
        img = img / 255.0
    print('Input:', Path(fn).stem, img.dtype, img.shape, img.min(), img.max())
    return img



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', action='store_true', help='Extract features only')
    parser.add_argument('--model_size', type=str, default='base', help='Model size: base or large')
    parser.add_argument('--image_path', type=str, default='./_example_images', help='Path to input images')
    args = parser.parse_args()

    assert args.model_size in ['base', 'large']

    # NOTE: ViT-Base and ViT-Large versions of MIRAGE are available
    if args.model_size == 'base':
        weights = './__weights/MIRAGE-Base.pth'
    else:
        weights = './__weights/MIRAGE-Large.pth'

    model = MIRAGEWrapper(weights=weights)
    model.eval()
    if args.features:
        model.model.output_adapters = None

    for fsid in Path(args.image_path).iterdir():
        bscan = to_tensor(fsid / 'bscan.npy')
        slo = to_tensor(fsid / 'slo.npy')
        bscanlayermap = to_tensor(fsid / 'bscanlayermap.npy')

        # NOTE: uncomment to test with different input modalities
        input_data = {
            'bscan': bscan,
            # 'slo': slo,
            # 'bscanlayermap': bscanlayermap,
        }

        with torch.no_grad():
            out = model(input_data)
            if args.features:
                print(out.shape)
                np.save(fsid / f'__out_features.npy', out.cpu().numpy())
            else:
                print('Outputs:')
                for k, v in out.items():
                    print('\t', k, v.shape, v.min(), v.max())
                    if 'layermap' in k:
                        v = v.argmax(1) / 12
                    save_image(v, fsid / f'__out_{k}.png')
