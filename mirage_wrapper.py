from functools import partial
from pathlib import Path
import copy
import argparse
from typing import Union

import numpy as np
import torch
from torch import nn
from skimage import io
from skimage.transform import resize
from torchvision.utils import save_image

from mirage.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from mirage.output_adapters import SpatialOutputAdapter
from mirage.model import MIRAGEModel
from mirage.utils import pair
from mutils.factory import get_factory_adder



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


class MIRAGEWrapper(nn.Module):
    def __init__(
        self,
        input_size=512,
        patch_size=32,
        modalities='bscan-slo-bscanlayermap',
        weights=None,
        device='cuda',
    ):
        super().__init__()

        assert weights is not None
        state_dict = torch.load(weights, map_location=device, weights_only=False)
        model_state_dict = state_dict["model"]

        args = state_dict["args"]

        args.in_domains = modalities.split('-')
        input_size = pair(input_size)
        patch_size = pair(patch_size)
        assert input_size is not None
        assert patch_size is not None
        args.patch_size = {}
        args.input_size = {}
        args.grid_size = {}
        for domain in args.in_domains:
            if domain != "bscanlayermap":
                args.patch_size[domain] = patch_size
                args.input_size[domain] = input_size
            else:
                args.patch_size[domain] = (8, 8)
                args.input_size[domain] = (128, 128)
            args.grid_size[domain] = []
            for i in range(len(input_size)):
                args.grid_size[domain].append(input_size[i] // patch_size[i])

        self.args = args
        self.model = self.get_model()
        msg = self.model.load_state_dict(model_state_dict, strict=False)
        # Print number of elements in each error message
        print('  Missing keys:', len(msg.missing_keys))
        print('  Unexpected keys:', len(msg.unexpected_keys))
        assert len(msg.missing_keys) == 0

    def get_output_adapters(self) -> Union[None, dict]:
        return {
            domain: DOMAIN_CONF[domain]['output_adapter'](
                stride_level=DOMAIN_CONF[domain]['stride_level'],
                patch_size_full=tuple(self.args.patch_size[domain]),
                dim_tokens=self.args.decoder_dim,
                depth=self.args.decoder_depth,
                num_heads=self.args.decoder_num_heads,
                use_task_queries=self.args.decoder_use_task_queries,
                task=domain,
                context_tasks=list(self.args.in_domains),
                use_xattn=self.args.decoder_use_xattn,
                image_size=self.args.input_size[domain],
            )
            for domain in self.args.out_domains
        }

    def get_model(self):
        """Creates and returns model from arguments."""
        print(
            f"Creating model: {self.args.model} for inputs {self.args.in_domains}"
            f" and outputs {self.args.out_domains}"
        )

        input_adapters = {
            domain: DOMAIN_CONF[domain]['input_adapter'](
                stride_level=DOMAIN_CONF[domain]['stride_level'],
                patch_size_full=tuple(self.args.patch_size[domain]),
                image_size=self.args.input_size[domain],
            )
            for domain in self.args.in_domains
        }

        output_adapters = self.get_output_adapters()

        if 'large' in self.args.model:
            model = MIRAGEModel(
                args=self.args,
                input_adapters=input_adapters,
                output_adapters=output_adapters,
                num_global_tokens=self.args.num_global_tokens,
                drop_path_rate=self.args.drop_path,
                dim_tokens=1024,
                depth=24,
                num_heads=16,
            )
        elif 'base' in self.args.model:
            model = MIRAGEModel(
                args=self.args,
                input_adapters=input_adapters,
                output_adapters=output_adapters,
                num_global_tokens=self.args.num_global_tokens,
                drop_path_rate=self.args.drop_path,
            )
        else:
            raise ValueError('Unknown model size:', self.args.model)

        return model

    def forward(self, x: dict):
        """
        Args:
            Dict[x, (B, C, H, W) tensor]. H and W are determined by the
            input_size parameter in the constructor. It expects a tensor
            in the range [0, 1].

        Returns:
            (B, C, H, W) tensor
        """
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
            x[k] = x[k].to(self.device)
        preds, _masks = self.model(
            x,
            mask_inputs=False,
            task_masks=masks,
        )
        return preds

    @property
    def device(self):
        return next(self.parameters()).device



add_miragecls, miragecls_factory = get_factory_adder()


@add_miragecls('global')
class MIRAGEClsGlobal(MIRAGEWrapper):
    def __init__(self, num_classes=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes > 0
        assert len(self.args.in_domains) == 1
        self.num_classes = num_classes
        self.model.output_adapters = None
        # Get the embedding dimension from the first layer norm of the
        #   type: (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.embed_dim = self.model.encoder[0].norm1.normalized_shape[0]
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-06, elementwise_affine=True)
        print('  Embedding dimension:', self.embed_dim)
        self.build_head()

    def build_head(self, factor=1):
        self.head = nn.Linear(self.embed_dim * factor, self.num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) tensor. H and W are determined by the
            input_size parameter in the constructor. It expects a tensor
            in the range [0, 1].
        Returns:
            (B, C, H, W) tensor
        """
        x_d = {self.args.in_domains[0]: x}
        out, _masks = self.model(x_d, mask_inputs=False)
        out = self.norm(out)
        out = self.pool(out)
        return self.head(out)

    def pool(self, x):
        return x[:, :-self.args.num_global_tokens, :].mean(dim=1)

    def get_output_adapters(self):
        return None


@add_miragecls('cls')
class MIRAGEClsCLS(MIRAGEClsGlobal):
    def pool(self, x):
        return x[:, -self.args.num_global_tokens:, :].mean(dim=1)


@add_miragecls('token_mix')
class MIRAGEClsTokenMix(MIRAGEClsGlobal):
    def build_head(self, factor=2):
        super().build_head(factor)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :].mean(dim=1)
        return torch.cat([patch, global_], dim=1)


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
    parser.add_argument('--model_size', type=str, default='base', help='Model size', choices=['base', 'large'])
    parser.add_argument('--image_path', type=str, default='./_example_images', help='Path to input images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    # NOTE: ViT-Base and ViT-Large versions of MIRAGE are available
    if args.model_size == 'base':
        weights = './__weights/MIRAGE-Base.pth'
    else:
        weights = './__weights/MIRAGE-Large.pth'

    model = MIRAGEWrapper(weights=weights, device=args.device)
    model.eval()
    if args.features:
        model.model.output_adapters = None

    print(f'Using device: {model.device}')

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
