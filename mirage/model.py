import itertools
import math
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor
from torch import nn
from torch.distributions.dirichlet import Dirichlet
from einops import repeat, rearrange

from mirage.utils import Block, trunc_normal_
from mutils.factory import get_factory_adder



add_model, model_factory = get_factory_adder()



class MIRAGEModel(nn.Module):
    """MIRAGE model.
    This module performs masking in its forward pass.
    The MultiViT module defined below inherits from this module and performs a regular forward pass,
    and should be used instead for downstream tasks


    Args:
         input_adapters: Dictionary of task -> input adapters
         output_adapters: Optional dictionary of task -> output adapters
         num_global_tokens: Number of additional global tokens to add (like cls tokens), default is 1
         dim_tokens: Dimension of encoder tokens
         depth: Depth of encoder
         num_heads: Number of attention heads
         mlp_ratio: MLP hidden dim ratio
         qkv_bias: Set to False to disable bias
         drop_rate: Dropout after MLPs and Attention
         attn_drop_rate: Attention matrix drop rate
         drop_path_rate: DropPath drop rate
         norm_layer: Type of normalization layer
    """
    def __init__(
        self,
        args,
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        num_global_tokens: int = 1,
        dim_tokens: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: partial[nn.LayerNorm] = partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()

        self.args = args

        # Initialize input and output adapters
        for adapter in input_adapters.values():
            adapter.init(dim_tokens=dim_tokens)
        self.input_adapters = nn.ModuleDict(input_adapters)
        if output_adapters is not None:
            for adapter in output_adapters.values():
                adapter.init(dim_tokens_enc=dim_tokens)
            self.output_adapters = nn.ModuleDict(output_adapters)
        else:
            self.output_adapters = None

        # Additional learnable tokens that can be used by encoder to process/store global information
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, dim_tokens))
        trunc_normal_(self.global_tokens, std=0.02)

        # Transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.encoder = nn.Sequential(*[
            Block(
                dim=dim_tokens,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
            elif isinstance(m, nn.Conv2d):
                if '.proj' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.input_info = None
        self.token_dist = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.encoder)

    @torch.jit.ignore  # type: ignore
    def no_weight_decay(self):
        no_wd_set = {'global_tokens'}

        for task, adapter in self.input_adapters.items():
            if hasattr(adapter, 'no_weight_decay'):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f'input_adapters.{task}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        if self.output_adapters is not None:
            for task, adapter in self.output_adapters.items():
                if hasattr(adapter, 'no_weight_decay'):
                    to_skip = adapter.no_weight_decay()
                    to_skip = set([f'output_adapters.{task}.{name}' for name in to_skip])
                    no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def sample_alphas(
        self,
        B: int,
        n_tasks: int,
        alphas: Union[float, List[float], Tensor] = 1.0,
        eps: float = 1e-5
    ):
        """Sample alphas for Dirichlet sampling such that tasks are
        first uniformly chosen and then Dirichlet sampling is performed
        over the chosen ones.

        Args:
            B: Batch size
            n_tasks: Number of input tasks
            alphas: Float or list to multiply task choices {0,1} by
            eps: Small constant since Dirichlet alphas need to be positive
        """
        valid_task_choices = torch.Tensor([list(i) for i in itertools.product([0, 1], repeat=n_tasks)][1:])
        rand_per_sample_choice = torch.randint(0, len(valid_task_choices), (B,))
        alphas_tensor = torch.index_select(valid_task_choices, 0, rand_per_sample_choice)
        alphas_tensor = alphas_tensor * torch.tensor(alphas) + eps
        return alphas_tensor

    def generate_random_masks(
        self,
        input_tokens: Dict[str, torch.Tensor],
        num_encoded_tokens: int,
        alphas: Union[float, List[float], Tensor] = 1.0,
        sample_tasks_uniformly: bool = False,
    ):
        """Sample a total of num_encoded_tokens from different tasks
        using Dirichlet sampling.

        Args:
            input_tokens: Dictionary of tensors to sample num_encoded_tokens from
            num_encoded_tokens: Number of tokens to select
            alphas: Dirichlet distribution parameter alpha. Lower alpha = harder,
                less uniform sampling. Can be float or list of floats.
            sample_tasks_uniformly: Set to True to first sample 1-n_tasks uniformly at random
                for each sample in the batch. Dirichlet sampling is then done over selected subsets.
        """
        B = list(input_tokens.values())[0].shape[0]
        device = list(input_tokens.values())[0].device

        if self.token_dist is None:
            total_tokens = 0
            for domain, tensor in input_tokens.items():
                print(domain, tensor.shape[1])
                total_tokens += tensor.shape[1]
            token_dist = {}
            for domain, tensor in input_tokens.items():
                token_dist[domain] = tensor.shape[1] / total_tokens
            self.token_dist = token_dist
            # Order the dictionary by value
            self.token_dist = dict(sorted(self.token_dist.items(), key=lambda item: item[1], reverse=True))
            print('> Token distribution:', self.token_dist)

        alphas = [alphas] * len(input_tokens) if isinstance(alphas, float) else alphas
        if sample_tasks_uniformly:
            alphas = self.sample_alphas(B, len(input_tokens), alphas=alphas)
            task_sampling_dist = Dirichlet(alphas).sample().to(device)
        else:
            task_sampling_dist = Dirichlet(torch.Tensor(alphas)).sample((B,)).to(device)

        samples_per_task = (task_sampling_dist * num_encoded_tokens).round().long()

        task_masks = []
        num_tokens_per_task = [task_tokens.shape[1] for task_tokens in input_tokens.values()]
        # print(samples_per_task, print(input_tokens.keys()))
        for i, num_tokens in enumerate(num_tokens_per_task):
            # Use noise to shuffle arange
            noise = torch.rand(B, num_tokens, device=device)  # noise in [0, 1]
            ids_arange_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            mask = torch.arange(num_tokens, device=device).unsqueeze(0).expand(B, -1)
            mask = torch.gather(mask, dim=1, index=ids_arange_shuffle)
            # 0 is keep (unmasked), 1 is remove (masked)
            mask = torch.where(mask < samples_per_task[:, i].unsqueeze(1), 0, 1)
            task_masks.append(mask)

        mask_all = torch.cat(task_masks, dim=1)
        ids_shuffle = torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_encoded_tokens]

        # Update binary mask to adjust for task rounding
        mask_all = torch.ones_like(mask_all)
        mask_all[:, :num_encoded_tokens] = 0
        # Unshuffle to get the binary mask
        mask_all = torch.gather(mask_all, dim=1, index=ids_restore)
        # Split to get task masks
        task_masks = torch.split(mask_all, num_tokens_per_task, dim=1)
        # Convert to dict
        task_masks = {domain: mask for domain, mask in zip(input_tokens.keys(), task_masks)}

        return task_masks, ids_keep, ids_restore

    @staticmethod
    def make_mask(
        N_H,
        N_W,
        xy_idxs,
        full_tasks=[],
        indicate_visible=True,
        flatten=True,
        device='cuda'
    ):
        """Creates masks for each task, given lists of un-masked x,y
        coordinates.
        """
        xy_idxs = {
            k: torch.LongTensor(v)
            for k, v in xy_idxs.items()
        }

        task_masks = {
            k: torch.ones(N_H, N_W).to(device)
            for k in xy_idxs.keys()
        }

        for k in xy_idxs.keys():
            if len(xy_idxs[k]) > 0:
                task_masks[k][xy_idxs[k][:, 1], xy_idxs[k][:, 0]] = 0

        for task in full_tasks:
            task_masks[task][:] = 0

        if not indicate_visible:
            task_masks = {k: 1 - v for k, v in task_masks.items()}

        if flatten:
            task_masks = {k: v.flatten().unsqueeze(0) for k, v in task_masks.items()}

        return task_masks

    def generate_input_info(self, input_task_tokens, image_size):
        input_info = OrderedDict()
        i = 0
        input_info['tasks'] = {}
        for domain, tensor in input_task_tokens.items():
            num_tokens = tensor.shape[1]
            d = {
                'num_tokens': num_tokens,
                'has_posemb': True,
                'start_idx': i,
                'end_idx': i + num_tokens,
            }
            if isinstance(image_size, dict):
                d['image_size'] = image_size[domain]
            if self.args.grid_sizes is not None:
                d['grid_size'] = self.args.grid_sizes[domain]
            i += num_tokens
            input_info['tasks'][domain] = d

        if isinstance(image_size, int):
            input_info['image_size'] = image_size
        input_info['num_task_tokens'] = i
        input_info['num_global_tokens'] = self.num_global_tokens

        return input_info

    def forward(
        self,
        x: Union[Dict[str, torch.Tensor], torch.Tensor],
        mask_inputs: bool = True,
        task_masks: Optional[Dict[str, torch.Tensor]] = None,
        num_encoded_tokens: int = 128,
        alphas: Union[float, List[float]] = 1.0,
        sample_tasks_uniformly: bool = False,
        return_all_layers: bool = False,
        reshape: bool = False,
    ):
        """
        Forward pass through input adapters, transformer encoder and output adapters.
        If specified, will randomly drop input tokens.

        Args:
            x: Input tensor or dictionary of tensors
            mask_inputs: Set to True to enable random masking of input patches
            task_masks: Optional dictionary of task->mask pairs.
            num_encoded_tokens: Number of tokens to randomly select for encoder.
                Only used if mask_inputs is True.
            alphas: Dirichlet distribution parameter alpha for task sampling.
                Higher alpha = harder, less uniform sampling. Can be float or list of floats.
            sample_tasks_uniformly: Set to True if tasks should be uniformly presampled,
                before Dirichlet sampling decides share of masked tokens between them.
            return_all_layers: Set to True to return the features of all transformer layers,
        """

        ## Processing input modalities
        # If input x is a Tensor, assume it's RGB
        x = {'bscan': x} if isinstance(x, Tensor) else x

        # Need image size for tokens->image reconstruction
        # We assume that at least one of rgb or semseg is given as input before masking
        if 'semseg' in x:
            B, H, W = x['semseg'].shape
            H *= self.input_adapters['semseg'].stride_level
            W *= self.input_adapters['semseg'].stride_level
        elif 'bscan' in x:
            B, _C, H, W = x['bscan'].shape
        elif 'slo' in x:
            B, _C, H, W = x['slo'].shape
        else:
            # TODO: Deal with case where not all have same shape
            B, _C, H, W = list(x.values())[0].shape

        # Encode selected inputs to tokens
        input_task_tokens = {
            domain: self.input_adapters[domain](tensor)
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }
        if self.input_info is None:
            self.input_info = self.generate_input_info(
                input_task_tokens=input_task_tokens,
                image_size=self.args.input_size
            )
        input_info = self.input_info

        # Select random subset of tokens from the chosen input tasks and concatenate them
        if mask_inputs:
            num_encoded_tokens = num_encoded_tokens if num_encoded_tokens is not None else self.num_encoded_tokens
        else:
            num_encoded_tokens = sum([tensor.shape[1] for tensor in input_task_tokens.values()])

        ## Generating masks
        if task_masks is None:
            task_masks, ids_keep, ids_restore = self.generate_random_masks(
                input_task_tokens,
                num_encoded_tokens,
                alphas=alphas,
                sample_tasks_uniformly=sample_tasks_uniformly,
            )
        else:
            mask_all = torch.cat([task_masks[task] for task in input_task_tokens.keys()], dim=1)
            ids_shuffle = torch.argsort(mask_all, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :(mask_all == 0).sum()]

        input_tokens = torch.cat([task_tokens for task_tokens in input_task_tokens.values()], dim=1)

        # Apply mask
        input_tokens = torch.gather(input_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, input_tokens.shape[2]))

        # Add global tokens to input tokens
        global_tokens = repeat(self.global_tokens, '() n d -> b n d', b=B)
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)

        if return_all_layers:
            features = OrderedDict()
            n = 0
            for _i, block in enumerate(self.encoder):
                input_tokens = block(input_tokens)
                # if i % 2 == 1:  # Save every other layer
                current_features = input_tokens[:, :-self.num_global_tokens]
                current_features = rearrange(
                    current_features, 'b (h w) d -> b d h w',
                    h=self.args.grid_sizes['bscan'][0], w=self.args.grid_sizes['bscan'][1]
                )
                features[f'layer_{n}'] = current_features
                n += 1
            return features

        ## Transformer forward pass
        encoder_tokens = self.encoder(input_tokens)

        ## Output decoders
        if self.output_adapters is None:
            if reshape:
                encoder_tokens = rearrange(
                    encoder_tokens[:, :-self.num_global_tokens], 'b (h w) d -> b d h w',
                    h=self.args.grid_sizes['bscan'][0], w=self.args.grid_sizes['bscan'][1]
                )
            return encoder_tokens, task_masks

        # Decode tokens for each task using task-specific output adapters
        preds = {
            domain: self.output_adapters[domain](
                encoder_tokens=encoder_tokens,
                input_info=input_info,
                ids_keep=ids_keep,
                ids_restore=ids_restore,
            )
            for domain in self.output_adapters
        }

        return preds, task_masks


@add_model
def miragepre_base(
    input_adapters: Dict[str, nn.Module],
    output_adapters: Optional[Dict[str, nn.Module]],
    args,
    **kwargs
):
    model = MIRAGEModel(
        args,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@add_model
def miragepre_large(
    input_adapters: Dict[str, nn.Module],
    output_adapters: Optional[Dict[str, nn.Module]],
    args,
    **kwargs
):
    model = MIRAGEModel(
        args,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


class MIRAGELight(MIRAGEModel):
    """MultiViT: Multi-modal Vision Transformer
    This is MIRAGE without masking and with a simplified / faster forward pass

    Args:
        input_adapters: Dictionary of task -> input adapters
        output_adapters: Optional dictionary of task -> output adapters
        num_global_tokens: Number of additional global tokens to add (like cls tokens), default is 1
        dim_tokens: Dimension of encoder tokens
        depth: Depth of encoder
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        qkv_bias: Set to False to disable bias
        drop_rate: Dropout after MLPs and Attention
        attn_drop_rate: Attention matrix drop rate
        drop_path_rate: DropPath drop rate
        norm_layer: Type of normalization layer
    """

    def process_input(self, x):

        # If input x is a Tensor, assume it's RGB
        x = {'bscan': x} if isinstance(x, torch.Tensor) else x
        # Need image size for tokens->image reconstruction
        if 'bscan' in x:
            B, _, H, W = x['bscan'].shape
        elif 'semseg' in x:
            B, H, W = x['semseg'].shape
            H *= self.input_adapters['semseg'].stride_level
            W *= self.input_adapters['semseg'].stride_level
        else:
            # TODO: Deal with case where not all have same shape
            B, _, H, W = list(x.values())[0].shape

        # Encode selected inputs to tokens
        input_task_tokens = {
            domain: self.input_adapters[domain](tensor)
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }

        input_info = self.generate_input_info(input_task_tokens=input_task_tokens, image_size=(H, W))
        input_tokens = torch.cat([task_tokens for task_tokens in input_task_tokens.values()], dim=1)

        # Add global tokens to input tokens
        global_tokens = repeat(self.global_tokens, '() n d -> b n d', b=B)
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)

        return input_tokens, input_info

    def forward(  # type: ignore
        self,
        x: Union[Dict[str, torch.Tensor], torch.Tensor],
        return_all_layers=False,
        **kwargs
    ):
        """Forward pass through input adapters, transformer encoder and
        output adapters.

        Args:
            x: Input tensor or dictionary of tensors
            return_all_layers: Set to True to return all transformer layers
        """

        input_tokens, input_info = self.process_input(x)

        # Pass tokens through Transformer
        if not return_all_layers:
            encoder_tokens = self.encoder(input_tokens)
        else:
            # Optionally access every intermediate layer
            encoder_tokens = []
            tokens = input_tokens
            for block in self.encoder:
                tokens = block(tokens)
                encoder_tokens.append(tokens)

        if self.output_adapters is None:
            return encoder_tokens

        # Decode tokens for each task using task-specific output adapters
        preds = {
            domain: self.output_adapters[domain](
                encoder_tokens=encoder_tokens,
                input_info=input_info,
            )
            for domain in self.output_adapters
        }

        return preds


@add_model
def miragelight_base(
    input_adapters: Dict[str, nn.Module],
    output_adapters: Optional[Dict[str, nn.Module]],
    args,
    **kwargs
):
    return MIRAGELight(
        args,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )


@add_model
def miragelight_large(
    input_adapters: Dict[str, nn.Module],
    output_adapters: Optional[Dict[str, nn.Module]],
    args,
    **kwargs
):
    return MIRAGELight(
        args,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
