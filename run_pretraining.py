import datetime
import yaml
import socket
import argparse
import json
import math
import os
import sys
import time
import datetime
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List
import copy

import torch
from torch.amp.autocast_mode import autocast
from torch.nn import functional as F
from einops import rearrange
from torchvision import utils as vutils

from mirage.model import model_factory
from mirage.criterion import MaskedCrossEntropyLoss, MaskedMSELoss
from mirage.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from mirage.output_adapters import SpatialOutputAdapter
from mutils import misc
from mutils import logger
from mutils.native_scaler import NativeScalerWithGradNormCount as NativeScaler
from mutils import native_scaler
from mutils.datasets_pretrain import build_mirage_pretraining_dataset
from mutils.optim_factory import create_optimizer
from mutils.factory import get_factory_adder
from mutils import checkpoint



DEFAULT_CONF = {
    'channels': 1,
    'stride_level': 1,
    'input_adapter': partial(PatchedInputAdapter, num_channels=1),
    'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
    'loss': MaskedMSELoss,
}

DOMAIN_CONF = {
    'bscan': copy.deepcopy(DEFAULT_CONF),
    'slo': copy.deepcopy(DEFAULT_CONF),
    'bscanlayermap': {
        'num_classes': 13,
        'stride_level': 1,
        'input_adapter': partial(SemSegInputAdapter, num_classes=13,
                                 dim_class_emb=64, interpolate_class_emb=False),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=13),
        'loss': partial(MaskedCrossEntropyLoss, label_smoothing=0.0),
    },
}

def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument(
        '-c', '--config', default='', type=str, metavar='FILE',
        help='YAML config file specifying default arguments. (default: %(default)s)'
    )

    parser = argparse.ArgumentParser('MIRAGE pre-training script', add_help=True)

    parser.add_argument(
        '--batch_size', default=256, type=int,
        help='Batch size per GPU. (default: %(default)s)'
    )
    parser.add_argument(
        '--epochs', default=1600, type=int,
        help='Number of epochs. (default: %(default)s)'
    )
    parser.add_argument(
        '--save_ckpt_freq', default=20, type=int,
        help='Checkpoint saving frequency in epochs. (default: %(default)s)'
    )

    # Task parameters
    parser.add_argument(
        '--in_domains', default='bscan-slo-bscanlayermap', type=str,
        help='Input domain names, separated by hyphen. (default: %(default)s)'
    )
    parser.add_argument(
        '--out_domains', default='rgb-depth-semseg', type=str,
        help='Output domain names, separated by hyphen. (default: %(default)s)'
    )

    # Model parameters
    parser.add_argument(
        '--model', default='miragepre_base', type=str, metavar='MODEL',
        help='Name of model to train. (default: %(default)s)'
    )
    parser.add_argument(
        '--num_encoded_tokens', default=98, type=int,
        help='Number of tokens to randomly choose for encoder. (default: %(default)s)'
    )
    parser.add_argument(
        '--perc_encoded_tokens', default=None, type=float,
        help='Percentage of tokens to randomly choose for encoder. (default: %(default)s)'
    )
    parser.add_argument(
        '--num_global_tokens', default=1, type=int,
        help='Number of global tokens to add to encoder. (default: %(default)s)'
    )
    parser.add_argument(
        '--patch_size', default=32, type=int,
        help='Base patch size for image-like modalities. (default: %(default)s)'
    )
    parser.add_argument(
        '--input_size', default=512, type=int,
        help='Images input size for backbone. (default: %(default)s)'
    )
    parser.add_argument(
        '--alphas', type=float, default=1.0,
        help='Dirichlet alphas concentration parameter. (default: %(default)s)'
    )
    parser.add_argument(
        '--sample_tasks_uniformly', default=False, action='store_true',
        help='Set to True/False to enable/disable uniform sampling over tasks'
            ' to sample masks for. (default: %(default)s)'
    )

    parser.add_argument(
        '--decoder_use_task_queries', default=True, action='store_true',
        help='Set to True/False to enable/disable adding of task-specific'
            ' tokens to decoder query tokens. (default: %(default)s)'
    )
    parser.add_argument(
        '--decoder_use_xattn', default=True, action='store_true',
        help='Set to True/False to enable/disable decoder cross attention.'
            ' (default: %(default)s)'
    )
    parser.add_argument(
        '--decoder_dim', default=256, type=int,
        help='Token dimension inside the decoder layers. (default: %(default)s)'
    )
    parser.add_argument(
        '--decoder_depth', default=2, type=int,
        help='Number of self-attention layers after the initial cross attention.'
            ' (default: %(default)s)'
    )
    parser.add_argument(
        '--decoder_num_heads', default=8, type=int,
        help='Number of attention heads in decoder. (default: %(default)s)'
    )
    parser.add_argument(
        '--drop_path', type=float, default=0.0, metavar='PCT',
        help='Drop path rate. (default: %(default)s)'
    )

    parser.add_argument(
        '--loss_on_unmasked', default=False, action='store_true',
        help='Set to True/False to enable/disable computing the loss on'
            ' non-masked tokens. (default: %(default)s)'
    )
    parser.add_argument('--no_loss_on_unmasked', action='store_false', dest='loss_on_unmasked')
    parser.set_defaults(loss_on_unmasked=False)

    # Optimizer parameters
    parser.add_argument(
        '--opt', default='adamw', type=str, metavar='OPTIMIZER',
        help='Optimizer. (default: %(default)s)'
    )
    parser.add_argument(
        '--opt_eps', default=1e-8, type=float, metavar='EPSILON',
        help='Optimizer epsilon. (default: %(default)s)'
    )
    parser.add_argument(
        '--opt_betas', default=[0.9, 0.95], type=float, nargs='+', metavar='BETA',
        help='Optimizer betas. (default: %(default)s)'
    )
    parser.add_argument(
        '--clip_grad', type=float, default=None, metavar='CLIPNORM',
        help='Clip gradient norm. (default: %(default)s)'
    )
    parser.add_argument(
        '--skip_grad', type=float, default=None, metavar='SKIPNORM',
        help='Skip update if gradient norm larger than threshold.'
            ' (default: %(default)s)'
    )
    parser.add_argument(
        '--momentum', type=float, default=0.9, metavar='M',
        help='SGD momentum. (default: %(default)s)'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.05,
        help='Weight decay. (default: %(default)s)'
    )
    parser.add_argument(
        '--weight_decay_end', type=float, default=None,
        help='Final value of the weight decay. We use a cosine schedule for WD'
            ' (Set the same value as args.weight_decay to keep weight decay unchanged).'
            ' (default: %(default)s)'
    )
    parser.add_argument('--decoder_decay', type=float, default=None, help='decoder weight decay')

    parser.add_argument(
        '--blr', type=float, default=1e-4, metavar='LR',
        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256.'
            ' (default: %(default)s)'
    )
    parser.add_argument(
        '--warmup_lr', type=float, default=1e-6, metavar='LR',
        help='Warmup learning rate. (default: %(default)s)'
    )
    parser.add_argument(
        '--min_lr', type=float, default=0., metavar='LR',
        help='Lower lr bound for cyclic schedulers that hit 0. (default: %(default)s)'
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=40, metavar='N',
        help='Epochs to warmup LR, if scheduler supports. (default: %(default)s)'
    )
    parser.add_argument(
        '--warmup_steps', type=int, default=-1, metavar='N',
        help='Epochs to warmup LR, if scheduler supports. (default: %(default)s)'
    )

    # Augmentation parameters
    parser.add_argument(
        '--hflip', type=float, default=0.5,
        help='Probability of horizontal flip. (default: %(default)s)'
    )
    parser.add_argument(
        '--intensity_shift', type=float, default=0.1,
        help='Intensity shift. (default: %(default)s)'
    )
    parser.add_argument(
        '--train_interpolation', type=str, default='bicubic',
        help='Training interpolation (random, bilinear, bicubic). (default: %(default)s)'
    )
    parser.add_argument(
        '--random-crop', type=float, default=1.0,
        help='Random crop. (default: %(default)s)'
    )
    parser.add_argument(
        '--affine', action='store_true',
        help='Apply random affine transformations. (default: %(default)s)',
    )
    parser.add_argument('--no_affine', action='store_false', dest='affine')
    parser.set_defaults(affine=True)

    # Misc.
    parser.add_argument(
        '--base_output_dir', default='./__output/pre', type=str,
        help='Path to save logs and checkpoints. (default: %(default)s)'
    )
    parser.add_argument(
        '--device', default='cuda',
        help='Device to use for training / testing. (default: %(default)s)'
    )

    parser.add_argument(
        '--seed', default=0, type=int,
        help='Random seed. (default: %(default)s)'
    )
    parser.add_argument(
        '--resume', default='',
        help='Resume from checkpoint. (default: %(default)s)'
    )
    parser.add_argument(
        '--auto_resume', action='store_true',
        help='Resume from the latest checkpoint, if any. (default: %(default)s)'
    )
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N',
        help='Start epoch. (default: %(default)s)'
    )
    parser.add_argument(
        '--num_workers', default=10, type=int,
        help='Number of workers for data loader. (default: %(default)s)'
    )
    parser.add_argument(
        '--pin_mem', action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes)'
            ' transfer to GPU. (default: %(default)s)'
    )
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument(
        '--show_user_warnings', default=False, action='store_true',
        help='Show user warnings. (default: %(default)s)'
    )

    parser.add_argument(
        '--print-model', action='store_true',
        help='Print model architecture. (default: %(default)s)'
    )

    required_parser = parser.add_argument_group('required arguments')
    required_parser.add_argument(
        '--weights', required=True, type=str,
        help='Pretrain from checkpoint (model weights). (required)'
    )
    required_parser.add_argument(
        '--data_path', required=True, type=str,
        help='Dataset path. (required)'
    )

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    if args.perc_encoded_tokens is not None:
        total_tokens = 0
        for domain, size in args.input_size.items():
            current_tokens = 1
            for i, s in enumerate(size):
                current_tokens *= s // args.patch_size[domain][i]
            total_tokens += current_tokens
        print('Computing number of encoded tokens based on percentage:', args.perc_encoded_tokens)
        args.num_encoded_tokens = int(total_tokens * args.perc_encoded_tokens)
        print(f'> Number of encoded tokens: {args.num_encoded_tokens} (out of {total_tokens})')

    domains = args.in_domains.split('-')
    if isinstance(args.patch_size, int):
        args.patch_size = {d: (args.patch_size, args.patch_size) for d in domains}

    if isinstance(args.input_size, int):
        args.input_size = {d: (args.input_size, args.input_size) for d in domains}

    args.grid_sizes = {}
    for domain, size in args.input_size.items():
        args.grid_sizes[domain] = []
        for i, s in enumerate(size):
            args.grid_sizes[domain].append(s // args.patch_size[domain][i])

    if socket.gethostname() == 'hemingway':
        args.batch_size = 2

    # Print configuration
    args_dict = vars(args)
    # Order alphabetically
    args_dict = dict(sorted(args_dict.items()))
    print(json.dumps(args_dict, indent=2))

    return args


add_fm_config, fm_config_factory = get_factory_adder()


class FoundModel:
    def __init__(self):
        self.model: str


@add_fm_config('multimae-b')
class MIRAGEBaseFM(FoundModel):
    def __init__(self):
        self.model = 'miragepre_base'


@add_fm_config('mae_pretrain')
class MIRAGELargeFM(FoundModel):
    def __init__(self):
        self.model = 'miragepre_large'


def get_model(args):
    '''Creates and returns model from arguments
    '''
    print(f'Creating model: {args.model} for inputs {args.in_domains} and outputs {args.out_domains}')
    if isinstance(args.patch_size, int):
        args.patch_size = {domain: (args.patch_size, args.patch_size) for domain in args.all_domains}

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

    fm_config = None
    for kw in fm_config_factory.keys():
        if kw in args.weights.lower():
            fm_config = fm_config_factory[kw]()
            break
    if fm_config is None:
        raise ValueError(f"Unknown model: {args.weights}")

    model = model_factory[fm_config.model](
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path,
        args=args,
    )

    args.output_dir = os.path.join(args.base_output_dir, fm_config.__class__.__name__)
    # Create output directory
    (Path(args.output_dir) / 'debug').mkdir(exist_ok=True, parents=True)

    # Save configuration
    with open(Path(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    if args.weights:
        print(f'>> Loading weights from {args.weights}')
        weights = torch.load(args.weights, map_location=args.device, weights_only=False)['model']
        if '_vit_large' in args.weights:
            def do_replacements(key):
                return key.replace('blocks.', 'encoder.')
            weights = { do_replacements(k): v for k, v in weights.items() }
            model.load_state_dict(weights, strict=False)
        else:
            # Remove input and output adapters from the weights
            new_weights = {}
            for k, v in weights.items():
                if 'input_adapter' in k or 'output_adapter' in k:
                    continue
                new_weights[k] = v
            weights = new_weights
            msg = model.load_state_dict(weights, strict=False)
            print(msg)

    return model


def main(args):
    device = torch.device(args.device)

    misc.fix_seeds(args.seed)

    if not args.show_user_warnings:
        warnings.filterwarnings('ignore', category=UserWarning)

    args.in_domains = args.in_domains.split('-')
    args.out_domains = args.out_domains.split('-')
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))

    model = get_model(args)

    tasks_loss_fn = {
        domain: DOMAIN_CONF[domain]['loss'](
            patch_size=tuple(args.patch_size[domain]),
            stride=DOMAIN_CONF[domain]['stride_level']
        )
        for domain in args.out_domains
    }

    # Get dataset
    dataset_train = build_mirage_pretraining_dataset(args)

    num_training_steps_per_epoch = len(dataset_train) // args.batch_size

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.print_model:
        print(f'Model = %s' % str(model))
    print(f'Number of params: {n_parameters / 1e6} M')

    total_batch_size = args.batch_size
    args.lr = args.blr * total_batch_size / 256

    print('LR = %.8f' % args.lr)
    print('Batch size = %d' % total_batch_size)
    print('Number of training steps = %d' % num_training_steps_per_epoch)
    print('Number of training examples per epoch = %d' % (total_batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    print('Use step level LR & WD scheduler!')
    lr_schedule_values = native_scaler.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = native_scaler.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print('Max WD = %.7f, Min WD = %.7f' % (max(wd_schedule_values), min(wd_schedule_values)))

    checkpoint.auto_load_model(
        args=args,
        model=model,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            tasks_loss_fn=tasks_loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            max_skip_norm=args.skip_grad,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_encoded_tokens=args.num_encoded_tokens,
            in_domains=args.in_domains,
            loss_on_unmasked=args.loss_on_unmasked,
            alphas=args.alphas,
            sample_tasks_uniformly=args.sample_tasks_uniformly,
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                checkpoint.save_model(
                    args=args, model=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch
                )

        log_stats = {**{k: v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir:
            with open(os.path.join(args.output_dir, 'log.txt'), mode='a', encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    with open(Path(args.output_dir) / 'training_time.txt', 'w') as f:
        f.write(total_time_str)


def save_debug_images(preds, masks, input_dict, epoch, tasks_dict):
    to_save = None
    tasks_str = ''
    predsc = {}
    masksc = {}
    input_dictc = {}
    for task in preds:
        if task in tasks_dict:
            if tasks_str == '':
                tasks_str = task
            else:
                tasks_str += f'-{task}'
            # Create a copy of the prediction and mask
            predsc[task] = preds[task].clone().detach()
            masksc[task] = masks[task].clone().detach()
            input_dictc[task] = input_dict[task].clone().detach()
            # print(preds[task].shape, input_dict[task].shape)
            # torch.Size([8, 1, 768, 768]) torch.Size([8, 1, 768, 768])
            print(task, 'mask', masksc[task].dtype, masksc[task].shape,
                  masksc[task].sum().item(), masksc[task].numel())
            print(task, 'preds', predsc[task].dtype, predsc[task].shape,
                predsc[task].min().item(), predsc[task].max().item())
            print(task, 'input', input_dictc[task].dtype, input_dictc[task].shape,
                  input_dictc[task].min().item(), input_dictc[task].max().item())
            ## Replace unmasked patches in the prediction by the input
            # Reshape mask to image. Original shape is (B, L),
            #   where L is the number of tokens.
            #   We need to reshape it to (B, C, [D,] H, W)
            if task in ['layermaps', 'bscanlayermap']:
                # Apply softmax and undo one-hot encoding
                predsc[task] = F.softmax(predsc[task], dim=1)
                predsc[task] = predsc[task].argmax(dim=1, keepdim=True).float()
                # Add channel dimension to input
                input_dictc[task] = input_dictc[task].unsqueeze(1)
                # to 0-1 (for visualization)
                predsc[task] = predsc[task] / predsc[task].max()
                input_dictc[task] = input_dictc[task] / input_dictc[task].max()
                c = 1

            if isinstance(args.patch_size, int):
                c = input_dictc[task].shape[1]
                h = input_dictc[task].shape[2] // args.patch_size
                w = input_dictc[task].shape[3] // args.patch_size
            else:
                c = input_dictc[task].shape[1]
                h = input_dictc[task].shape[2] // args.patch_size[task][0]
                w = input_dictc[task].shape[3] // args.patch_size[task][1]

            mask = rearrange(masksc[task], 'b (c h w) -> b c h w', c=c, h=h, w=w).float()
            # print(task, 'new_mask', mask.shape, mask.sum().item(), mask.numel())
            fh, fw = input_dictc[task].shape[2:]
            mask = F.interpolate(mask, size=(fh, fw), mode='nearest')
            # 1 means masked (predicted), 0 means input
            predsc[task] = predsc[task] * mask + input_dictc[task] * (1 - mask)
            print(task, 'new_mask', mask.dtype, mask.shape, mask.sum().item(), mask.numel())
            input_to_save = input_dictc[task]
            pred_to_save = predsc[task]

            # Fixed size for saving
            input_to_save = F.interpolate(input_to_save, (128, 128), mode='nearest')
            pred_to_save = F.interpolate(pred_to_save, (128, 128), mode='nearest')
            if to_save is None:
                to_save = torch.cat([input_to_save, pred_to_save], dim=2)
            else:
                # Add a white line between images
                white_line = torch.ones(
                    (input_to_save.shape[0], input_to_save.shape[1], 8, input_to_save.shape[3]),
                    device=to_save.device
                )
                to_save = torch.cat([to_save, white_line, input_to_save, pred_to_save], dim=2)
    if to_save is not None:
        # Maximum supported image dimension is 65500 pixels
        # if to_save.shape[2] * to_save.shape[3] > 65500:
        to_save = F.interpolate(to_save, size=(to_save.shape[2], to_save.shape[3]), mode='bilinear')
        current_save_path = Path(args.output_dir) / 'debug' / f'{epoch}_{tasks_str}.jpg'
        print(current_save_path)
        vutils.save_image(to_save, current_save_path)


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    tasks_loss_fn: Dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = None,  # type: ignore
    max_skip_norm: float = None,  # type: ignore
    lr_scheduler=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_encoded_tokens: int = 196,
    in_domains: List[str] = [],
    loss_on_unmasked: bool = True,
    alphas: float = 1.0,
    sample_tasks_uniformly: bool = False,
):
    model.train()
    metric_logger = logger.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for task in tasks_loss_fn:
        tasks_loss_fn[task].epoch = epoch  # type: ignore

    for step, (x, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step % print_freq == 0:
            print(f'> EPOCH {epoch}, STEP {step}', flush=True)
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration  # type: ignore
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for _, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group['lr'] = lr_schedule_values[it] * param_group['lr_scale']
                if wd_schedule_values is not None and param_group['weight_decay'] > 0:
                    param_group['weight_decay'] = wd_schedule_values[it]

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        with autocast('cuda'):
            preds, masks = model(
                input_dict,
                num_encoded_tokens=num_encoded_tokens,
                alphas=alphas,
                sample_tasks_uniformly=sample_tasks_uniformly,
            )

            if args.output_dir and step == 0:
                # Save images for debugging on first step
                print('Saving debug images')
                save_debug_images(preds, masks, tasks_dict, epoch, tasks_dict)

            task_losses = {}
            for task in preds:
                target = tasks_dict[task]

                if loss_on_unmasked:
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target)
                else:
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target, mask=masks.get(task, None))

            loss = sum(task_losses.values())

        loss_value = sum(task_losses.values()).item()  # type: ignore
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order  # type: ignore
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            skip_grad=max_skip_norm,
            parameters=model.parameters(),
            create_graph=is_second_order
        )
        loss_scale_value = loss_scaler.state_dict()['scale']

        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group['lr'])
            max_lr = max(max_lr, group['lr'])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group['weight_decay'] > 0:
                weight_decay_value = group['weight_decay']
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)  # type: ignore
    # gather the stats from all processes
    print('Averaged stats:', metric_logger)
    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = get_args()
    main(args)
