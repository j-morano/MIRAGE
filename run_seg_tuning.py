import argparse
import datetime
import socket
import json
import os
from os.path import join
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Iterable, Optional, Dict
import random

import numpy as np
from skimage import io
import torch
from torch import Tensor
from torch.amp.autocast_mode import autocast
from torchvision.utils import save_image
import yaml

from mirage.model import model_factory
from mirage.output_adapters import (
    ConvNeXtAdapter,
    DPTOutputAdapter,
    LinearSegAdapter,
    SegmenterMaskTransformerAdapter
)
import mutils.checkpoint
import mutils.logger
from mutils import native_scaler
from mutils.native_scaler import NativeScalerWithGradNormCount as NativeScaler
from mutils.datasets_semseg import build_semseg_dataset, simple_transform
from mutils.optim_factory import LayerDecayValueAssigner, create_optimizer
from mutils.semseg_metrics import mean_iou
from mutils.gdice import CEGDiceLoss
from mutils.misc import fix_seeds, SortingHelpFormatter
from fm_seg_config import fm_factory



def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument(
        '-c', '--config', default=None, type=str, metavar='FILE',
        help='YAML config file specifying default arguments. (default: %(default)s)'
    )

    parser = argparse.ArgumentParser(
        'MIRAGE semantic segmentation fine-tuning script',
        add_help=True,
        formatter_class=SortingHelpFormatter
    )
    parser.add_argument(
        '--batch_size', default=8, type=int,
        help='Batch size per GPU (default: %(default)s)'
    )
    parser.add_argument(
        '--epochs', default=200, type=int,
        help='Number of epochs to train (default: %(default)s)'
    )
    parser.add_argument(
        '--save_ckpt_freq', default=20, type=int,
        help='Frequency of saving checkpoints, in epochs (default: %(default)s)'
    )

    # Task parameters
    parser.add_argument(
        '--in_domains', default='bscan', type=str,
        help='Input domain names, separated by hyphen (default: %(default)s)'
    )
    parser.add_argument(
        '--mapping_fn', default=None, type=str,
        help='Mapping file for the dataset. If not provided (None), INFO.json'
            ' in the dataset path is used. (default: %(default)s)'
    )

    # Model parameters
    parser.add_argument(
        '--model', default='multivit_base', type=str, metavar='MODEL',
        help='Name of model to train. (default: %(default)s)',
    )
    parser.add_argument(
        '--num_global_tokens', default=1, type=int,
        help='Number of global tokens to add to encoder. (default: %(default)s)',
    )
    parser.add_argument(
        '--patch_size', default=32, type=int,
        help='Base patch size for image-like modalities. (default: %(default)s)',
    )
    parser.add_argument(
        '--input_size', default=1024, type=int,
        help='Images input size for backbone. (default: %(default)s)',
    )
    parser.add_argument(
        '--drop_path_encoder', type=float, default=0.1, metavar='PCT',
        help='Drop path rate (default: %(default)s)',
    )
    parser.add_argument(
        '--learnable_pos_emb', action='store_true',
        help='Makes the positional embedding learnable. (default: %(default)s)'
    )
    parser.add_argument('--no_learnable_pos_emb', action='store_false', dest='learnable_pos_emb')
    parser.set_defaults(learnable_pos_emb=False)
    parser.add_argument(
        '--freeze_encoder', action='store_true',
        help='Freeze the encoder weights. (default: %(default)s)'
    )
    parser.add_argument('--no_freeze_encoder', action='store_false', dest='freeze_encoder')
    parser.set_defaults(freeze_encoder=True)
    parser.add_argument(
        '--ignore_index',
        type=int,
        default=None,
        help='Index to ignore in the loss (if loss==CE-ignore-bg) and metrics.'
            ' -1: no ignore, None: auto detect. (default: %(default)s)'
    )
    parser.add_argument(
        '--output_adapter', type=str, default='convnext',
        choices=['segmenter', 'convnext', 'dpt', 'linear'],
        help='Output adapter: segmentation head (decoder) type. (default: %(default)s)'
    )
    parser.add_argument(
        '--decoder_interpolate_mode', type=str, default='bilinear',
        choices=['bilinear', 'nearest'],
        help='Interpolation mode for the decoder. (default: %(default)s)'
    )
    parser.add_argument(
        '--decoder_main_tasks', type=str, default='bscan',
        help='Main tasks for the decoder (separated with a hypen).'
            ' (default: %(default)s)'
    )
    parser.add_argument(
        '--loss', default='CE-ignore-bg', type=str,
        help='Loss function. (default: %(default)s)',
        choices=['CE', 'CE-ignore-bg', 'CEGDice']
    )
    parser.add_argument(
        '--minmax', action='store_true',
        help='Whether to use minmax normalization. (default: %(default)s)'
    )

    # Optimizer parameters
    parser.add_argument(
        '--opt', default='adamw', type=str, metavar='OPTIMIZER',
        help='Optimizer. (default: %(default)s)'
    )
    parser.add_argument(
        '--opt_eps', default=1e-8, type=float, metavar='EPSILON',
        help='Optimizer Epsilon. (default: %(default)s)'
    )
    parser.add_argument(
        '--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
        help='Optimizer Betas. (default: %(default)s)'
    )
    parser.add_argument(
        '--clip_grad', type=float, default=None, metavar='NORM',
        help='Clip gradient norm. None = no clipping. (default: %(default)s)'
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
        help='Final value of the weight decay. We use a cosine schedule for WD.'
            ' (Set the same value with args.weight_decay to keep weight decay no change)'
            ' (default: %(default)s)'
    )
    parser.add_argument(
        '--decoder_decay', type=float, default=None,
        help='Decoder weight decay multiplier. (default: %(default)s)'
    )
    parser.add_argument(
        '--no_lr_scale_list', type=str, default='',
        help='Weights that should not be affected by layer decay rate,'
        ' separated by hyphen. (default: %(default)s)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4, metavar='LR',
        help='Learning rate. (default: %(default)s)'
    )
    parser.add_argument(
        '--warmup_lr', type=float, default=1e-6, metavar='LR',
        help='Warmup learning rate. (default: %(default)s)'
    )
    parser.add_argument(
        '--min_lr', type=float, default=0.0, metavar='LR',
        help='Lower LR bound for cyclic schedulers that hit 0.'
            ' (default: %(default)s)'
    )
    parser.add_argument(
        '--layer_decay', type=float, default=0.75,
        help='Layer-wise lr decay from ELECTRA. (default: %(default)s)'
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=1, metavar='N',
        help='Epochs to warmup LR, if scheduler supports it. (default: %(default)s)'
    )
    parser.add_argument(
        '--warmup_steps', type=int, default=-1, metavar='N',
        help='Steps to warmup LR, if scheduler supports it. (default: %(default)s)'
    )

    # Dataset parameters
    parser.add_argument(
        '--test_data_path', default=None, type=str,
        help='Path to test dataset. If not provided, loads from `data_path/test`.'
        ' (default: %(default)s)'
    )
    parser.add_argument(
        '--eval_freq', default=1, type=int,
        help="Validation frequency (in epochs). (default: %(default)s)"
    )

    # Runtime parameters
    parser.add_argument(
        '--base_output_dir', default='./__output/seg', type=str,
        help='Base output directory. (default: %(default)s)'
    )
    parser.add_argument(
        '--device', default='cuda',
        help='Device to use for training / testing. (default: %(default)s)'
    )
    parser.add_argument(
        '--seed', default=42, type=int,
        help='Seed. (default: %(default)s)'
    )
    parser.add_argument(
        '--resume', default='',
        help='Resume from checkpoint. (default: %(default)s)'
    )
    parser.add_argument(
        '--auto_resume', action='store_true',
        help='Automatically resume from the latest checkpoint. (default: %(default)s)'
    )
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument(
        '--save_ckpt', action='store_true',
        help='Whether to save model checkpoints. (default: %(default)s)'
    )
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)
    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N',
        help='Start epoch. (default: %(default)s)'
    )
    parser.add_argument(
        '--infer_only', action='store_true',
        help='Perform inference only. (default: %(default)s)'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Perform testing only. (default: %(default)s)'
    )
    parser.add_argument(
        '--num_workers', default=8, type=int,
        help='Number of workers for DataLoader. (default: %(default)s)'
    )
    parser.add_argument(
        '--pin_mem', action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes)'
        ' transfer to GPU. (default: %(default)s)'
    )
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument(
        '--fp16', action='store_true',
        help='Use mixed precision training. (default: %(default)s)'
    )
    parser.add_argument('--no_fp16', action='store_false', dest='fp16')
    parser.set_defaults(fp16=True)

    # Logging parameters
    parser.add_argument(
        '--log_images_freq', default=5, type=int,
        help="Frequency of image logging (in epochs). (default: %(default)s)"
    )
    parser.add_argument(
        '--log_images', action='store_true',
        help='Whether to log images. (default: %(default)s)'
    )
    parser.add_argument(
        '--show_user_warnings', default=False, action='store_true',
        help='Whether to show user warnings. (default: %(default)s)'
    )
    parser.add_argument(
        '-v', '--version', default='v1', type=str,
        help='Version of the experiment. (default: %(default)s)'
    )

    required_parser = parser.add_argument_group('required arguments')
    required_parser.add_argument(
        '--weights', required=True, type=str,
        help='Finetune from checkpoint (model weights). (required)'
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

    return process_args(args)


def process_args(args):
    args.in_domains = args.in_domains.split('-')
    domains = args.in_domains
    if isinstance(args.patch_size, int):
        args.patch_size = {d: (args.patch_size, args.patch_size) for d in domains}

    if isinstance(args.input_size, int):
        args.input_size = {d: (args.input_size, args.input_size) for d in domains}

    args.grid_sizes = {}
    for domain, size in args.input_size.items():
        args.grid_sizes[domain] = []
        for i, s in enumerate(size):
            args.grid_sizes[domain].append(s // args.patch_size[domain][i])

    args.data_path = Path(args.data_path)
    args.dataset_name = args.data_path.stem
    args.train_data_path = args.data_path / 'train'
    args.eval_data_path = args.data_path / 'val'
    if args.infer_only and args.test and args.test_data_path is None:
        args.test_data_path = args.data_path / 'test'
    if args.mapping_fn is None:
        args.mapping_fn = args.data_path / 'INFO.json'
        with open(args.mapping_fn, 'r') as f:
            original_mapping = json.load(f)
        mapping = {}
        for k, v in original_mapping.items():
            if args.ignore_index is None:
                for bg_name in ['background', 'bg']:
                    if bg_name in v['label'].lower():
                        args.ignore_index = int(k)
                        break
            mapping[v['value']] = int(k)
        args.mapping = mapping
    args.inverse_mapping = {v: k for k, v in args.mapping.items()}
    print('Mapping:')
    print(json.dumps(args.mapping, indent=2))
    if args.ignore_index is not None:
        print('-> Ignoring index', args.ignore_index)
    args.num_classes = len(args.mapping)

    args.output_dir = str(
        Path(args.base_output_dir)
        / args.version
        / args.dataset_name
    ) + '/'
    args.output_dir += Path(args.weights).stem
    if args.freeze_encoder:
        args.output_dir += '_frozen'
    args.output_dir += f'_{args.output_adapter}'
    args.output_dir += f'_{args.loss}'
    if args.minmax:
        args.output_dir += '_minmax'
    print(f">> Output dir: {args.output_dir}")

    # NOTE: Fixed out domain: segmentation
    args.out_domains = ['semseg']
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))

    return args


def main(args):
    device = torch.device(args.device)

    fix_seeds(args.seed)

    model_config = None
    for kw in fm_factory.keys():
        if kw in args.weights.lower():
            model_config = fm_factory[kw]()
            break
    if model_config is None:
        raise ValueError(f"Unknown model: {args.weights}")

    # Forced minmax normalization
    if args.minmax:
        # args.norm = 'minmax'
        model_config.norm = 'minmax'

    model_config.build_domain_conf()

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)


    # Dataset stuff
    additional_targets = {domain: model_config.domain_conf[domain]['aug_type'] for domain in args.all_domains}

    train_transform = simple_transform(
        train=True,
        additional_targets=additional_targets,
        input_size=args.input_size[args.in_domains[0]][0],
        norm=model_config.norm
    )
    val_transform = simple_transform(
        train=False,
        additional_targets=additional_targets,
        input_size=args.input_size[args.in_domains[0]][0],
        norm=model_config.norm
    )

    dataset_train = build_semseg_dataset(args, data_path=args.train_data_path, transform=train_transform)
    print(f"Training on {len(dataset_train)} images")
    dataset_val = build_semseg_dataset(args, data_path=args.eval_data_path, transform=val_transform)
    print(f"Validating on {len(dataset_val)} images")
    args.external = None
    if args.test:
        if 'Duke_iAMD' in str(args.test_data_path):
            args.external = 'Duke_iAMD'
    if args.test_data_path is not None:
        dataset_test = build_semseg_dataset(args, data_path=args.test_data_path, transform=val_transform)
        print(f"Testing on {len(dataset_test)} images")
    else:
        dataset_test = None

    if args.external is not None:
        images_dir = Path(args.output_dir, f'preds_{args.external}')
    else:
        images_dir = Path(args.output_dir, 'preds')

    if (
        args.infer_only
        and args.test
        and dataset_test is not None
        and images_dir.exists()
        and images_dir.is_dir()
        and len(list(images_dir.iterdir())) == len(dataset_test)
    ):
        print('Inference already done. Skipping...')
        exit(0)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=2,
    )
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_test = None

    # Model
    input_adapters = {
        domain: model_config.domain_conf[domain]['input_adapter'](
            stride_level=model_config.domain_conf[domain]['stride_level'],
            patch_size_full=args.patch_size[domain],
            image_size=args.input_size[domain],
            learnable_pos_emb=args.learnable_pos_emb,
        )
        for domain in args.in_domains
    }

    # DPT settings are fixed for ViT-B. Modify them if using a different backbone.
    if '_base' not in model_config.model and args.output_adapter == 'dpt':
        raise NotImplementedError('Unsupported backbone: DPT head is fixed for ViT-B.')

    adapter_factory = {
        'segmenter': partial(
            SegmenterMaskTransformerAdapter,
            main_tasks=args.decoder_main_tasks.split('-'),
            embed_dim=768,
        ),
        'convnext': partial(
            ConvNeXtAdapter,
            preds_per_patch=16,
            depth=4,
            interpolate_mode=args.decoder_interpolate_mode,
            main_tasks=args.decoder_main_tasks.split('-'),
            embed_dim=6144,
        ),
        'dpt': partial(
            DPTOutputAdapter,
            stride_level=1,
            main_tasks=args.decoder_main_tasks.split('-'),
            head_type='semseg',
            embed_dim=256,
        ),
        'linear': partial(
            LinearSegAdapter,
            interpolate_mode=args.decoder_interpolate_mode,
            main_tasks=args.decoder_main_tasks.split('-')
        ),
    }

    print(f"> Using '{args.output_adapter}' output adapter")

    output_adapters = {
        'semseg': adapter_factory[args.output_adapter](
            num_classes=args.num_classes,
            patch_size=args.patch_size[args.in_domains[0]],
            task='semseg',
            image_size=args.input_size[args.in_domains[0]],
        ),
    }

    model = model_factory[model_config.model](
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        drop_path_rate=args.drop_path_encoder,
        args=args,
    )

    if args.weights:
        print('>> Loading weights from', args.weights)
        if args.weights.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.weights, map_location='cpu')
        else:
            checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)

        model = model_config(model, checkpoint)


    model.to(device)

    # print("Model =", model)

    total_batch_size = args.batch_size
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    print(f"LR = {args.lr:.8f}")
    print(f"Batch size =", total_batch_size)
    print(f"Number of training steps =", num_training_steps_per_epoch)
    print("Number of training examples per epoch =", (total_batch_size * num_training_steps_per_epoch))

    num_layers = model.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values =", str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    optimizer = create_optimizer(
        args,
        model,
        skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None
    )
    loss_scaler = NativeScaler(enabled=args.fp16)

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = native_scaler.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = native_scaler.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        num_training_steps_per_epoch
    )
    print(f"Max WD = {max(wd_schedule_values):.7f}, Min WD = {min(wd_schedule_values):.7f}")

    if args.loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'CE-ignore-bg':
        if args.ignore_index is None:
            raise ValueError("Ignore index is not set")
        criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    elif args.loss == 'CEGDice':
        criterion = CEGDiceLoss()
    else:
        raise ValueError(f"Invalid loss: {args.loss}")
    print("criterion = ", criterion)

    # Specifies if transformer encoder should only return last layer or all layers for DPT
    return_all_layers = args.output_adapter in ['dpt']

    lookup_table = get_lookup_table(args.inverse_mapping, device=device)

    mutils.checkpoint.auto_load_model(
        args=args,
        model=model,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        best=args.test,
    )

    if args.test:
        assert dataset_test is not None
        test_stats = evaluate(
            model=model,
            criterion=criterion,
            data_loader=data_loader_test,
            device=device,
            epoch=-1,
            in_domains=args.in_domains,
            num_classes=args.num_classes,
            dataset_name=args.dataset_name,
            mode='test',
            fp16=args.fp16,
            return_all_layers=return_all_layers,
            log_images=True,
            output_dir=args.output_dir,
            infer_only=args.infer_only,
            external=args.external,
            lookup_table=lookup_table
        )
        print(f"Performance of the network on the {len(dataset_test)} test images")
        miou = test_stats['mean_iou']
        a_acc = test_stats['pixel_accuracy']
        acc = test_stats['mean_accuracy']
        loss = test_stats['loss']
        print(f'* mIoU {miou:.3f} aAcc {a_acc:.3f} Acc {acc:.3f} Loss {loss:.3f}')
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_miou = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            in_domains=args.in_domains,
            fp16=args.fp16,
            return_all_layers=return_all_layers,
            output_dir=args.output_dir,
            lookup_table=lookup_table,
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                mutils.checkpoint.save_model(
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch
                )

        if data_loader_val is not None and (epoch % args.eval_freq == 0 or epoch == args.epochs - 1):
            val_stats = evaluate(
                model=model,
                criterion=criterion,
                data_loader=data_loader_val,
                device=device,
                epoch=epoch,
                in_domains=args.in_domains,
                num_classes=args.num_classes,
                log_images=args.log_images,
                dataset_name=args.dataset_name,
                mode='val',
                fp16=args.fp16,
                return_all_layers=return_all_layers,
                output_dir=args.output_dir,
                lookup_table=lookup_table,
            )
            if max_miou < val_stats["mean_iou"]:
                max_miou = val_stats["mean_iou"]
                if args.output_dir and args.save_ckpt:
                    mutils.checkpoint.save_model(
                        args=args,
                        model=model,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best"
                    )
            print(f'Max mIoU: {max_miou:.3f}')

            log_stats = {
                **{f'train/{k}': v for k, v in train_stats.items()},
                **{f'val/{k}': v for k, v in val_stats.items()},
                'epoch': epoch,
            }
        else:
            log_stats = {
                **{f'train/{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
            }

        if log_writer is not None:
            log_writer.update(log_stats)

        if args.output_dir:
            with open(join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Test with best checkpoint
    if data_loader_test is not None:
        print('Loading model with best validation mIoU')
        checkpoint = torch.load(join(args.output_dir, 'checkpoint-best.pth'), map_location='cpu', weights_only=False)
        state_dict = {}
        for k,v in checkpoint['model'].items():
            state_dict[f'module.{k}'] = v
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

        print('Testing with best checkpoint')
        test_stats = evaluate(
            model=model,
            criterion=criterion,
            data_loader=data_loader_test,
            device=device,
            epoch=checkpoint['epoch'],
            in_domains=args.in_domains,
            num_classes=args.num_classes,
            log_images=True,
            dataset_name=args.dataset_name,
            mode='test',
            fp16=args.fp16,
            return_all_layers=return_all_layers,
            lookup_table=lookup_table,
        )
        log_stats = {f'test/{k}': v for k, v in test_stats.items()}
        if log_writer is not None:
            log_writer.set_step(args.epochs * num_training_steps_per_epoch)
            log_writer.update(log_stats)
        if args.output_dir:
            with open(join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


def to_range(tensor, target_min, target_max):
    """Rescale the values of a tensor to a specified range.

    Args:
        tensor: Input torch tensor
        target_min: Minimum value of the target range
        target_max: Maximum value of the target range

    Returns:
    Tensor with values rescaled to the target range
    """
    # Get the minimum and maximum of the input tensor
    input_min = torch.min(tensor)
    input_max = torch.max(tensor)

    # Avoid division by zero if input_min == input_max
    if input_min == input_max:
        return torch.full_like(tensor, target_min)

    ## Rescale the tensor to the target range
    # Scale to [0, 1]
    rescaled_tensor = (tensor - input_min) / (input_max - input_min)
    # Scale to [target_min, target_max]
    rescaled_tensor = rescaled_tensor * (target_max - target_min) + target_min

    return rescaled_tensor


def get_lookup_table(mapping: Dict[int, int], device: torch.device) -> torch.Tensor:
    # Create a lookup table
    max_key = max(mapping.keys())  # Get the largest key in the dictionary
    lookup_table = torch.full((max_key + 1,), -1)  # Initialize with default values
    for key, value in mapping.items():
        lookup_table[key] = value  # Fill the lookup table
    lookup_table = lookup_table.to(device)
    return lookup_table


def save_predictions(
    input_image: Tensor,
    seg_pred: Tensor,
    seg_gt: Tensor,
    output_dir: Optional[str],
    epoch: int,
    suffix: str,
    lookup_table: torch.Tensor,
):
    input_image = input_image.clone().detach()
    seg_pred = seg_pred.clone().detach()
    seg_gt = seg_gt.clone().detach()
    print('input_image', input_image.shape, input_image.min().item(), input_image.max().item())
    print('pred semseg', seg_pred.shape, seg_pred.min().item(), seg_pred.max().item())
    print('gt semseg', seg_gt.shape, seg_gt.min().item(), seg_gt.max().item())
    assert output_dir is not None
    os.makedirs(join(output_dir, 'debug'), exist_ok=True)
    seg_gt_to_save = seg_gt.float().unsqueeze(1)
    seg_gt_to_save = lookup_table[seg_gt_to_save.long()]
    print('seg_gt_to_save', seg_gt_to_save.shape, torch.unique(seg_gt_to_save))
    seg_pred_to_save = seg_pred.argmax(dim=1).float().unsqueeze(1)
    seg_pred_to_save = lookup_table[seg_pred_to_save.long()]
    # print('seg_pred_to_save', seg_pred_to_save.shape, torch.unique(seg_pred_to_save))
    # to 0-1
    if input_image.shape[1] > 1:
        seg_gt_to_save = torch.cat([seg_gt_to_save,] * 3, dim=1)
        seg_pred_to_save = torch.cat([seg_pred_to_save,] * 3, dim=1)
    input_image = to_range(input_image, 0, 255)
    to_save = torch.cat([input_image, seg_gt_to_save, seg_pred_to_save], dim=2)
    save_image(to_save, join(output_dir, 'debug', f'{epoch}{suffix}.png'), normalize=True)


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    log_writer=None,
    start_steps=0,
    lr_schedule_values=None,
    wd_schedule_values=None,
    in_domains=None,
    fp16=True,
    return_all_layers=False,
    output_dir=None,
    lookup_table=None
):
    model.train()
    metric_logger = mutils.logger.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', mutils.logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', mutils.logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    assert in_domains is not None

    for step, (x, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for _i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        # Forward + backward
        with autocast('cuda', enabled=fp16):
            preds = model(input_dict, return_all_layers=return_all_layers)
            seg_pred, seg_gt = preds['semseg'], tasks_dict['semseg']
            if step == 0:
                assert lookup_table is not None
                save_predictions(
                    input_dict[in_domains[0]], seg_pred, seg_gt, output_dir, epoch,
                    '_train', lookup_table)
            # print(seg_pred.shape, seg_gt.shape)
            loss = criterion(seg_pred, seg_gt)

        loss_value = loss.item()

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order  # type: ignore
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order
        )
        if fp16:
            loss_scale_value = loss_scaler.state_dict()["scale"]
        else:
            loss_scale_value = None

        # Metrics and logging
        metric_logger.update(loss=loss_value)
        if fp16:
            metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(
                {
                    'loss': loss_value,
                    'lr': max_lr,
                    'weight_decay': weight_decay_value,
                    'grad_norm': grad_norm,
                }
            )
            log_writer.set_step()

    print("Averaged stats:", metric_logger)
    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model,
    criterion,
    data_loader,
    device,
    epoch,
    in_domains,
    num_classes,
    dataset_name,
    log_images=False,
    mode='val',
    fp16=True,
    return_all_layers=False,
    output_dir=None,
    infer_only=False,
    external=None,
    lookup_table=None,
    ignore_index=None,
):
    # Switch to evaluation mode
    model.eval()

    metric_logger = mutils.logger.MetricLogger(delimiter="  ")
    if mode == 'val':
        header = f'{dataset_name}: (Eval) Epoch: [{epoch}]'
    elif mode == 'test':
        header = f'{dataset_name}: (Test) Epoch: [{epoch}]'
    else:
        raise ValueError(f'Invalid eval mode {mode}')
    print_freq = 20

    seg_preds = []
    seg_gts = []
    input_images = []
    sids = []

    step_to_save = random.randint(0, len(data_loader))

    if log_images or infer_only:
        assert output_dir is not None
        if external is not None:
            save_dir = Path(output_dir, f'preds_{external}')
        else:
            save_dir = Path(output_dir, 'preds')
        save_dir.mkdir(exist_ok=True)
    else:
        save_dir = None

    for step, (x, _, sid) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # print(sid)
        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        # Forward + backward
        loss = None
        with autocast('cuda', enabled=fp16):
            preds = model(input_dict, return_all_layers=return_all_layers)
            input_image = input_dict[in_domains[0]]
            seg_pred, seg_gt = preds['semseg'], tasks_dict['semseg']
            if not infer_only:
                assert lookup_table is not None
                if step == step_to_save:
                    save_predictions(
                        input_dict[in_domains[0]], seg_pred, seg_gt, output_dir, epoch,
                        '_val', lookup_table)
                loss = criterion(seg_pred, seg_gt)

        # If there is void, exclude it from the preds and take second highest class
        seg_pred_argmax = seg_pred.argmax(dim=1)
        if infer_only or log_images:
            assert save_dir is not None
            assert lookup_table is not None
            for i in range(len(sid)):
                pred_i = lookup_table[seg_pred_argmax[i].long()].cpu().numpy().astype(np.uint8)
                io.imsave(save_dir / f'{sid[i]}.png', pred_i)
        else:
            assert loss is not None
            loss_value = loss.item()
            seg_preds.extend(list(seg_pred_argmax.cpu().numpy()))
            seg_gts.extend(list(seg_gt.cpu().numpy()))
            input_images.extend(list(input_image.cpu().numpy()))
            sids.extend(list(sid))
            metric_logger.update(loss=loss_value)

    if infer_only:
        print('Inference done. Exiting...')
        exit(0)

    scores = compute_metrics(
        seg_preds,
        seg_gts,
        num_classes=num_classes,
        device=device,
        ignore_index=ignore_index,
    )

    for k, v in scores.items():
        metric_logger.update(**{f"{k}": v})

    print(
        f'* mIoU {metric_logger.mean_iou.global_avg:.3f}'
        f' aAcc {metric_logger.pixel_accuracy.global_avg:.3f}'
        f' Acc {metric_logger.mean_accuracy.global_avg:.3f}'
        f' Loss {metric_logger.loss.global_avg:.3f}'
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_metrics(
    seg_preds,
    seg_gts,
    num_classes,
    device,
    ignore_index=None,
):
    seg_preds = torch.tensor(seg_preds, device=device)
    seg_gts = torch.tensor(seg_gts, device=device)

    ret_metrics = mean_iou(
        results=[seg_preds],
        gt_seg_maps=[seg_gts],
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    ret_metrics_mean = torch.tensor(
        [
            np.round(np.nanmean(ret_metric.astype(float)) * 100, 2)
            for ret_metric in ret_metrics
        ],
        dtype=float,  # type: ignore
        device=device,
    )
    pix_acc, mean_acc, miou = ret_metrics_mean
    ret = dict(pixel_accuracy=pix_acc, mean_accuracy=mean_acc, mean_iou=miou)
    return ret


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        if opts.minmax:
            opts.output_dir += '_minmax'
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    print('>>>', opts.output_dir)
    if (
        not opts.test
        and Path(opts.output_dir, 'checkpoint-best.pth').exists()
        and Path(opts.output_dir, 'checkpoint-199.pth').exists()
    ):
        print('Model already trained. Skipping...')
        exit(0)
    if opts.test:
        assert (
            Path(opts.output_dir, 'checkpoint-best.pth').exists()
            # and Path(opts.output_dir, 'checkpoint-199.pth').exists()
        ), 'ERROR: Model not fully trained'

    main(opts)
