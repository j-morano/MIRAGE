from copy import deepcopy
import json
import sys
import re
import hashlib

import argparse
import datetime
import pandas as pd
import time
from pathlib import Path
import socket

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn

from timm.loss import LabelSmoothingCrossEntropy

from mutils.vit import vit_large_patch16, vit_base_patch16
from main import MIRAGEWrapper
from mutils.classification import build_dataset
# from src.retfound.pos_embed import interpolate_pos_embed
import mutils.lr_utils as lru
from mutils import misc
from mutils.classification import train_1_epoch, evaluate, EarlyStopping
from mutils.factory import get_factory_adder
from mutils.misc import fix_seeds



def get_args_parser():
    parser = argparse.ArgumentParser(
        'Retinal image classification experiments', add_help=False
    )
    parser.add_argument(
        '--input_size',
        default=224,
        type=int,
        help='Images input size: 224 for Random, ImageNet, RetFound and 512 for MIRAGE.',
    )
    parser.add_argument(
        '--drop_path',
        type=float,
        default=0.1,
        metavar='PCT',
        help='Drop path rate (default: 0.1)',
    )

    parser.add_argument(
        '--weight_decay', type=float, default=0.05, help='Weight decay (default: 0.05)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5,
        metavar='LR',
        help='learning rate (absolute lr)',
    )
    parser.add_argument(
        '--layer_decay',
        type=float,
        default=0.75,
        help='layer-wise lr decay from ELECTRA/BEiT',
    )
    parser.add_argument(
        '--min_lr', type=float, default=1e-8, metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0',
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR'
    )
    parser.add_argument(
        '--smoothing', type=float, default=0.1,
        help='Label smoothing (default: 0.1). If 0 -> CrossEntropyLoss.',
    )
    parser.add_argument(
        '--accum_iter', default=1, type=int,
        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints).',
    )

    # * Supervised training params
    parser.add_argument(
        '--linear_probing', action='store_true',
        help='Set to True for not training the encoder weights.',
    )
    parser.add_argument(
        '--label_efficiency_exp', default=False, type=bool,
        help='Set to True for label efficiency experiments.',
    )
    parser.add_argument(
        '--model',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--init_weights',
        default=None,
        type=str,
        help='Pre-trained weights to initialise the model with. Default: RetFound.',
    )
    parser.add_argument(
        '--resume', default='', help='Resume from checkpoint - for supervised training.'
    )
    parser.add_argument(
        '--pool',
        default='global',
        type=str,
        help='By default we use global pooling instead of the CLS token for classifier input.',
        choices=['global', 'cls', 'token_mix'],
    )
    parser.add_argument(
        '--base_output_dir',
        default='./__output_cls',
        help='path where to save, empty for no saving',
    )

    # * Dataset parameters
    parser.add_argument(
        '--data_root',
        default='/msc/home/jmoran82/FullVIBES-v2/Downstream_datasets/',
        type=str,
        help='dataset path',
    )
    parser.add_argument('--data_set', default='', type=str, help='dataset folder name')
    parser.add_argument(
        '--nb_classes',
        default=None,
        type=int,
        help='Number of the classification types. None for automatic counting.'
    )
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.',
    )
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # * Training parameters
    parser.add_argument(
        '--device', default='cuda', help='device to use for training / testing'
    )
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N', help='start epoch'
    )
    parser.add_argument(
        '--batch_size', default=None, type=int,
        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus). None for automatic calculation.',
    )
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument(
        '--eval', action='store_true',
        help='Set to True for only running the evaluation on the test set.',
    )
    parser.add_argument(
        '--early_stopping_epochs', default=200, type=int,
        help='Parameter to control how many epochs to wait for the validation loss to improve before stopping..',
    )
    parser.add_argument(
        '--early_stopping_delta', default=0.01, type=float,
        help='Parameter to specify the minimum change in the validation loss required to consider it an improvement.',
    )
    parser.add_argument(
        '--early_stopping_delta_two', default=0.01, type=float,
        help='Parameter to specify the minimum change in the validation loss'
            ' required to consider it an improvement.',
    )
    parser.add_argument(
        '--early_start_from',
        default=0,
        type=int,
        help='Parameter to specify the epoch to start early stopping.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
    )
    parser.add_argument(
        '--full_finetune',
        action='store_true',
    )
    # parser.add_argument(
    #     '--rot',
    #     action='store_true',
    # )
    parser.add_argument(
        '--all-tokens',
        action='store_true',
    )
    parser.add_argument(
        '--input-modality',
        default='bscan',
    )
    # parser.add_argument(
    #     '--scaler',
    #     default='z-score',
    #     type=str,
    #     help='Scaler to use for the input images. Options: z-score, min-max, none.',
    # )
    parser.add_argument(
        '--version',
        required=True,
    )
    parser.add_argument(
        '--get-embeddings',
        type=str,
        default=None,
        choices=['test', 'all'],
    )
    parser.add_argument(
        '--dont-override-args',
        action='store_true',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
    )
    parser.add_argument(
        '--val_metric',
        default='loss',
        type=str,
    )
    parser.add_argument(
        '--val_metric_two',
        default='bacc',
        type=str,
    )
    parser.add_argument(
        '--save_predictions',
        action='store_true',
    )
    parser.add_argument(
        '--fill',
        default=None,
        type=float,
    )
    parser.add_argument(
        '--no_affine',
        action='store_true',
    )
    parser.add_argument(
        '--no_minmax',
        action='store_true',
    )

    return parser

add_config, config_factory = get_factory_adder()


class Config:
    def __init__(self, args):
        args.input_size = 224
        args.lr = 1e-5
        args.weight_decay = 1e-2
        args.full_finetune = True
        args.all_tokens = False
        self.args = args


@add_config('MIRAGE')
class MIRAGEConfig(Config):
    def __init__(self, args):
        super().__init__(args)
        # TODO: Simplify this
        if '_bscan_' in args.init_weights:
            args.input_modality = 'bscan'
        elif '_slo_' in args.init_weights:
            args.input_modality = 'slo'
        elif '_bscan-slo' in args.init_weights:
        # also for '_bscan-slo-bscanlayermap_'
            if 'oct_slo' in args.data_set.lower():
                args.input_modality = 'bscan-slo'
            elif 'slo' in args.data_set.lower():
                args.input_modality = 'slo'
            else:
                args.input_modality = 'bscan'
        elif '_rgb_' in args.init_weights:
            args.input_modality = 'rgb'
        elif '_rgb-depth-semseg_' in args.init_weights:
            if 'oct_slo' in args.data_set.lower():
                args.input_modality = 'rgb-depth'
            else:
                args.input_modality = 'rgb'

        if 'multivit' in args.init_weights.lower():
            # For pretrained MIRAGE models on ImageNet
            args.input_size = 224
        else:
            args.input_size = 512
        args.lr = 1e-5
        args.weight_decay = 1e-2
        args.full_finetune = True
        args.all_tokens = True
        # args.scaler = 'min-max'
        if '_bscan_' in args.init_weights:
            args.input_modality = 'bscan'
        elif '_slo_' in args.init_weights:
            args.input_modality = 'slo'
        elif '_bscan-slo' in args.init_weights:
        # also for '_bscan-slo-bscanlayermap_'
            if 'oct_slo' in args.data_set.lower():
                args.input_modality = 'bscan-slo'
                # args.scaler = 'none'
            elif 'slo' in args.data_set.lower():
                args.input_modality = 'slo'
            else:
                args.input_modality = 'bscan'
        elif '_rgb_' in args.init_weights:
            args.input_modality = 'rgb'
        elif '_rgb-depth-semseg_' in args.init_weights:
            if 'oct_slo' in args.data_set.lower():
                args.input_modality = 'rgb-depth'
            else:
                args.input_modality = 'rgb'

        if '_bscan-slo_' in args.init_weights:
            modalities = 'bscan-slo'
        elif '_bscan_' in args.init_weights:
            modalities = 'bscan'
        elif '_slo_' in args.init_weights:
            modalities = 'slo'
        elif '_bscan-slo-bscanlayermap_' in args.init_weights:
            modalities = 'bscan-slo-bscanlayermap'
        elif '_rgb_' in args.init_weights:
            modalities = 'rgb'
        elif '_rgb-depth-semseg_' in args.init_weights:
            modalities = 'rgb-depth-semseg'
        else:
            raise ValueError('Unknown modalities.')

        self.model = MIRAGEWrapper(
            input_size=args.input_size,
            patch_size=32,
            num_classes=args.nb_classes,
            all_tokens=args.all_tokens,
            modalities=modalities,
            input_modality=args.input_modality,
            # NOTE: weights are loaded in the model
            weights=args.init_weights,
        )
        if '-l' in args.init_weights:
            self.model_disp_name = 'MIRAGE-L'
        else:
            self.model_disp_name = 'MIRAGE-B'

        self.args = args


class ViT21kConfig(Config):
    def load_model(self, model_fun, args, state_dict_fn):
        model = model_fun(
            img_size=args.input_size,
            num_classes=args.nb_classes,
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


@add_config('vit_l_16_21k')
class ViTL16_21kConfig(ViT21kConfig):
    def __init__(self, args):
        super().__init__(args)
        self.model_disp_name = 'ViT-L-21k/16'
        self.model = self.load_model(
            vit_large_patch16,
            args,
            './_weights/timm--vit_large_patch16_224.orig_in21k.bin',
        )


@add_config('vit_b_16_21k')
class ViTB16_21kConfig(ViT21kConfig):
    def __init__(self, args):
        super().__init__(args)
        self.model_disp_name = 'ViT-B-21k/16'
        self.model = self.load_model(
            vit_base_patch16,
            args,
            './_weights/timm--vit_base_patch16_224.orig_in21k.bin',
        )


class DinoTokenMix(nn.Module):
    def __init__(self, model, embed_dim, nb_classes):
        super().__init__()
        self.model = model
        self.head = nn.Linear(embed_dim * 2, nb_classes)

    def forward(self, x):
        out = self.model.forward_features(x)
        out = torch.cat([
            out['x_norm_clstoken'],
            out['x_norm_patchtokens'].mean(dim=1)
        ], dim=1)
        out = self.head(out)
        return out


@add_config('dinov2_vitb14')
class DINOv2ViTB14Config(Config):
    def __init__(self, args):
        super().__init__(args)
        self.model_disp_name = 'DINOv2-ViT-B/14'
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        print('Loaded ImageNet weights for DINOv2 ViT-B/14')
        if args.pool == 'token_mix':
            print('ViT: Using token mix')
            model = DinoTokenMix(model, 768, args.nb_classes)
        else:
            model.head = nn.Linear(768, args.nb_classes)  # type: ignore
        self.model = model


@add_config('dinov2_vitl14')
class DINOv2ViTL14Config(Config):
    def __init__(self, args):
        super().__init__(args)
        self.model_disp_name = 'DINOv2-ViT-L/14'
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        if args.pool == 'token_mix':
            model = DinoTokenMix(model, 1024, args.nb_classes)
        else:
            model.head = nn.Linear(1024, args.nb_classes)  # type: ignore
        print('Loaded ImageNet weights for DINOv2 ViT-L/14')
        self.model = model


class ViTLargeConfig(Config):
    def __init__(self, args):
        args.input_size = 224
        if args.linear_probing:
            args.lr = 1e-5
        else:
            if 'octdl' in args.data_set.lower():
                args.lr = 1e-2
            else:
                args.lr = 1e-4
        args.weight_decay = 1e-2
        args.all_tokens = False
        self.args = args

    def load_model(self, args):
        model = vit_large_patch16(
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            pool=args.pool,
        )

        # load pre-trained weights if path provided
        checkpoint_model = torch.load(args.init_weights, map_location='cpu', weights_only=False)['model']

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        assert set(msg.missing_keys) == {
            'head.weight',
            'head.bias',
        }
        return model


@add_config('RETFound')
class RETFoundConfig(ViTLargeConfig):
    def __init__(self, args):
        super().__init__(args)
        self.args.init_weights = './__weights/RETFound_vit-l_bscan_weights.pth'
        self.args.full_finetune = False
        self.model = self.load_model(self.args)
        self.model_disp_name = 'RETFound'



def main(args):
    fix_seeds(args.seed)

    device = torch.device(args.device)

    hostname = socket.gethostname()
    if hostname == 'hemingway':
        print(f'Running on {hostname}')
        args.data_root = '/mnt/Data/SSHFS/msc_server/FullVIBES-v2/Downstream_datasets/'
        # args.batch_size = 2

    args.data_path = args.data_root + args.data_set

    train_data_path = args.data_path + '/train'
    if args.nb_classes is None:
        args.nb_classes = 0
        for class_dir in Path(train_data_path).iterdir():
            if class_dir.is_dir():
                args.nb_classes += 1
    num_samples = 0
    for class_dir in Path(train_data_path).iterdir():
        if class_dir.is_dir():
            num_samples += len(list(class_dir.iterdir()))
    print(f'Number of classes: {args.nb_classes}')
    print(f'Number of training samples: {num_samples}')
    # Batch size is closest power of 2 to 1/10 of the dataset
    if args.batch_size is None:
        args.batch_size = min(64, 2 ** (int(round(num_samples * 0.25)).bit_length() - 1))
        if args.batch_size < 1:
            args.batch_size = 8
    print(f'Batch size: {args.batch_size}')

    dataset_train = None
    dataset_val = None
    if not args.eval:
        if args.get_embeddings is not None:
            augment_train = False
            shuffle = False
        else:
            augment_train = True
            shuffle = True
        dataset_train = build_dataset(subset='train', args=args, augment=augment_train)
        try:
            print(dataset_train.class_to_idx)
        except AttributeError:
            pass
        train_loader = DataLoader(
            dataset_train,
            shuffle=shuffle,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f'Number of training samples: {len(dataset_train)}')

        dataset_val = build_dataset(subset='val', args=args, augment=False)
        valid_loader = DataLoader(
            dataset_val,
            shuffle=shuffle,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f'Number of validation samples: {len(dataset_val)}')
    else:
        train_loader = None
        valid_loader = None

    if 'cross_train' not in args.data_set.lower():
        dataset_test = build_dataset(subset='test', args=args, augment=False)
        if args.get_embeddings == 'all':
            # Join all data in a single dataloader
            if not args.eval and dataset_train is not None and dataset_val is not None:
                dataset_test = ConcatDataset(
                    [dataset_train, dataset_val, dataset_test]
                )
        test_loader = DataLoader(
            dataset_test,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f'Number of test samples: {len(dataset_test)}')
    else:
        test_loader = None

    # Initialize the model
    model_config = config_factory[args.model](args)
    model = model_config.model
    model_disp_name = model_config.model_disp_name
    args = model_config.args

    args.all_tokens = False

    if args.fill is not None:
        if args.fill < 0:
            args.fill = None

    # print(model.state_dict().keys())
    if args.linear_probing:
        print('> Linear probing')
        print('Freezing encoder layers for linear probing')
        args.full_finetune = False
        # NOTE: Works way better with a higher learning rate like the
        # following.
        args.lr = 1e-3
        # freeze encoder layers for linear probing
        print('Tuned parameters:')
        for name, param in model.named_parameters():
            if 'head.' not in name:  # and 'norm.' not in name:
                param.requires_grad = False
            else:
                print('\t', name)
    else:
        for param in model.parameters():
            param.requires_grad = True

    model.to(device)
    # print(model)

    # Print model info
    n_parameters = sum(p.numel() for p in model.parameters())
    n_tr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (N): %.2e' % (n_parameters))
    print('number of params (N):', n_parameters)
    print('number of trainable params (M): %.2e' % (n_tr_parameters))
    print('number of trainable params (M):', n_tr_parameters)

    # Save args in the name of the model as a checksum and in a json file
    args_vars = vars(args).copy()
    # Remove unnecessary keys
    model_config_keys = [
      'accum_iter', 'all_tokens', 'drop_path', 'early_start_from',
      'early_stopping_delta', 'early_stopping_delta_two',
      'early_stopping_epochs', 'fill', 'full_finetune', 'init_weights',
      'input_modality', 'input_size', 'label_efficiency_exp', 'layer_decay',
      'linear_probing', 'lr', 'min_lr', 'model', 'no_affine', 'no_minmax',
      'pool', 'smoothing', 'start_epoch', 'val_metric', 'val_metric_two',
      'warmup_epochs', 'weight_decay',
    ]
    for key in list(args_vars.keys()):
        if key not in model_config_keys:
            args_vars.pop(key, None)
    args_str = json.dumps(args_vars, indent=2, sort_keys=True)
    args_checksum = hashlib.md5(args_str.encode('utf-8')).hexdigest()[:8]
    print(f'Args checksum: {args_checksum}')
    args.output_dir += f'_{args_checksum}/'

    output_dir = Path(args.output_dir)

    # Create output directory
    print(f'> Saving to {args.output_dir}')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'args.json', 'w') as f:
        f.write(args_str)

    print(f'Args:\n{args_str}')

    if (
        (output_dir / 'test_eval.csv').exists()
        and (args.get_embeddings is None)
        and not args.overwrite
        and not args.save_predictions
    ):
        print('Experiment already run. Exiting.')
        sys.exit(0)

    if (
        (output_dir / 'predictions.npz').exists()
        and args.save_predictions
        and not args.overwrite
    ):
        print('Predictions already saved. Exiting.')
        sys.exit(0)

    if args.dry_run:
        print('Dry run. Exiting.')
        sys.exit(0)

    if args.get_embeddings is not None:
        assert test_loader is not None
        if args.eval and args.get_embeddings == 'test':
            # Evaluate on the best checkpoint
            args.resume = f'{args.output_dir}/checkpoint-best-model.pth'
            misc.load_model(args=args, model=model, optimizer=None)
            save_path = Path(f'{args.output_dir}/embeddings/ft_test')
        elif not args.eval and args.get_embeddings == 'test':
            save_path = Path(f'{args.output_dir}/embeddings/test')
        else:
            save_path = Path(f'{args.output_dir}/embeddings/all')
        save_path.mkdir(parents=True, exist_ok=True)
        try:
            model.forward_features
        except AttributeError:
            model.heads = nn.Sequential(nn.Identity())
        test_stats = evaluate(
            model, test_loader, 'Best', device, args.nb_classes, mode='Test',
            get_embeddings=True, save_path=save_path
        )
        exit(0)
    elif args.save_predictions:
        assert test_loader is not None
        print('Getting predictions for the best checkpoint')
        args.resume = f'{args.output_dir}/checkpoint-best-model.pth'
        misc.load_model(args=args, model=model, optimizer=None)
        save_path = args.output_dir
        test_stats = evaluate(
            model, test_loader, 'Best', device, args.nb_classes, mode='Test',
            save_predictions=True, save_path=save_path
        )
        exit(0)

    if args.full_finetune or args.linear_probing:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lru.param_groups_lrd(
            model,
            args.weight_decay,
            no_weight_decay_list=model.no_weight_decay(),
            layer_decay=args.layer_decay,
            num_layers=len(model.model.encoder) if 'MIRAGE' in args.init_weights else None,
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    if not args.eval:
        if args.smoothing > 0.0:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        greater_is_better = args.val_metric != 'loss'
        greater_is_better_two = args.val_metric_two != 'loss'

        # Initialize early stopping object
        early_stopping = EarlyStopping(
            patience=args.early_stopping_epochs,
            delta=args.early_stopping_delta,
            delta_two=args.early_stopping_delta_two,
            greater_is_better=greater_is_better,
            greater_is_better_two=greater_is_better_two,
            start_from=args.early_start_from,
        )

        start_time = time.time()
        train_stats_all, val_stats_all = [], []
        best_model = argparse.Namespace()
        assert train_loader is not None
        assert valid_loader is not None
        for epoch in range(args.start_epoch, args.epochs):
            try:
                train_stats = train_1_epoch(
                    model,
                    criterion,
                    train_loader,
                    optimizer,
                    device,
                    epoch,
                    args=args,
                )
            except ValueError as e:
                print('Early stopping')
                print(e)
                break

            train_stats_all.append(train_stats.values())

            val_stats = evaluate(
                model, valid_loader, epoch, device, args.nb_classes,
                mode='Valid', args=args
            )
            assert val_stats is not None
            val_stats_all.append(val_stats.values())

            # If the validation loss has improved, save checkpoint
            # Check if early stopping criterion is met
            is_best = early_stopping(val_stats[args.val_metric], val_stats[args.val_metric_two], epoch)
            if early_stopping.early_stop:
                print(f'Early stopping @ epoch {epoch}')
                break
            else:
                if is_best and args.output_dir:
                    # Save in memory to avoid writing to disk all the time
                    best_model= argparse.Namespace(
                        # NOTE: Pass model and optimizer state_dicts as
                        #   values (copies), not as references.
                        model=deepcopy(model.state_dict()),
                        optimizer=deepcopy(optimizer.state_dict()),
                        epoch=epoch,
                    )
                    # misc.save_model(args, epoch, model, optimizer)
                    print(
                        f'New best {model_disp_name} model'
                        f' on {args.data_set} with seed {args.seed}'
                        f' @ epoch {epoch}'
                        f'\n\t({early_stopping.best_value}, {early_stopping.best_value_two})'
                    )

        misc.save_model(args, epoch=best_model.epoch, model=best_model.model, optimizer=best_model.optimizer)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        # Save evaluation results
        pd.DataFrame(
            data=train_stats_all, columns=['Epoch', 'Loss', 'BAcc', 'F1-score']
        ).to_csv(f'{args.output_dir}/train_eval.csv', index=False)

        pd.DataFrame(
            data=val_stats_all,
            columns=['Epoch', 'Loss', 'BAcc', 'AUROC', 'AP', 'F1-score', 'MCC'],
        ).to_csv(f'{args.output_dir}/valid_eval.csv', index=False)

    if test_loader is not None:
        # Evaluate on the best checkpoint
        args.resume = f'{args.output_dir}/checkpoint-best-model.pth'
        misc.load_model(args=args, model=model, optimizer=optimizer)
        test_stats = evaluate(
            model, test_loader, 'Best', device, args.nb_classes, mode='Test'
        )
        assert test_stats is not None
        pd.DataFrame(
            data=[test_stats.values()],
            columns=['Epoch', 'Loss', 'BAcc', 'AUROC', 'AP', 'F1-score', 'MCC'],
        ).to_csv(f'{args.output_dir}/test_eval.csv', index=False)



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    args.output_dir = args.base_output_dir
    if args.output_dir[-1] != '/':
        args.output_dir += '/'
    args.output_dir += f'{args.version}/'
    args.output_dir += f'{args.seed}/'
    args.output_dir += f'{args.data_set}/'
    args.output_dir += f'{args.model}'
    if args.linear_probing:
        args.output_dir += '_linear'
    elif args.full_finetune:
        args.output_dir += '_fullfinetune'
    else:
        args.output_dir += '_finetune'
    if args.init_weights is not None:
        args.output_dir += '_iw'

    main(args)
