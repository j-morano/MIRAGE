from argparse import HelpFormatter
from operator import attrgetter
import random

import torch
import numpy as np
from torch.backends import cudnn



class SortingHelpFormatter(HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)


def fix_seeds(seed):
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


def save_model(args, epoch, model, optimizer, loss_scaler=None):
    torch.save(
        {
            'model': model,
            'optimizer': optimizer,
            'epoch': epoch,
            'args': args,
            'loss_scaler': loss_scaler,
        },
        f'{args.output_dir}/checkpoint-best-model.pth',
    )


def load_model(args, model, optimizer, loss_scaler=None):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print('Resume checkpoint %s' % args.resume)
        if (
            'optimizer' in checkpoint
            and 'epoch' in checkpoint
            and not (hasattr(args, 'eval') and args.eval)
            and optimizer is not None
        ):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint and loss_scaler is not None:
                loss_scaler.load_state_dict(checkpoint['loss_scaler'])
            print('With optim & sched!')
