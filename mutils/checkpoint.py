import os
from pathlib import Path
import glob

import torch



def save_model(args, epoch, model, optimizer, loss_scaler, loss_balancer=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)
        to_save = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args
        }

        if loss_balancer is not None:
            to_save['loss_balancer'] = loss_balancer.state_dict()

        torch.save(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag="checkpoint-%s" % epoch_name,
            client_state=client_state
        )


def auto_load_model(
    args,
    model,
    optimizer,
    loss_scaler,
    best=False,
):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        if args.auto_resume and len(args.resume) == 0:
            if best:
                args.resume = os.path.join(output_dir, 'checkpoint-best.pth')
                assert os.path.exists(args.resume), f"Best checkpoint not found at {args.resume}"
            else:
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu')
            else:
                checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'])
            if not best:
                if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    args.start_epoch = checkpoint['epoch'] + 1
                    if 'scaler' in checkpoint:
                        loss_scaler.load_state_dict(checkpoint['scaler'])
                    print("With optim & sched!")
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
