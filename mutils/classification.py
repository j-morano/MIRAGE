import os
from typing import Optional, Iterable, Union
from collections import OrderedDict
from pathlib import Path
import math
import random

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torchvision.utils import save_image
from torchvision import datasets
from torchvision.datasets.folder import default_loader
import torchvision.transforms as tvtr
import torchvision.transforms.functional as VF
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import mutils.lr_utils as lru



class EarlyStopping:
    def __init__(
        self,
        patience=50,
        delta=0.01,
        greater_is_better=False,
        delta_two=0.01,
        greater_is_better_two=False,
        start_from=0
    ):
        self.patience = patience
        self.delta = delta
        self.delta_two = delta_two
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.greater_is_better = greater_is_better
        if greater_is_better:
             self.is_better = lambda x, y: (x - y) > self.delta
        else:
            self.is_better = lambda x, y: (y - x) > self.delta
        if greater_is_better_two:
            self.is_better_two = lambda x, y: (x - y) > self.delta_two
        else:
            self.is_better_two = lambda x, y: (y - x) > self.delta_two
        self.is_same = lambda x, y: abs(x - y) < self.delta
        self.start_from = start_from

    def __call__(self, value, value_two, epoch):
        '''Returns True if the value is the best one so far, False
        otherwise.
        '''
        if (
            self.best_value is None
            # or (
            #     self.is_better(value, self.best_value)
            #     and self.is_better_two(value_two, self.best_value_two)
            # )
            or self.is_better(value, self.best_value)
            or ( # first value is the same, but second is better
                self.is_same(value, self.best_value)
                and self.is_better_two(value_two, self.best_value_two)
            )
        ):
            self.best_value = value
            self.best_value_two = value_two
            self.counter = 0
            return True
        elif epoch >= self.start_from:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return False


def train_1_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args=None,
):
    assert args is not None, "args must be provided"

    model.train(True)
    optimizer.zero_grad()

    losses, true_labels, predictions = [], [], []
    for i, batch in enumerate(data_loader):
        images, targets = batch[0], batch[-1]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # if i == 0:
        #     save_image(images, "images.png", normalize=True)

        # we use a per iteration (instead of per epoch) lr scheduler
        if i % args.accum_iter == 0:
            lru.adjust_learning_rate(optimizer, i / len(data_loader) + epoch, args)

        with autocast('cuda', enabled=True):
            # print(args.output_dir)
            # format: 3 numbers always
            if i == 0 and epoch % 10 == 0:
                epoch_str = str(epoch).zfill(3)
                save_fn = Path(args.output_dir, "debug", f"{epoch_str}_train.jpg")
                save_fn.parent.mkdir(exist_ok=True)
                print('Saving images for debugging')
                print('  images', images.shape, images.min().item(), images.max().item())
                print('  targets', targets.shape, targets.min().item(), targets.max().item())
                save_image(images, save_fn, normalize=True)
            # Make predictions for this batch
            outputs = model(images)

            # Compute the loss and its gradients
            loss = criterion(outputs, targets)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        # if loss in inf stop
        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            raise ValueError('Loss is infinite or NaN')
            # sys.exit(1)

        # reset gradients every accum_iter steps
        if (i + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        # adjust lrs
        min_lr, max_lr = 10.0, 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        # save true and predicted labels
        prediction_softmax = nn.Softmax(dim=1)(outputs)
        _, prediction_decode = torch.max(prediction_softmax, 1)

        predictions.extend(prediction_decode.cpu().detach().numpy())
        true_labels.extend(targets.cpu().detach().numpy())

    # gather the stats from all processes
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    # compute avg loss over batches and Bacc
    avg_loss = np.mean(losses)
    bacc = balanced_accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")

    if epoch % 5 == 0:
        print(
            f"[Train] Epoch {epoch} - Loss: {avg_loss:.4f}, Bacc: {bacc:.4f}, F1-score: {f1:.4f}"
        )

    return OrderedDict({
        'epoch': epoch,
        'loss': avg_loss,
        'bacc': bacc,
        'f1': f1
    })


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable,
    epoch: Union[int, str],
    device: torch.device,
    num_class: int,
    mode: str,
    get_embeddings: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    args=None,
    save_predictions: bool = False,
) -> Optional[dict]:
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []

    # switch to evaluation mode
    model.eval()

    all_embeddings = {
        'embeddings': None,
        'targets': None
    }
    # Get dataloader length
    data_loader_len = len(data_loader)  # type: ignore
    period = max(int(round(0.1 * data_loader_len)), 1)
    # assert period > 0
    for bi, batch in enumerate(data_loader):
        images = batch[0]
        targets = batch[-1]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        true_label = F.one_hot(targets.to(torch.int64), num_classes=num_class)

        # compute output
        # TODO: Do we really want to use mixed precision here?
        with autocast('cuda', enabled=True):
            if (
                (
                    isinstance(epoch, int)
                    and bi == 0
                    and epoch % 10 == 0
                    and args is not None
                 )
                or (
                    get_embeddings
                    and bi % period == 0
                )
            ):
                epoch_str = str(epoch).zfill(3)
                if get_embeddings:
                    assert save_path is not None
                    output_dir = Path(save_path).parent
                else:
                    assert args is not None
                    output_dir = args.output_dir
                save_fn = Path(output_dir, "debug", f"{epoch_str}_{mode}_{bi}.jpg")
                save_fn.parent.mkdir(exist_ok=True)
                print('Saving images for debugging')
                print('  images', images.shape, images.min().item(), images.max().item())
                print('  targets', targets.shape, targets.min().item(), targets.max().item())
                save_image(images, save_fn, normalize=True)
            if get_embeddings:
                # embeddings = model(images, get_embeddings=True)
                # embeddings = model.features(images)
                try:
                    embeddings = model.forward_features(images)  # type: ignore
                except AttributeError:
                    embeddings = model(images)
                if save_path is not None:
                    for k, v in [('embeddings', embeddings), ('targets', targets)]:
                        try:
                            v.cpu()
                        except AttributeError:
                            v = v['x_norm_patchtokens']
                            # Compute the average of the patch embeddings
                            v = v.mean(dim=1)
                        if all_embeddings[k] is None:
                            all_embeddings[k] = v.cpu().detach().numpy()
                        else:
                            all_embeddings[k] = np.concatenate(
                                [all_embeddings[k], v.cpu().detach().numpy()]  # type: ignore
                            )
                    print(all_embeddings['embeddings'].shape, all_embeddings['targets'].shape)
                    # with open(save_fn, "wb") as f:
                    #     pickle.dump(
                    #         dict(
                    #             embeddings=embeddings.cpu().detach().numpy(),
                    #             targets=targets.cpu().detach().numpy(),
                    #         ),
                    #         f,
                    #     )
                continue
            output = model(images)
            loss = criterion(output, targets)
            losses.append(loss.item())

            prediction_softmax = nn.Softmax(dim=1)(output)
            _, prediction_decode = torch.max(prediction_softmax, 1)
            _, true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())

    if get_embeddings and save_path is not None:
        assert all_embeddings['embeddings'] is not None
        assert all_embeddings['targets'] is not None
        save_fn = Path(save_path, "embeddings_targets.npz")
        np.savez_compressed(
            save_fn,
            embeddings=all_embeddings['embeddings'],
            targets=all_embeddings['targets']
        )
        # with open(save_fn, "wb") as f:
        #     pickle.dump(all_embeddings, f)
        return

    # if get_embeddings:
    #     return

    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)

    if save_predictions:
        print("Saving predictions")
        print('\ttrue_label', true_label_decode_list.shape)
        print('\tprediction', prediction_decode_list.shape)
        assert save_path is not None, "save_path must be provided"
        save_fn = Path(save_path, "predictions.npz")
        np.savez_compressed(
            save_fn,
            true_label_decode_list=true_label_decode_list,
            prediction_decode_list=prediction_decode_list,
            true_label_onehot_list=true_label_onehot_list,
            prediction_list=prediction_list,
        )
        return

    avg_loss = np.mean(losses)
    acc = balanced_accuracy_score(true_label_decode_list, prediction_decode_list)
    auc_roc = roc_auc_score(
        true_label_onehot_list,
        prediction_list,
        multi_class="ovr",
        average="weighted",
    )
    auc_pr = average_precision_score(
        true_label_onehot_list, prediction_list, average="weighted"
    )
    f1 = f1_score(
        true_label_decode_list,
        prediction_decode_list,
        average="weighted",
        zero_division=0.0,  # type: ignore
    )
    mcc = matthews_corrcoef(true_label_decode_list, prediction_decode_list)

    if type(epoch) != str and epoch % 5 == 0:
        print(
            "[{}] Epoch {} - Loss: {:.4f}, Bacc: {:.4f} AUROC: {:.4f} AP: {:.4f} F1-score: {:.4f} MCC: {:.4f}".format(
                mode, epoch, avg_loss, acc, auc_roc, auc_pr, f1, mcc
            )
        )

    return OrderedDict({
        'epoch': epoch,
        'loss': avg_loss,
        'bacc': acc,
        'auroc': auc_roc,
        'ap': auc_pr,
        'f1': f1,
        'mcc': mcc
    })



class PartialImageFolder(datasets.ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        percentage=0.5,
    ):
        super(PartialImageFolder, self).__init__(
            root, transform, target_transform, loader
        )
        self.percentage = percentage
        self.sample_indices = self._generate_sample_indices()

    def _generate_sample_indices(self):
        sample_indices = {}
        for target_class in self.classes:
            class_dir = os.path.join(self.root, target_class)
            all_images = os.listdir(class_dir)
            num_samples = int(len(all_images) * self.percentage)
            sample_indices[target_class] = random.sample(
                range(len(all_images)), num_samples
            )
        return sample_indices

    def __getitem__(self, index):
        print(self.classes)
        _path, target = self.samples[index]
        class_samples = self.sample_indices[
            self.classes[target]
        ]  # Get samples for the target class
        sample_index = class_samples[
            index % len(class_samples)
        ]  # Use modulo to handle index out of range
        sample_path = os.path.join(self.samples[sample_index][0])
        sample = self.loader(sample_path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def build_dataset(subset, args, augment=False):
    transform = build_transform(args, subset, augment)
    root = os.path.join(args.data_path, subset)
    if subset == "train" and args.label_efficiency_exp:
        dataset = PartialImageFolder(
            root, transform=transform, percentage=args.train_ds_perc
        )
    else:
        dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


class MinMaxScaler:
    """
    Transforms each channel to the range [0, 1].
    """
    def __call__(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())


class MinMaxScalerChannel:
    def __init__(self) -> None:
        super().__init__()
        self.scaler = MinMaxScaler()

    def __call__(self, tensor):
        for i in range(tensor.shape[0]):
            if tensor[i].max() > 0:
                tensor[i] = self.scaler(tensor[i:i+1].clone())
        return tensor


class NaiveScaler:
    """
    Transforms each channel to the range [0, 1], if it is not already.
    """
    def __call__(self, tensor):
        if tensor.min() < 0:
            raise ValueError("Tensor contains negative values")
        elif tensor.max() > 1 and tensor.max() <= 255:
            tensor = tensor / 255.0
        elif tensor.max() > 255:
            tensor = tensor / 65535.0
        return tensor


class NaiveScalerChannel:
    """
    Transforms each channel to the range [0, 1], if it is not already.
    """
    def __init__(self) -> None:
        super().__init__()
        self.scaler = NaiveScaler()

    def __call__(self, tensor):
        for i in range(tensor.shape[0]):
            tensor[i] = self.scaler(tensor[i:i+1].clone())
        return tensor


class Identity:
    def __call__(self, img):
        return img


class ToRGB:
    def __call__(self, img: torch.Tensor):
        return img.repeat(3, 1, 1)


class RandomIntensity(nn.Module):
    def __init__(self, intensity_range=(0.8, 1.2)):
        super().__init__()
        self.intensity_range = intensity_range

    @staticmethod
    def get_abs_max(tensor):
        if tensor.max() <= 1:
            abs_max = 1
        elif tensor.max() > 1 and tensor.max() <= 255:
            abs_max = 255
        elif tensor.max() > 255:
            abs_max = 65535
        else:
            raise ValueError(
                "Image values are not in the expected range:"
                f" [{tensor.max()}, {tensor.min()}], {torch.unique(tensor)}"
            )
        return abs_max

    def forward(self, img):
        intensity = torch.empty(1).uniform_(*self.intensity_range).item()
        return torch.clamp(img * intensity, 0, self.get_abs_max(img))


class RandomIntensityChannel(nn.Module):
    def __init__(self, intensity_range=(0.8, 1.2)):
        super().__init__()
        self.intensity_range = intensity_range
        self.intensity = RandomIntensity(intensity_range)

    def forward(self, img):
        for i in range(img.shape[0]):
            if img[i].max() > 0:
                img[i] = self.intensity(img[i:i+1].clone())
        return img


class RandomAffineChannel(tvtr.RandomAffine):
    """Same as RandomAffine but with a random rotation for every
    channel.
    """
    def __init__(self, p=1.0, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        if random.random() < (1 - self.p):
            return img

        if self.fill == 0.5:
            fill = random.uniform(img.min().item(), img.max().item())
        else:
            fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)]
            else:
                fill = [float(f) for f in fill]  # type: ignore

        img_size = VF.get_image_size(img)

        for i in range(img.shape[0]):
            # Apply a transformation only in 90% of the cases
            if random.random() < 0.9:
                ret = self.get_params(
                    self.degrees, self.translate, self.scale, self.shear,
                    img_size
                )
                img[i] = VF.affine(
                    img[i:i+1].clone(), *ret, interpolation=self.interpolation,
                    fill=fill, center=self.center  # type: ignore
                )
        return img


def build_transform(args, subset, augment):
    multimodal = len(args.input_modality.split('-')) > 1
    if multimodal:
        end = 'for multimodal data'
    else:
        end = ''
    print(f'>>> Building transform "{subset}"', end)
    intensity_msg = 'Random intensity shift'
    intensity = RandomIntensityChannel()
    if args.fill is None:
        if 'kermany' in args.data_set.lower():
            fill = 1
        else:
            fill = 0
    else:
        fill = args.fill
    affine_msg = f'Random affine (fill={fill})'
    affine = RandomAffineChannel(
        degrees=10,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=5,
        interpolation=tvtr.InterpolationMode.BILINEAR,
        fill=fill,
    )
    if args.no_affine:
        affine_msg = 'No random affine'
        affine = Identity()
    grayscale = Identity()
    min_max = Identity()
    scaler_list = [ NaiveScalerChannel() ]
    if not multimodal:
        grayscale = tvtr.Grayscale(num_output_channels=1)
    if args.model == 'MIRAGE':
        # If it is any of our models
        scaler_msg = 'Naive scaler'
        if args.no_minmax:
            scaler_list = [ Identity() ]
            min_max = Identity()
        else:
            scaler_list += [ MinMaxScalerChannel() ]
            min_max = MinMaxScalerChannel()
    elif args.model == 'VisionFM':
        # OCT standard mean and std for VisionFM
        # https://github.com/ABILab-CUHK/VisionFM/blob/main/utils.py#L46
        mean = (0.21091926, 0.21091926, 0.21091919)
        std = (0.17598894, 0.17598891, 0.17598893)
        scaler_msg = 'OCT scaler'
        if not multimodal:
            # First to RGB, then normalize
            scaler_list += [ ToRGB() ]
        scaler_list += [
            tvtr.Normalize(mean, std)
        ]
    else:
        # If it is a SOTA model
        scaler_msg = 'ImageNet scaler'
        if not multimodal:
            # First to RGB, then normalize
            scaler_list += [ ToRGB() ]
        scaler_list += [
            tvtr.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]

    transforms_list = [
        tvtr.Resize(
            size=(args.input_size, args.input_size),
            interpolation=tvtr.InterpolationMode.BILINEAR,
        ),
        grayscale,
        tvtr.ToTensor(),
        tvtr.ConvertImageDtype(torch.float32),
        min_max,
    ]
    if augment:
        print('Random horizontal flip (0.5)')
        print(intensity_msg)
        print(affine_msg)
        transforms_list += [
            tvtr.RandomHorizontalFlip(p=0.5),
            intensity,
            affine,
        ]
    print(scaler_msg)
    transforms_list += scaler_list
    transforms = tvtr.Compose(transforms_list)

    return transforms

