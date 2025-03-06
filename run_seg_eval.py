import argparse
from pathlib import Path
import json

import numpy as np
from skimage import io
from skimage.transform import resize
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
import pandas as pd

from mutils.misc import SortingHelpFormatter



def get_args():
    parser = argparse.ArgumentParser(
        'Evaluate segmentation results',
        formatter_class=SortingHelpFormatter
    )
    parser.add_argument(
        '-d', '--datasets_path', type=str, default='./__datasets/Segmentation/',
        help='Path to the datasets directory. (default: %(default)s)'
    )
    parser.add_argument(
        '-m', '--model_path', type=str, required=True,
        help='Path to the trained model directory. It expects a subdirectory'
            ' "preds" with the predictions. (required)'
    )
    parser.add_argument(
        '-e', '--external', type=str, default=None,
        help='Name of the external dataset on which the model made the '
        ' predictions. (default: %(default)s)'
    )
    parser.add_argument(
        '--ignore_bg', action='store_true',
        help='Ignore the background class when computing metrics.'
            ' (default: %(default)s)'
    )
    parser.add_argument('--no_ignore_bg', dest='ignore_bg', action='store_false')
    parser.set_defaults(ignore_bg=True)
    parser.add_argument(
        '--empty_sets_nan', action='store_true',
        help='Return NaN if the prediction OR the ground truth is empty.'
            ' (default: %(default)s)'
    )
    parser.add_argument('--no_empty_sets_nan', dest='empty_sets_nan', action='store_false')
    parser.set_defaults(empty_sets_nan=True)
    return parser.parse_args()


def dice_score(y_pred, y_true):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    return 2.0 * intersection / (np.sum(y_true) + np.sum(y_pred) + 1e-6)


def iou_score(y_pred, y_true):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-6)


def to_one_hot(y, num_classes=2):
    y_onehot = np.zeros((num_classes,) + y.shape, dtype=np.float32)
    for i in range(num_classes):
        y_onehot[i] = y == i
    return y_onehot[np.newaxis]


def volume_hausdorff_distance(y_pred, y_true, percentile=95, empty_sets_nan=True):
    hd95s = []
    for i in range(y_pred.shape[0]):
        unique_pred = np.unique(y_pred[i])
        unique_true = np.unique(y_true[i])
        if unique_true.size == 1 and unique_pred.size == 1:
            # If the prediction and the ground truth are empty, the
            #   distance is 0.
            hd95 = 0
        elif unique_true.size == 1 or unique_pred.size == 1:
            # If only one of them is empty, the distance is the diagonal
            #   length of the image, i.e., the maximum possible
            #   distance.
            if empty_sets_nan:
                hd95 = np.nan
            else:
                hd95 = np.sqrt(
                    np.power(y_true[i].shape[0], 2)
                    + np.power(y_true[i].shape[1], 2)
                )
        else:
            hd95 = compute_hausdorff_distance(
                to_one_hot(y_pred[i]),
                to_one_hot(y_true[i]),
                percentile=percentile
            ).item()
        hd95s.append(hd95)
    return np.nanmean(hd95s)


def print_mean_metrics(results_df):
    print('  Dice: {:.2f}'.format(results_df['Dice'].mean() * 100))
    print('  IoU: {:.2f}'.format(results_df['IoU'].mean() * 100))
    print('  HD95: {:.2f}'.format(results_df['HD95'].mean()))


def translate_to_dukeiamd_from_aroi(y_pred, y_true):
    '''Convert from AROI to Duke iAMD semantic classes.
    Duke iAMD semantic classes:
        {
            "0": "Invalid",
            "51": "Above ILM",
            "102": "ILM-Inner RPEDC",
            "153": "Inner RPEDC-Outer BM",
            "204": "Below BM"
        }
    AROI semantic classes:
        {
            "0": "Above ILM",
            "23": "ILM-IPL/INL",
            "46": "IPL/INL-RPE",
            "69": "RPE-BM",
            "92": "Under BM",
            "115": "Cyst",
            "138": "PED",
            "161": "SRF"
        }
    '''
    mapping = {
        0: 51,
        23: 102,
        46: 102,
        69: 153,
        92: 204,
        # Map to invalid
        115: 0,
        138: 0,
        161: 0,
    }
    y_pred_dukeiamd = np.vectorize(mapping.get)(y_pred)
    y_true_dukeiamd = y_true.copy()
    # Assign invalid class to the pixels whose classes are not in the
    #   Duke iAMD semantic classes and were assigned to the invalid
    #   class.
    y_true_dukeiamd[y_pred_dukeiamd == 0] = 0
    return y_pred_dukeiamd, y_true_dukeiamd


def main():
    args = get_args()
    model_path = Path(args.model_path)
    datasets_path = Path(args.datasets_path)
    if args.external is not None:
        preds_path = model_path / ('preds_' + args.external)
        dataset = args.external
        suffix = f'_{args.external}'
        print(f'Evaluating on external dataset: {args.external}')
        if not (datasets_path / dataset / 'test').exists():
            gt_masks_path = datasets_path / dataset / 'semseg'
        else:
            gt_masks_path = datasets_path / dataset / 'test' / 'semseg'
        source_dataset = model_path.parent.name
    else:
        preds_path = model_path / 'preds'
        dataset = model_path.parent.name
        suffix = ''
        gt_masks_path = datasets_path / dataset / 'test' / 'semseg'
        source_dataset = dataset

    if dataset.startswith('Duke_iAMD') and source_dataset == 'AROI':
        print('Using translator from AROI to Duke iAMD.')
        translator = translate_to_dukeiamd_from_aroi
    else:
        translator = lambda x, y: (x, y)

    if not preds_path.exists():
        raise ValueError(f'Path "{preds_path}" does not exist.')

    if not gt_masks_path.exists():
        raise ValueError(f'Path "{gt_masks_path}" does not exist.')

    with open(datasets_path / dataset / 'INFO.json', 'r') as f:
        info = json.load(f)

    sem_classes = {}
    for _k, v in info.items():
        sem_classes[v['value']] = v['label']

    print('Semantic classes:')
    print(json.dumps(sem_classes, indent=4))

    volumes = {}
    for gt_mask_fn in gt_masks_path.iterdir():
        last_underscore = gt_mask_fn.stem.rfind('_')
        scan_id = gt_mask_fn.stem[:last_underscore]
        if scan_id not in volumes:
            volumes[scan_id] = {}
        slice_num = int(gt_mask_fn.stem[last_underscore+1:])
        if slice_num not in volumes[scan_id]:
            volumes[scan_id][slice_num] = gt_mask_fn.stem

    # Order slices
    for scan_id, slice_nums in volumes.items():
        volumes[scan_id] = [volumes[scan_id][i] for i in sorted(slice_nums.keys())]

    fg_classes = []
    invalid_classes = []
    for sc in sem_classes:
        if 'invalid' in sem_classes[sc].lower():
            invalid_classes.append(sc)
        elif not (args.ignore_bg and (
            'bg' in sem_classes[sc].lower()
            or 'background' in sem_classes[sc].lower()
            or 'above ilm' in sem_classes[sc].lower()
        )):
            fg_classes.append(sc)

    print('Foreground classes:', fg_classes)
    print('Invalid classes:', invalid_classes)

    results_df = pd.DataFrame()
    for scan_id, slices in volumes.items():
        print(scan_id)
        gt = []
        pred = []
        for slice_id in slices:
            gt_mask_fn = gt_masks_path / (slice_id + '.png')
            gt_mask = io.imread(gt_mask_fn)
            gt.append(gt_mask)
            pred_mask_fn = preds_path / (slice_id + '_pred.png')
            pred_mask = io.imread(pred_mask_fn)
            pred.append(pred_mask)
        gt = np.array(gt)
        pred = np.array(pred)
        if gt.shape != pred.shape:
            pred = resize(pred, gt.shape, order=0, preserve_range=True)
        pred, gt = translator(pred, gt)
        for sc in invalid_classes:
            pred[gt == sc] = sc
        for sc in fg_classes:
            sc_gt = gt == sc
            sc_pred = pred == sc
            sc_dice = dice_score(sc_pred, sc_gt)
            sc_iou = iou_score(sc_pred, sc_gt)
            sc_hd95 = volume_hausdorff_distance(sc_pred, sc_gt, 95, args.empty_sets_nan)
            results_df = pd.concat([
                results_df,
                pd.DataFrame({
                    'ID': [scan_id],
                    'Class': [sem_classes[sc]],
                    'Dice': [sc_dice],
                    'IoU': [sc_iou],
                    'HD95': [sc_hd95]
                })
            ], ignore_index=True)

    print('\nAverage results:')
    print_mean_metrics(results_df)

    if dataset in ['Duke_DME', 'AROI']:
        # Save separate results for layers and lesions
        lesion_classes = [
            # Duke_DME:
            'Fluid',
            # AROI:
            'Cyst', 'PED', 'SRF'
        ]
        results_layers_df = results_df[~results_df['Class'].isin(lesion_classes)]
        print('\nAverage results (layers):')
        print_mean_metrics(results_layers_df)
        results_lesions_df = results_df[results_df['Class'].isin(lesion_classes)]
        print('\nAverage results (lesions):')
        print_mean_metrics(results_lesions_df)
        with open(model_path / f'results_layers{suffix}.csv', 'w') as f:
            results_layers_df.to_csv(f, index=False)
        with open(model_path / f'results_lesions{suffix}.csv', 'w') as f:
            results_lesions_df.to_csv(f, index=False)
    else:
        # Save all results in a single file
        with open(model_path / f'results{suffix}.csv', 'w') as f:
            results_df.to_csv(f, index=False)
    print(f'\nResults saved to "{model_path}" path.')



if __name__ == '__main__':
    main()

