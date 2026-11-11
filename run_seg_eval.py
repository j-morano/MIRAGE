"""Evaluation of the segmentation results (Dice, IoU and HD95) with the protocol
of the paper, for AROI, Duke_DME, GOALS and, with `-e Duke_iAMD`, the
cross-dataset evaluation of the models trained on AROI.
"""
import argparse
import gc
import hashlib
import json
import multiprocessing
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label as label_components
from skimage import io
from skimage.transform import resize
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

from mutils.misc import SortingHelpFormatter


# Class indices are the keys of INFO.json
CONFIGS = {
    'Duke_DME': dict(
        vol_layers=[1, 2, 3, 4, 5, 6, 7],
        hd_layers=[1, 2, 3, 4, 5, 6, 7, 8],
        lesions=[9],
        lesion_to_layer={9: 0},
        eval_shape=(496, 600),
    ),
    'AROI': dict(
        vol_layers=[1, 2, 3],
        hd_layers=[1, 2, 3, 4],
        lesions=[5, 6, 7],
        lesion_to_layer={5: 2, 6: 3, 7: 2},
    ),
    'GOALS': dict(
        vol_layers=[1, 2, 3],
        hd_layers=[1, 2, 3, 4],
        lesions=[],
        lesion_to_layer={},
        below_class=4,
        pooled_volume=True,
    ),
    # Cross-dataset (model trained on AROI), evaluated with `-e`
    'Duke_iAMD': dict(
        vol_layers=[1, 2],
        hd_layers=[1, 2, 3],
        lesions=[],
        lesion_to_layer={},
        eval_shape=(512, 1000),
        gt_dataset='Duke_iAMD_labeled',
        gt_map={0: 0, 1: 0, 2: 1, 3: 2, 4: 3},
        pred_map={0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 1, 6: 2, 7: 1},
        labels={1: 'ILM-Inner RPEDC', 2: 'Inner RPEDC-Outer BM', 3: 'Below BM'},
        remove_classes=[1, 2, 3],
        min_size=1000,
        column_mask=True,
        crop_hd=True,
        include='AMD_Subject',
    ),
}


def get_args():
    parser = argparse.ArgumentParser(
        'Evaluate segmentation results (protocol of the paper)',
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
        help='Name of the external dataset on which the model made the'
            ' predictions (folder "preds_<external>" in the model directory),'
            ' e.g., Duke_iAMD for a model trained on AROI. (default: none)'
    )
    parser.add_argument(
        '--empty_hd95', type=str, default='nan', choices=['nan', 'diagonal', 'missed'],
        help='HD95 when only one of the prediction and the ground truth is'
            ' empty. nan: skipped in the means; diagonal: the diagonal of the'
            ' image; missed: the diagonal if only the prediction is empty,'
            ' NaN otherwise. (default: %(default)s)'
    )
    parser.add_argument(
        '--shape', type=int, nargs=2, default=None, metavar=('H', 'W'),
        help='Resize the ground truth and the predictions (nearest) to H x W'
            ' before computing the metrics. (default: dataset-specific)'
    )
    parser.add_argument(
        '--native_shape', action='store_true',
        help='Use the size of the ground truth. (default: %(default)s)'
    )
    parser.add_argument(
        '--all_volumes', action='store_true',
        help='Also evaluate the control subjects of Duke_iAMD. (default: %(default)s)'
    )
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Number of processes. (default: %(default)s)'
    )
    parser.add_argument(
        '--no_cache', action='store_true',
        help='Do not save the results of each patient in <model>/eval_cache*/.'
    )
    return parser.parse_args()


def below_from_merged(cls, below_class):
    """Give `below_class` to the background pixels below the retina."""
    out = cls.copy()
    fg = cls > 0
    rows = np.arange(cls.shape[-2])[:, None]
    deepest = np.where(fg, rows, -1).max(axis=-2, keepdims=True)
    below = (rows > deepest) & (deepest >= 0)
    out[below & ~fg] = below_class
    return out


def nanmean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


def nanstd(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


def volume_dice_iou(pred, gt):
    inter = float(np.sum(pred & gt))
    total = float(pred.sum() + gt.sum())
    if total == 0:
        return np.nan, np.nan
    return 2 * inter / total, inter / (total - inter)


def slice_hd95(pred, gt, one_empty, mode='nan', diagonal=None):
    pred_any, gt_any = pred.any(), gt.any()
    if not pred_any and not gt_any:
        return np.nan if mode == 'nan' else 0.0
    if mode == 'missed':
        if not gt_any:
            return np.nan
        if not pred_any:
            return diagonal
    elif not pred_any or not gt_any:
        return one_empty
    value = compute_hausdorff_distance(
        np.stack([~pred, pred]).astype(np.float32)[np.newaxis],
        np.stack([~gt, gt]).astype(np.float32)[np.newaxis],
        percentile=95
    ).item()
    return np.nan if not np.isfinite(value) else value


def remove_small_components(cls_map, classes, min_size):
    """Remove, per B-scan and class, the components smaller than `min_size`."""
    out = np.zeros_like(cls_map)
    for s in range(cls_map.shape[0]):
        for c in classes:
            labeled, n = label_components(cls_map[s] == c, structure=np.ones((3, 3)))
            if n == 0:
                continue
            keep = np.bincount(labeled.ravel()) >= min_size
            keep[0] = False
            out[s][keep[labeled]] = c
    return out


_CTX = {}  # shared with the worker processes (inherited by fork)


def _init_worker():
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def process_volume(scan_id, slices, ctx):
    config, groups, labels = ctx['config'], ctx['groups'], ctx['labels']
    rows, hd_rows = [], []
    gt = np.array([io.imread(ctx['gt_masks_path'] / f'{s}.png') for s in slices])
    pred = np.array([io.imread(ctx['preds_path'] / f'{s}.png') for s in slices])
    if gt.shape != pred.shape:
        pred = resize(pred, gt.shape, order=0, preserve_range=True).astype(np.uint8)
    if ctx['eval_shape'] is not None:
        shape = (gt.shape[0],) + tuple(ctx['eval_shape'])
        gt = resize(gt, shape, order=0, preserve_range=True).astype(np.uint8)
        pred = resize(pred, shape, order=0, preserve_range=True).astype(np.uint8)
    gt, pred = ctx['lut'][gt], ctx['pred_lut'][pred]
    if 'min_size' in config:
        pred = remove_small_components(pred, config['remove_classes'], config['min_size'])
    for group, (vol_classes, hd_classes) in groups.items():
        gt_g, pred_g = gt, pred
        if group == 'layers':
            for lesion_class, layer_class in config['lesion_to_layer'].items():
                gt_g = np.where(gt_g == lesion_class, layer_class, gt_g)
                pred_g = np.where(pred_g == lesion_class, layer_class, pred_g)
        gt_h, pred_h = gt_g, pred_g
        if group == 'layers' and 'below_class' in config:
            gt_h = below_from_merged(gt_g, config['below_class'])
            pred_h = below_from_merged(pred_g, config['below_class'])
        for c in vol_classes:
            pred_c, gt_c = pred_g == c, gt_g == c
            if config.get('column_mask'):
                cols = gt_c.any(axis=-2, keepdims=True) & ~gt_c.all(axis=-2, keepdims=True)
                pred_c = pred_c & cols
            dice, iou = volume_dice_iou(pred_c, gt_c)
            rows.append(dict(ID=scan_id, Group=group, Class=labels[c], Dice=dice, IoU=iou))
        diagonal = float(np.sqrt(gt.shape[1] ** 2 + gt.shape[2] ** 2))
        one_empty = diagonal if ctx['empty_hd95'] == 'diagonal' else np.nan
        for s, slice_id in enumerate(slices):
            gt_s, pred_s = gt_h[s], pred_h[s]
            if config.get('crop_hd'):
                cols = (gt_s > 0).any(axis=0)
                gt_s, pred_s = gt_s[:, cols], pred_s[:, cols]
            for c in hd_classes:
                hd_rows.append(dict(
                    ID=scan_id, Slice=slice_id, Group=group, Class=labels[c],
                    HD95=slice_hd95(
                        pred_s == c, gt_s == c, one_empty,
                        ctx['empty_hd95'], diagonal)
                ))
    return rows, hd_rows


def _run_volume(item):
    scan_id, slices = item
    cache_dir = _CTX['cache_dir']
    path = None if cache_dir is None else cache_dir / (scan_id.replace(os.sep, '_') + '.pkl')
    if path is not None and path.exists():
        with open(path, 'rb') as f:
            return scan_id, pickle.load(f)
    result = process_volume(scan_id, slices, _CTX)
    gc.collect()
    if path is not None:
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(result, f)
        os.replace(tmp, path)
    return scan_id, result


def main():
    args = get_args()
    model_path = Path(args.model_path)
    datasets_path = Path(args.datasets_path)
    if args.external is not None:
        dataset = args.external
        preds_path = model_path / f'preds_{dataset}'
    else:
        dataset = model_path.parent.name
        preds_path = model_path / 'preds'
    if dataset not in CONFIGS:
        raise ValueError(f'No configuration for "{dataset}". Available: {list(CONFIGS)}')
    config = CONFIGS[dataset]
    gt_dataset = config.get('gt_dataset', dataset)
    gt_masks_path = datasets_path / gt_dataset / 'test' / 'semseg'
    if not gt_masks_path.exists():
        gt_masks_path = datasets_path / gt_dataset / 'semseg'
    for p in (preds_path, gt_masks_path):
        if not p.exists():
            raise ValueError(f'Path "{p}" does not exist.')

    with open(datasets_path / gt_dataset / 'INFO.json', 'r') as f:
        info = json.load(f)
    labels = {int(k): v['label'] for k, v in info.items()}
    labels.update(config.get('labels', {}))
    if 'below_class' in config:
        labels[config['below_class']] = 'Below retina'
    # Class of every gray value, for the ground truth and for the predictions
    lut = np.zeros(256, dtype=np.uint8)
    for k, v in info.items():
        lut[v['value']] = config.get('gt_map', {}).get(int(k), int(k))
    pred_lut = lut
    if 'pred_map' in config:
        with open(datasets_path / model_path.parent.name / 'INFO.json', 'r') as f:
            source_info = json.load(f)
        pred_lut = np.zeros(256, dtype=np.uint8)
        for k, v in source_info.items():
            pred_lut[v['value']] = config['pred_map'][int(k)]

    if args.shape is not None:
        eval_shape = tuple(args.shape)
    elif args.native_shape:
        eval_shape = None
    else:
        eval_shape = config.get('eval_shape')
    include = None if args.all_volumes else config.get('include')

    volumes = {}
    for gt_fn in gt_masks_path.iterdir():
        if config.get('pooled_volume'):
            scan_id, slice_num = 'all', gt_fn.stem
        elif '_' in gt_fn.stem:
            scan_id, slice_num = gt_fn.stem.rsplit('_', 1)
        else:
            scan_id, slice_num = gt_fn.stem, 0
        if include is not None and include not in scan_id:
            continue
        volumes.setdefault(scan_id, {})[int(slice_num)] = gt_fn.stem
    volumes = {k: [v[i] for i in sorted(v)] for k, v in volumes.items()}
    print(f'Volumes: {len(volumes)} | evaluation shape: {eval_shape or "size of the ground truth"}')

    # group -> (classes for Dice/IoU, classes for HD95)
    groups = {}
    if config['vol_layers']:
        groups['layers'] = (config['vol_layers'], config['hd_layers'])
    if config['lesions']:
        groups['lesions'] = (config['lesions'], config['lesions'])

    # The results of the external evaluation do not overwrite the in-domain ones
    suffix = f'_{args.external}' if args.external is not None else ''
    cache_dir = None
    if not args.no_cache:
        key = hashlib.md5(json.dumps(dict(
            dataset=dataset, empty_hd95=args.empty_hd95, shape=eval_shape,
            include=include),
            sort_keys=True, default=str).encode()).hexdigest()[:8]
        cache_dir = model_path / f'eval_cache{suffix}_{key}'
        cache_dir.mkdir(exist_ok=True)
    _CTX.update(
        config=config, groups=groups, labels=labels, lut=lut, pred_lut=pred_lut,
        eval_shape=eval_shape, gt_masks_path=gt_masks_path, preds_path=preds_path,
        empty_hd95=args.empty_hd95, cache_dir=cache_dir)

    todo = list(volumes.items())
    results = {}
    t0 = time.time()
    if args.workers > 1:
        ctx = multiprocessing.get_context('fork')
        with ctx.Pool(args.workers, initializer=_init_worker, maxtasksperchild=4) as pool:
            for k, (scan_id, result) in enumerate(pool.imap_unordered(_run_volume, todo), 1):
                results[scan_id] = result
                print(f'[{k}/{len(todo)}] {scan_id} ({time.time() - t0:.0f} s)', flush=True)
    else:
        for k, item in enumerate(todo, 1):
            scan_id, result = _run_volume(item)
            results[scan_id] = result
            print(f'[{k}/{len(todo)}] {scan_id} ({time.time() - t0:.0f} s)', flush=True)
    rows = [r for s in volumes for r in results[s][0]]
    hd_rows = [r for s in volumes for r in results[s][1]]

    df = pd.DataFrame(rows)
    hd_df = pd.DataFrame(hd_rows)
    mean_results = {'model': model_path.name}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for group in groups:
            d = df[df['Group'] == group]
            h = hd_df[hd_df['Group'] == group]
            # Mean over classes per patient, then over patients
            dice_pat = [nanmean(p['Dice']) * 100 for _, p in d.groupby('ID')]
            iou_pat = [nanmean(p['IoU']) * 100 for _, p in d.groupby('ID')]
            # HD95: mean over classes per B-scan, then over B-scans, then over patients
            hd95_pat = [
                nanmean([nanmean(b['HD95']) for _, b in p.groupby('Slice')])
                for _, p in h.groupby('ID')]
            dice, iou, hd95 = nanmean(dice_pat), nanmean(iou_pat), nanmean(hd95_pat)
            dice_std, iou_std, hd95_std = nanstd(dice_pat), nanstd(iou_pat), nanstd(hd95_pat)
            mean_results[group] = {
                'Dice': dice, 'IoU': iou, 'HD95': hd95,
                'Dice_std': dice_std, 'IoU_std': iou_std, 'HD95_std': hd95_std}
            print(f'\n{group}: Dice {dice:.2f} ± {dice_std:.2f}'
                  f' | IoU {iou:.2f} ± {iou_std:.2f}'
                  f' | HD95 {hd95:.2f} ± {hd95_std:.2f}')
            for c in h['Class'].unique():
                dc = d[d['Class'] == c]
                print('  {:12s} Dice {:6.2f} | IoU {:6.2f} | HD95 {:6.2f}'.format(
                    c,
                    nanmean(dc['Dice']) * 100 if len(dc) else np.nan,
                    nanmean(dc['IoU']) * 100 if len(dc) else np.nan,
                    nanmean(h[h['Class'] == c]['HD95']),
                ))
    df.to_csv(model_path / f'results_volume{suffix}.csv', index=False)
    hd_df.to_csv(model_path / f'results_hd95{suffix}.csv', index=False)
    mean_results = {
        k: ({m: (None if np.isnan(x) else float(x)) for m, x in v.items()}
            if isinstance(v, dict) else v)
        for k, v in mean_results.items()
    }
    with open(model_path / f'results_summary{suffix}.json', 'w') as f:
        json.dump(mean_results, f, indent=4)
    print(f'\nResults saved to "{model_path}" path.')


if __name__ == '__main__':
    main()
