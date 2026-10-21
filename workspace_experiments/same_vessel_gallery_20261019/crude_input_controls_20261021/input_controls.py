"""Matched source-trained heads for pre-convolution body/full input controls."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.models import resnet50
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score, log_loss
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT.parent / 'local_condition_module'
PREVIOUS = ROOT.parent / 'condition_outcomes_20261021'
sys.path.insert(0, str(LOCAL))
sys.path.insert(0, str(ROOT.parent / 'conditional_evidence_pilot'))
from native_input import crop_pad
from prepare import masks_for, WEIGHTS
import run_pilot as prior

ARMS = ('full_target', 'body_target', 'full_global', 'body_global', 'outside_global')


def load_data():
    frame = pd.read_csv(prior.ROOT / 'cohort.csv', dtype={'mmsi': str})
    views = json.loads((LOCAL / 'view_snapshot.json').read_text(encoding='utf-8'))
    assert [v['obs_id'] for v in views] == frame.obs_id.tolist()
    conflict = pd.read_csv(LOCAL / 'crossview_alignment.csv').set_index('obs_id')
    disabled = conflict.loc[frame.obs_id, 'disable_new_vh'].to_numpy(bool)
    return frame, views, disabled


def prepare_view(row, view, pol):
    raw = np.asarray(Image.open(row['image_' + pol]).convert('L'))
    assert raw.shape == (view['h'], view['w'])
    pix, mask, place = crop_pad(raw, masks_for(view), 160)
    assert place['native_target_pixels'] == place['retained_target_pixels']
    valid = mask.sum(0)
    axis = view['priors'].get('ship_axis')
    shape = np.array([axis['length_px'], axis['width_px']], np.float32) if axis else np.zeros(2, np.float32)
    return pix.astype(np.float32)[None] / 255., mask, valid, shape, place


def make_stem():
    torch.set_num_threads(2)
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
    return torch.nn.Sequential(model.conv1, model.bn1, model.relu).cuda().eval()


def pool(features, footprint):
    weight = F.avg_pool2d(footprint, 4)
    avg = (features * weight).sum((2, 3)) / weight.sum((2, 3)).clamp_min(1e-8)
    maximum = features.masked_fill(weight <= 0, -torch.inf).amax((2, 3))
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return torch.cat((avg, maximum), 1)


def describe(stem, x, target, valid, inputs=('full', 'body', 'outside')):
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    accum = {}
    for kind in inputs:
        source = x if kind == 'full' else x * (target if kind == 'body' else valid - target)
        for k in range(4):
            image = torch.rot90(source, k, (-2, -1)).expand(-1, 3, -1, -1)
            feat = F.avg_pool2d(stem((image - mean) / std), 2)
            for name, mask in (('target', target), ('global', valid)):
                arm = kind + '_' + name
                if arm not in ARMS:
                    continue
                value = pool(feat, torch.rot90(mask, k, (-2, -1))) / 4
                accum[arm] = accum.get(arm, 0) + value
    return accum


def extract():
    started = time.perf_counter()
    frame, views, disabled = load_data()
    stem = make_stem()
    descriptors = {arm: np.zeros((len(frame), 2, 128), np.float32) for arm in ARMS}
    shapes = np.zeros((len(frame), 2, 2), np.float32)
    usable = np.zeros((len(frame), 2), bool)
    coverage, checks = [], []
    with torch.inference_mode():
        for first in range(0, len(frame), 24):
            images, targets, valids, addresses = [], [], [], []
            for i in range(first, min(first + 24, len(frame))):
                row = frame.iloc[i]
                if row.obs_id == 31237:
                    continue
                for p, pol in enumerate(('vv', 'vh')):
                    pix, mask, valid, shape, place = prepare_view(row, views[i]['images'][pol], pol)
                    ok = bool(mask[0].any() and not (p == 1 and disabled[i]))
                    usable[i, p] = ok
                    coverage.append(dict(obs_id=int(row.obs_id), pol=pol, usable=ok,
                                         disabled_alignment=bool(p == 1 and disabled[i]), **place))
                    if not ok:
                        continue
                    shapes[i, p] = shape
                    images.append(pix)
                    targets.append(mask[0:1])
                    valids.append(valid[None])
                    addresses.append((i, p))
            if not images:
                continue
            x, target, valid = [torch.from_numpy(np.stack(z)).cuda() for z in (images, targets, valids)]
            result = describe(stem, x, target, valid)
            for arm, values in result.items():
                values = values.cpu().numpy()
                for b, (i, p) in enumerate(addresses):
                    descriptors[arm][i, p] = values[b]
            if first == 0:
                changed = x[:2] * target[:2] + .731 * (valid[:2] - target[:2])
                body = describe(stem, changed, target[:2], valid[:2], ('body',))
                rotated = describe(stem, *[torch.rot90(t[:2], 1, (-2, -1)) for t in (x, target, valid)], ('full',))
                for arm in ('body_target', 'body_global'):
                    error = float((body[arm] - result[arm][:2]).abs().max())
                    assert error == 0, (arm, error)
                    checks.append(dict(test='outside_change_cannot_enter_' + arm, max_abs=error))
                error = float((rotated['full_global'] - result['full_global'][:2]).abs().max())
                assert error < 2e-5
                checks.append(dict(test='C4_joint_rotation', max_abs=error))
            if first % 480 == 0:
                print('extract', first, len(frame), round(time.perf_counter() - started, 1), flush=True)
    arrays = {arm: np.concatenate((value.reshape(len(frame), -1), shapes.reshape(len(frame), -1)), 1)
              for arm, value in descriptors.items()}
    np.savez_compressed(ROOT / 'features.npz', **arrays, usable=usable)
    pd.DataFrame(coverage).to_csv(ROOT / 'input_coverage.csv', index=False)
    pd.DataFrame(checks).to_csv(ROOT / 'input_checks.csv', index=False)
    (ROOT / 'EXTRACTION.json').write_text(json.dumps(dict(seconds=time.perf_counter() - started,
        rows=len(frame), vv_usable=int(usable[:, 0].sum()), vh_usable=int(usable[:, 1].sum()),
        dimension=260, canvas=160, resize=False), indent=2), encoding='utf-8')


def metrics(y, scores):
    values = prior.metrics(y, scores)
    values['macro_f1'] = float(f1_score(y, scores.argmax(1), labels=np.arange(8), average='macro', zero_division=0))
    values['nll'] = float(log_loss(y, softmax(scores, axis=1), labels=np.arange(8)))
    return values


def fit():
    started = time.perf_counter()
    frame = pd.read_csv(prior.ROOT / 'cohort.csv', dtype={'mmsi': str})
    cache = np.load(ROOT / 'features.npz')
    excluded = pd.read_csv(LOCAL / 'EXCLUSIONS.csv').obs_id.to_numpy()
    legal = cache['usable'][:, 0] & ~frame.obs_id.isin(excluded).to_numpy()
    y = frame.class_id.to_numpy()
    measures, selection, predictions, partitions = [], [], [], []
    all_scores = {}
    for fold in range(3):
        train, val, test = [ix[legal[ix]] for ix in prior.indices(frame, fold)]
        assert set(y[train]) == set(range(8))
        for name, ix in (('train', train), ('val', val), ('test', test)):
            partitions.extend(dict(fold=fold, partition=name, obs_id=int(frame.iloc[i].obs_id)) for i in ix)
        weight = prior.weights(frame, train)
        for arm in ARMS:
            raw = cache[arm]
            mean, std = raw[train].mean(0), np.maximum(raw[train].std(0), 1e-6)
            x = (raw - mean) / std / np.sqrt(raw.shape[1])
            candidates = []
            for alpha in (.1, 1., 10.):
                model = RidgeClassifier(alpha=alpha, solver='cholesky').fit(x[train], y[train], sample_weight=weight)
                metric = metrics(y[val], model.decision_function(x[val]))
                selection.append(dict(fold=fold, arm=arm, alpha=alpha, **metric))
                candidates.append((metric, alpha, model))
            _, alpha, model = max(candidates, key=lambda v: (v[0]['ba'], -v[0]['brier']))
            np.savez_compressed(ROOT / f'head_f{fold}_{arm}.npz', mean=mean, std=std,
                coef=model.coef_, intercept=model.intercept_, dimension=raw.shape[1])
            for name, ix in (('train', train), ('val', val), ('test', test)):
                scores = model.decision_function(x[ix])
                measures.append(dict(fold=fold, arm=arm, partition=name, alpha=alpha, **metrics(y[ix], scores)))
            all_scores[(fold, arm)] = (test, scores)
            other = scores.copy()
            other[np.arange(len(test)), y[test]] = -np.inf
            for b, i in enumerate(test):
                row = frame.iloc[i]
                predictions.append(dict(fold=fold, arm=arm, obs_id=int(row.obs_id), mmsi=row.mmsi,
                    class_id=int(y[i]), port=row.port, product=row['product'], prediction=int(scores[b].argmax()),
                    correct=bool(scores[b].argmax() == y[i]), true_score=float(scores[b, y[i]]),
                    wrong_score=float(other[b].max()), margin=float(scores[b, y[i]] - other[b].max())))
            print('fit', fold, arm, measures[-1], flush=True)
    pd.DataFrame(measures).to_csv(ROOT / 'metrics.csv', index=False)
    pd.DataFrame(selection).to_csv(ROOT / 'validation_selection.csv', index=False)
    pd.DataFrame(partitions).to_csv(ROOT / 'partitions.csv', index=False)
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(ROOT / 'predictions.csv', index=False)
    summary = []
    for arm in ARMS:
        ix = np.concatenate([all_scores[(fold, arm)][0] for fold in range(3)])
        scores = np.concatenate([all_scores[(fold, arm)][1] for fold in range(3)])
        np.savez_compressed(ROOT / f'scores_{arm}.npz', rows=ix, scores=scores)
        summary.append(dict(arm=arm, **metrics(y[ix], scores)))
    pd.DataFrame(summary).to_csv(ROOT / 'summary.csv', index=False)
    predictions.groupby(['arm', 'class_id']).agg(n=('correct', 'size'), recall=('correct', 'mean')).reset_index().to_csv(ROOT / 'class_recall.csv', index=False)
    crude_pairs(predictions)
    (ROOT / 'FIT.json').write_text(json.dumps(dict(seconds=time.perf_counter() - started,
        selected_heads=15, alpha_candidate_fits=45, folds=3, stochastic_training=False,
        trainable='separately fitted linear classifier heads; frozen stem', target_support=False), indent=2), encoding='utf-8')


def crude_pairs(predictions):
    pairs = pd.read_csv(PREVIOUS / 'crude_pair_details.csv')
    records, summary = [], []
    for arm in ARMS:
        p = predictions[predictions.arm.eq(arm)].set_index('obs_id')
        for row in pairs.itertuples():
            if row.obs_id_a not in p.index or row.obs_id_b not in p.index:
                continue
            a, b = p.loc[row.obs_id_a], p.loc[row.obs_id_b]
            records.append(dict(arm=arm, pair_id=row.pair_id, obs_id_a=row.obs_id_a, obs_id_b=row.obs_id_b,
                fold=row.fold, correct_a=bool(a.correct), correct_b=bool(b.correct),
                delta_margin=float(b.margin-a.margin), vv_contrast=row.vv_contrast, vh_contrast=row.vh_contrast))
    result = pd.DataFrame(records)
    result.to_csv(ROOT / 'crude_pairs.csv', index=False)
    for arm, g in result.groupby('arm', sort=False):
        summary.append(dict(arm=arm, pairs=len(g), both_correct=int((g.correct_a & g.correct_b).sum()),
            one_correct=int((g.correct_a != g.correct_b).sum()), both_wrong=int((~g.correct_a & ~g.correct_b).sum()),
            vv_rho=float(spearmanr(g.vv_contrast, g.delta_margin).statistic),
            vh_rho=float(spearmanr(g.vh_contrast, g.delta_margin).statistic)))
    pd.DataFrame(summary).to_csv(ROOT / 'crude_pair_summary.csv', index=False)


def perturb():
    frame, views, disabled = load_data()
    pairs = pd.read_csv(PREVIOUS / 'crude_pair_details.csv')
    ids = set(pairs.obs_id_a) | set(pairs.obs_id_b)
    predictions = pd.read_csv(ROOT / 'predictions.csv')
    legal_ids = set(predictions.obs_id)
    indices = [i for i, oid in enumerate(frame.obs_id) if oid in ids & legal_ids]
    stem = make_stem()
    heads = [np.load(ROOT / f'head_f{fold}_full_global.npz') for fold in range(3)]
    records = []
    with torch.inference_mode():
        for first in range(0, len(indices), 16):
            subset = indices[first:first+16]
            inputs, targets, valids, rests, addresses, shapes = [], [], [], [], [], np.zeros((len(subset), 2, 2), np.float32)
            for b, i in enumerate(subset):
                row = frame.iloc[i]
                for p, pol in enumerate(('vv', 'vh')):
                    pix, masks, valid, shape, _ = prepare_view(row, views[i]['images'][pol], pol)
                    if not masks[0].any() or (p == 1 and disabled[i]):
                        continue
                    inputs.append(pix); targets.append(masks[0:1]); rests.append(masks[2:3]); valids.append(valid[None])
                    addresses.append((b, p)); shapes[b, p] = shape
            x, target, rest, valid = [torch.from_numpy(np.stack(z)).cuda() for z in (inputs, targets, rests, valids)]
            settings = [('baseline', 'none', 0.)] + [(r, p, s) for r in ('target', 'rest') for p in ('vv', 'vh') for s in (-.05, .05)]
            for region, pol, shift in settings:
                selector = x.new_tensor([float(p == ('vv', 'vh').index(pol)) if pol != 'none' else 0.
                                         for _, p in addresses])[:, None, None, None]
                mask = (target if region == 'target' else rest) * selector
                before_clip = x + mask * shift
                altered = before_clip.clamp(0, 1)
                descriptor = describe(stem, altered, target, valid, ('full',))['full_global'].cpu().numpy()
                clipped = (((before_clip < 0) | (before_clip > 1)) * mask.bool()).sum((1, 2, 3)).cpu().numpy()
                count = mask.sum((1, 2, 3)).cpu().numpy()
                block = np.zeros((len(subset), 2, 128), np.float32)
                clipcounts = np.zeros((len(subset), 2), float)
                for j, (b, p) in enumerate(addresses):
                    block[b, p] = descriptor[j]
                    clipcounts[b] += (clipped[j], count[j])
                raw = np.concatenate((block.reshape(len(subset), -1), shapes.reshape(len(subset), -1)), 1)
                for b, i in enumerate(subset):
                    row = frame.iloc[i]; head = heads[int(row.fold)]
                    standardized = (raw[b] - head['mean']) / head['std'] / np.sqrt(raw.shape[1])
                    scores = standardized @ head['coef'].T + head['intercept']
                    other = scores.copy(); other[int(row.class_id)] = -np.inf
                    records.append(dict(obs_id=int(row.obs_id), fold=int(row.fold), region=region, pol=pol, shift=shift,
                        prediction=int(scores.argmax()), true_score=float(scores[int(row.class_id)]),
                        wrong_score=float(other.max()), margin=float(scores[int(row.class_id)]-other.max()),
                        clipped_fraction=float(clipcounts[b, 0]/max(clipcounts[b, 1], 1))))
    result = pd.DataFrame(records)
    base = result[result.region.eq('baseline')].set_index('obs_id')
    previous = predictions[predictions.arm.eq('full_global')].set_index('obs_id').loc[base.index]
    assert np.max(np.abs(previous.margin.to_numpy()-base.margin.to_numpy())) < 1e-5
    result['delta_margin'] = result.margin - result.obs_id.map(base.margin)
    result.to_csv(ROOT / 'perturbations.csv', index=False)
    result.groupby(['region', 'pol', 'shift']).agg(n=('obs_id', 'size'), delta_mean=('delta_margin', 'mean'),
        delta_median=('delta_margin', 'median'), fraction_increase=('delta_margin', lambda x: float((x > 0).mean())),
        mean_clipped=('clipped_fraction', 'mean')).reset_index().to_csv(ROOT / 'perturbation_summary.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('extract', 'fit', 'perturb'))
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        globals()[args.stage]()
