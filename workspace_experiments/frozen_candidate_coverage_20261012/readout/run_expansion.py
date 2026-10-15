"""Expand frozen integration/CV recipes; never select using query outcomes."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
OLD = WORK/'target_domain_gap_20261012'
spec = importlib.util.spec_from_file_location('frozen_previous_cv', OLD/'support_cv_selection/run.py')
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)
old = cv.old
ROOT = old.ROOT
TASK_ROOT = ROOT/'learned_frequency_residual_20261009/tasks'
CACHE = Path('C:/Users/25791/AppData/Local/Temp/SAR_frozen_candidate_coverage_20261012/readout')
ARMS = old.ARMS
STAGES = ('expert_prior', 'expert', 'full')
POLICIES = ARMS + ('cv_oa', 'cv_nll')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ship_groups(frame, idcol, shipcol):
    ships = frame[shipcol].astype(str).str.strip()
    known = ships.str.fullmatch(r'[0-9]{9}') & ships.ne('000000000')
    return np.where(known, 'ship:'+ships, 'sample:'+frame[idcol]).astype(str), known.to_numpy()


class FullContext:
    def __init__(self, folder):
        self.info = read(folder/'PREPARED.json')
        self.data = data = old.load_task(folder)
        self.ctx = ctx = old.ExpertContext(folder)
        ctx._prepare_source()
        np.testing.assert_array_equal(ctx.source_rows, data['source_rows'])
        self.source_labels = ctx.source_labels
        if ctx.dataset == 'self24ports':
            self.hs, self.ht = old.head_inputs(ctx, data)
        else:
            xs = old.previous._expert_inputs(ctx.raw[ctx.source_rows], ctx.extra, ctx.model)
            protocol = read(ctx.joint.parent/'protocol.json')
            target = Path(protocol['target'])
            if ctx.dataset == 'f0921':
                raw = np.load(target/'paired_r50.npy', mmap_mode='r')
            else:
                with np.load(target/'features.npz') as z:
                    raw = z['paired_r50']
            extra = np.load(ctx.joint/'target_extra.npy', mmap_mode='r')
            if ctx.dataset == 'oss2':
                # Preserve the previous integration's FP32 PCA batch layout.
                xt = old.previous._expert_inputs(raw, extra, ctx.model)
            else:
                xt = np.concatenate([old.previous._expert_inputs(raw[lo:lo+2048], extra[lo:lo+2048], ctx.model)
                                     for lo in range(0, len(raw), 2048)])
            self.hs = np.c_[xs.astype(float), np.ones(len(xs))]
            self.ht = np.c_[xt.astype(float), np.ones(len(xt))]
        self.head = None
        self.refdir = ROOT/'backbone_ablation_20261010/targets'/ctx.dataset/ctx.task
        if ctx.dataset != 'self24ports':
            self.head = old.load_npz(self.refdir/'coefficients.npz')
        self.archived = old.load_npz(self.refdir/'predictions.npz')
        for key in ('rows', 'support'):
            np.testing.assert_array_equal(self.archived[key], data[key])
        if ctx.dataset == 'self24ports':
            frame = pd.read_csv(old.SOURCE/'manifest.csv', usecols=['instance_id', 'mmsi'],
                                dtype=str, keep_default_na=False).iloc[data['rows']]
            idcol, shipcol = 'instance_id', 'mmsi'
        else:
            shipcol = 'mmsi' if ctx.dataset == 'f0921' else 'mmsi_or_ship_id'
            frame = pd.read_csv(ROOT/old.DATASETS[ctx.dataset][0]/'target_manifest.csv',
                                usecols=['sample_id', shipcol], dtype=str, keep_default_na=False).iloc[data['rows']]
            idcol = 'sample_id'
        np.testing.assert_array_equal(frame[idcol].to_numpy(str), data['ids'])
        self.groups, self.known = ship_groups(frame, idcol, shipcol)

    def components(self, support):
        data = self.data
        pe, ke = self.ctx.target_components(data, support)
        if self.head is not None:
            pe = (self.ht[:, :-1]@self.head['dual_coef'] + self.head['dual_intercept']).astype(np.float32).astype(float)
        kb = old.base_cross(data, slice(None), support, self.info['eta'])
        return pe, ke, data['prior'], kb

    def reference(self, draw):
        z = self.archived
        modes = list(z['modes'])
        return dict(full_predictions=z['predictions'][draw, modes.index('dual_refit_late')],
                    expert_predictions=z['predictions'][draw, modes.index('dual_refit_expert')],
                    base_predictions=z['predictions'][draw, modes.index('base')])


def tasks():
    folders = sorted((TASK_ROOT/'self24ports').glob('*/PREPARED.json'))
    if len(folders) != 23:
        raise ValueError('Expected 23 eligible historical ports')
    return [p.parent for p in folders] + [TASK_ROOT/'oss2', TASK_ROOT/'f0921']


def internal_selection(stats, h, pe, ke, pb, kb, y, groups, frozen_folds=None):
    oof = np.full((len(cv.SEEDS), len(ARMS), len(y), 8), np.nan)
    fold_ids = np.full((len(cv.SEEDS), len(y)), -1, int)
    inventory = []
    for repeat, seed in enumerate(cv.SEEDS):
        with warnings.catch_warnings(record=True) as notes:
            parts = cv.folds(y, groups, seed) if frozen_folds is None else [
                (np.flatnonzero(frozen_folds[repeat]!=f), np.flatnonzero(frozen_folds[repeat]==f))
                for f in range(4)]
        seen = np.zeros(len(y), int)
        for fold, (train, val) in enumerate(parts):
            if set(groups[train]) & set(groups[val]):
                raise ValueError('Support vessel straddles folds')
            prob = cv.fold_predict(stats, h, pe, ke, pb, kb, train, y[train])
            oof[repeat][:, val] = prob[:, val]
            fold_ids[repeat, val] = fold
            seen[val] += 1
            missing = np.setdiff1d(np.unique(y[val]), np.unique(y[train]))
            inventory.append(dict(repeat=repeat, seed=seed, fold=fold,
                train_n=len(train), val_n=len(val), missing_classes=missing.tolist(),
                missing_class_validation_rows=int(np.isin(y[val], missing).sum()),
                warnings='; '.join(str(n.message) for n in notes)))
        np.testing.assert_array_equal(seen, np.ones(len(y), int))
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete OOF predictions')
    pooled = oof.transpose(1, 0, 2, 3).reshape(len(ARMS), -1, 8)
    yy = np.tile(y, len(cv.SEEDS))
    rows = [dict(arm=a, **old.metric(yy, pooled[i])) for i, a in enumerate(ARMS)]
    chosen = dict(cv_oa=cv.pick(rows), cv_nll=cv.pick(rows, 'nll'))
    return chosen, rows, oof, fold_ids, inventory


def freeze_task(ctx):
    started = time.perf_counter()
    info, data = ctx.info, ctx.data
    task, dataset = info['task'], info['dataset']
    dst = OUT/'tasks'/dataset/task
    dst.mkdir(parents=True, exist_ok=True)
    if (dst/'FROZEN.json').exists():
        print('REUSE', task, flush=True)
        return read(dst/'FROZEN.json')
    stats = old.source_statistics(ctx.hs, ctx.source_labels)
    n = len(data['ids'])
    cache = CACHE/dataset/task
    cache.mkdir(parents=True, exist_ok=True)
    arrays = {s: np.lib.format.open_memmap(cache/(s+'.npy'), mode='w+', dtype=np.float32,
                shape=(5, len(ARMS), n, 8)) for s in STAGES}
    rows, choices, folds, support_inventory = [], [], [], []
    replays = []
    split_plan = old.load_npz(dst/'CV_FOLDS.npz') if (dst/'CV_FOLDS.npz').exists() else None
    for draw, support in enumerate(data['support']):
        yy = data['support_labels'][draw]
        if len(support) != 64 or len(np.unique(support)) != 64 or data['eval_masks'][draw, support].any():
            raise ValueError('Historical support/query protocol changed')
        pe, ke, pb, kb = ctx.components(support)
        plan = None
        if split_plan is not None:
            np.testing.assert_array_equal(split_plan['support'][draw], support)
            np.testing.assert_array_equal(split_plan['groups'][draw], ctx.groups[support])
            np.testing.assert_array_equal(split_plan['labels'][draw], yy)
            plan = split_plan['fold_ids'][draw]
        result, scores, oof, fold_ids, fold_rows = internal_selection(stats, ctx.ht[support],
            pe[support], ke[support], pb[support], kb[support], yy, ctx.groups[support], plan)
        choices.append(dict(draw=draw, **result))
        rows.extend(dict(draw=draw, **r) for r in scores)
        folds.extend(dict(draw=draw, **r) for r in fold_rows)
        support_inventory.append(dict(draw=draw, classes=len(np.unique(yy)),
            counts=np.bincount(yy, minlength=8).tolist(),
            known_ships=len(np.unique(ctx.groups[support][ctx.known[support]])),
            unknown_rows=int((~ctx.known[support]).sum())))
        np.savez_compressed(dst/f'oof_{draw}.npz', oof=oof, fold_ids=fold_ids, y=yy,
                            support=support, support_ids=data['ids'][support], groups=ctx.groups[support])
        # Only legal source/support labels have been used at this point.
        old.write(dst/'SELECTION_FROZEN.json', dict(choices=choices, seeds=cv.SEEDS,
                                                   primary='cv_oa', query_labels_used=False))
        delta = old.increment(stats, ctx.ht[support], yy, ctx.ht)
        expert = old.integrate(pe, ke, support, yy, delta)
        base = pb + kb@old.coefficients(kb[support], pb[support], yy, 8.)
        bp = softmax(base, axis=1)
        ep = softmax(expert, axis=-1)
        full = .5*(bp[None] + ep)
        prior = softmax(np.stack((pe, pe+delta, pe+.5*delta, pe+delta)), axis=-1)
        reference = ctx.reference(draw)
        for prob, name in ((full[0], 'full'), (ep[0], 'expert'), (bp, 'base')):
            np.testing.assert_array_equal(prob.argmax(1), reference[name+'_predictions'], err_msg=f'{task}/{draw}/{name}')
        replay = dict(draw=draw, original_classes_exact=True)
        for name, values in (('full', full), ('expert', ep)):
            refkey = name+'_probabilities'
            if refkey in reference:
                np.testing.assert_allclose(values[0], reference[refkey], atol=3e-7, rtol=1e-7)
                replay[name+'_prob_error'] = float(np.max(np.abs(values[0]-reference[refkey])))
        previous = OLD/'residual_integration/target'/task/'predictions.npz'
        if previous.exists() and dataset != 'oss1':
            with np.load(previous) as z:
                np.testing.assert_array_equal(z['support'][draw], support)
                np.testing.assert_array_equal(full.argmax(-1), z['full'][draw].argmax(-1))
                np.testing.assert_allclose(full, z['full'][draw], atol=3e-7, rtol=1e-7)
                replay['all_old_arms_error'] = float(np.max(np.abs(full-z['full'][draw])))
            old_oof = OLD/'support_cv_selection/oof'/task/f'draw_{draw}.npz'
            with np.load(old_oof) as z:
                np.testing.assert_array_equal(fold_ids, z['fold_ids'])
                np.testing.assert_array_equal(oof.argmax(-1), z['oof'].argmax(-1))
                np.testing.assert_allclose(oof, z['oof'], atol=3e-7, rtol=1e-7)
                replay['old_oof_error'] = float(np.max(np.abs(oof-z['oof'])))
            past_choices = read(OLD/'support_cv_selection/SELECTION_FROZEN.json')['selected']
            past = next(r for r in past_choices if r['task']==task and r['draw']==draw)
            if any(result[k]!=past[k] for k in ('cv_oa','cv_nll')):
                raise ValueError('Expanded implementation changed previous CV selection')
        for name, values in zip(STAGES, (prior, ep, full)):
            arrays[name][draw] = values
            # Precision used by downstream metrics must not change top-1 outcomes.
            np.testing.assert_array_equal(arrays[name][draw].argmax(-1), values.argmax(-1))
            arrays[name].flush()
        replays.append(replay)
        print('FROZEN DRAW', task, draw, result, flush=True)
    np.savez_compressed(dst/'inventory.npz', rows=data['rows'], ids=data['ids'], support=data['support'],
        support_labels=data['support_labels'], eval_masks=data['eval_masks'], common_eval=data['eval_masks'].all(0))
    pd.DataFrame(rows).to_csv(dst/'cv_metrics.csv', index=False)
    pd.DataFrame(folds).to_csv(dst/'fold_inventory.csv', index=False)
    pd.DataFrame(support_inventory).to_csv(dst/'support_inventory.csv', index=False)
    import sklearn
    receipt = dict(task=task, dataset=dataset, operator=info.get('operator', 'full_frequency'),
        n=n, source_n=len(ctx.hs), head_width=ctx.ht.shape[1]-1, arms=ARMS,
        cache=str(cache), seconds=time.perf_counter()-started, replays=replays,
        neural_training=False, query_labels_used=False, numpy_version=np.__version__,
        sklearn_version=sklearn.__version__, predefined_folds=split_plan is not None)
    old.write(dst/'FROZEN.json', receipt)
    del arrays
    print('TASK COMPLETE', task, receipt['seconds'], flush=True)
    return receipt


def freeze(selected=None, with_oss1=True):
    pending = tasks()
    if selected:
        pending = [p for p in pending if p.name in selected]
    for folder in pending:
        info = read(folder/'PREPARED.json')
        if (OUT/'tasks'/info['dataset']/info['task']/'FROZEN.json').exists():
            print('REUSE', info['task'], flush=True)
            continue
        ctx = FullContext(folder)
        freeze_task(ctx)
        del ctx
        gc.collect()
    if with_oss1 and (selected is None or 'oss1' in selected):
        sys.path.insert(0, str(OUT))
        from oss1_context import OSS1Context
        if not (OUT/'tasks/oss1/oss1/FROZEN.json').exists():
            ctx = OSS1Context()
            freeze_task(ctx)
            del ctx
            gc.collect()
    records = [read(p) for p in sorted((OUT/'tasks').glob('*/*/FROZEN.json'))]
    old.write(OUT/'PREDICTIONS_FROZEN.json', dict(tasks=records, arms=ARMS,
        all_planned_complete=len(records)==26, target_metrics_read_by_this_runner=False))


def labels_for(info, data):
    dataset = info['dataset']
    if dataset == 'self24ports':
        frame = pd.read_csv(old.SOURCE/'manifest.csv', usecols=['instance_id', 'class_id']).iloc[data['rows']]
        idcol = 'instance_id'
    else:
        frame = pd.read_csv(ROOT/old.DATASETS[dataset][0]/'target_manifest.csv',
                            usecols=['sample_id', 'class_id']).iloc[data['rows']]
        idcol = 'sample_id'
    np.testing.assert_array_equal(frame[idcol].to_numpy(str), data['ids'])
    y = frame.class_id.to_numpy(int)
    np.testing.assert_array_equal(y[data['support']], data['support_labels'])
    return y


def measures(y, prob):
    p = prob.astype(float)
    classes = np.unique(y)
    correct = p.argmax(1)==y
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.))
    brier = np.sum((p-np.eye(8)[y])**2, axis=1)
    return dict(oa=100*correct.mean(), ba=100*np.mean([correct[y==c].mean() for c in classes]),
        nll=nll.mean(), brier=brier.mean(), macro_nll=np.mean([nll[y==c].mean() for c in classes]),
        macro_brier=np.mean([brier[y==c].mean() for c in classes]))


def evaluate():
    frozen = read(OUT/'PREDICTIONS_FROZEN.json')
    if not frozen['all_planned_complete']:
        raise ValueError('Freeze all 26 planned tasks before new outcome readout')
    records, class_rows, inventory = [], [], []
    for info in frozen['tasks']:
        dataset, task = info['dataset'], info['task']
        dst = OUT/'tasks'/dataset/task
        data = old.load_npz(dst/'inventory.npz')
        y = labels_for(info, data)
        choices = {r['draw']:r for r in read(dst/'SELECTION_FROZEN.json')['choices']}
        for stage in STAGES:
            probs = np.load(Path(info['cache'])/(stage+'.npy'), mmap_mode='r')
            for draw in range(5):
                chosen = {a:a for a in ARMS}
                chosen.update({k:choices[draw][k] for k in ('cv_oa', 'cv_nll')})
                for scope, mask in (('original', data['eval_masks'][draw]), ('common', data['common_eval'])):
                    yy = y[mask]
                    if not len(yy):
                        continue
                    basep = probs[draw, 0, mask]
                    base = measures(yy, basep)
                    before = basep.argmax(1)
                    for policy, arm in chosen.items():
                        p = probs[draw, ARMS.index(arm), mask]
                        mm = measures(yy, p)
                        after = p.argmax(1)
                        rescue = (before!=yy)&(after==yy)
                        harm = (before==yy)&(after!=yy)
                        meta = dict(dataset=dataset, task=task, operator=info['operator'],
                                    draw=draw, stage=stage, scope=scope, policy=policy, arm=arm)
                        records.append(dict(**meta, n=len(yy), classes=len(np.unique(yy)), **mm,
                            **{'delta_'+k:mm[k]-base[k] for k in mm},
                            rescues=int(rescue.sum()), harms=int(harm.sum())))
                        if stage=='full':
                            contributions = []
                            for c in np.unique(yy):
                                ix = yy==c
                                contribution = 100*(rescue[ix].sum()-harm[ix].sum())/ix.sum()
                                contributions.append(contribution)
                                class_rows.append(dict(**meta, class_id=int(c), n=int(ix.sum()),
                                    recall=100*(after[ix]==c).mean(), recall_delta=contribution,
                                    rescues=int(rescue[ix].sum()), harms=int(harm[ix].sum())))
                            np.testing.assert_allclose(np.mean(contributions), mm['ba']-base['ba'], atol=1e-10)
        inventory.append(info)
        print('EVALUATED', task, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(class_rows).to_csv(OUT/'class_recalls.csv', index=False)
    keys = ['dataset','task','operator','stage','scope','policy']
    cols = ['oa','ba','nll','brier','macro_nll','macro_brier','delta_oa','delta_ba',
            'delta_nll','delta_brier','delta_macro_nll','delta_macro_brier','rescues','harms']
    summary = frame.groupby(keys, sort=False)[cols].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    variation = frame.groupby(keys, sort=False).agg(delta_oa_sd=('delta_oa','std'),
        delta_ba_sd=('delta_ba','std'), ba_positive=('delta_ba',lambda s:int((s>1e-10).sum())),
        ba_negative=('delta_ba',lambda s:int((s < -1e-10).sum()))).reset_index()
    variation.to_csv(OUT/'support_variation.csv', index=False)
    aggregate = summary.groupby(['dataset','operator','stage','scope','policy'], sort=False)[cols].mean().reset_index()
    aggregate.to_csv(OUT/'aggregate.csv', index=False)
    old.write(OUT/'COMPLETE.json', dict(tasks=len(inventory), full_tasks=25, compact_tasks=1,
        support_sets=130, folds=1560, neural_training=False, main_model_unchanged=True))
    print(aggregate.query("stage=='full' and scope=='original'")[['dataset','policy','oa','ba','delta_oa','delta_ba']].to_string(index=False), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['freeze','evaluate'])
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--skip-oss1', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.phase=='freeze':
            freeze(args.tasks, not args.skip_oss1)
        else:
            evaluate()
