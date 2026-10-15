"""Frozen diagnostics: gradient leverage, motion readout, class/port rescue/harm."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
import torch
from knowledge_train_only import (SEEDS, KnowledgeStudent, load_upstream,
    make_schedule, batch_losses, arm_loss)
from relative_content_condition import load_data, BASE
from probe_knowledge_attributes import fit_ridge, predict_ridge, write_json
from scene_training import sample_weights

GEO = BASE/'geometry_train_teacher_20261011'
MOTION = BASE/'motion_knowledge_20261011/training'
PREVIOUS = BASE/'knowledge_train_only_20261011/training'
OUT = BASE/'non_size_knowledge_20261011'


def pair_delta_error(prediction, truth, meta):
    numerator, denominator, rows = [], [], []
    for ship, group in meta.reset_index(drop=True).groupby('mmsi'):
        cells = [g.index.to_numpy() for _, g in group.groupby('product')]
        for a in range(len(cells)):
            for b in range(a+1, len(cells)):
                i, j = cells[a], cells[b]
                dt = float(truth[j].mean()-truth[i].mean())
                dp = float(prediction[j].mean()-prediction[i].mean())
                rows.append(dict(mmsi=ship, class_id=int(group.iloc[0].class_id),
                                 error=(dt-dp)**2, zero_error=dt**2))
    if not rows:
        return dict(pairs=0, ships=0, skill=np.nan)
    frame = pd.DataFrame(rows)
    w = sample_weights(frame); w /= w.sum()
    den = float(w@frame.zero_error.to_numpy())
    return dict(pairs=len(frame), ships=int(frame.mmsi.nunique()),
                skill=100*(1-float(w@frame.error.to_numpy())/den) if den else np.nan)


def main():
    torch.set_num_threads(1)
    for folder in (GEO, GEO/'near_specificity', MOTION):
        if not (folder/'COMPLETE.json').exists():
            raise RuntimeError('All planned fits/evaluations must finish before diagnostics')
    OUT.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    del norm, fixed, h
    fit, held = split['factor_fit'], split['test_source']
    fm, hm = [meta.iloc[ix].reset_index(drop=True) for ix in (fit, held)]
    y = torch.tensor(meta.class_id.to_numpy(copy=True))
    prob = sample_weights(fm); prob /= prob.sum()
    with np.load(PREVIOUS/'evaluation_episodes.npz') as f:
        validation = [(f[f'validation_{j}_support'], f[f'validation_{j}_query']) for j in range(3)]
    dummy = dict(real=torch.zeros(len(fm), 2), control=torch.zeros(len(fm), 2),
                 valid=torch.zeros(len(fm), dtype=torch.bool), importance=torch.zeros(len(fm)))
    gradients = []
    for seed in SEEDS:
        up = load_upstream(BASE/'crossfit_evidence_20261011', seed, raw, meta, fit, validation)
        model = KnowledgeStudent(up['x'][fit], y[fit], prob)
        model.load_state_dict(torch.load(GEO/f'initial_{seed}.pt', weights_only=True))
        schedule = make_schedule(fm, prob, seed)
        with np.load(GEO/f'teachers_{seed}.npz') as f:
            teachers = {arm: torch.from_numpy(f[arm]) for arm in ('B', 'far', 'near', 'sham')}
        with np.load(GEO/'near_specificity'/f'teachers_{seed}.npz') as f:
            teachers['sham_near'] = torch.from_numpy(f['sham_near'])
        for step in range(8):
            values = {}
            for arm, teacher in teachers.items():
                losses = batch_losses(model, up['x'][fit], y[fit], up['kernel'][fit], teacher,
                    up['teacher_valid'], dummy, schedule, step)
                values[arm] = torch.autograd.grad(arm_loss(losses, 'B', 0.), model.frame)[0].detach()
            for arm, grad in values.items():
                gradients.append(dict(seed=seed, batch=step, arm=arm,
                    gradient_change_percent=float(100*(grad-values['B']).norm()/values['B'].norm())))
    pd.DataFrame(gradients).to_csv(OUT/'geometry_gradient_leverage.csv', index=False)

    with np.load(MOTION.parent/'attributes.npz') as f:
        attributes = {k: f[k] for k in f.files}
    with np.load(MOTION/'attribute_normalization.npz') as f:
        normal = {k: f[k] for k in f.files}
    masks = {role: attributes[role+'_sog_valid'] & (attributes[role+'_sog'] < 102.2) for role in ('fit', 'held')}
    mm = dict(fit=fm.loc[masks['fit']].reset_index(drop=True), held=hm.loc[masks['held']].reset_index(drop=True))
    target = {r: ((np.log1p(attributes[r+'_sog'][masks[r]])-normal['log_mean'][0])/normal['log_scale'][0])[:, None]
              for r in ('fit', 'held')}
    weights = {r: sample_weights(mm[r])/sample_weights(mm[r]).sum() for r in ('fit', 'held')}
    cm = (normal['class_log_means']-normal['log_mean'])/normal['log_scale']
    readers, delta_readers, coverage = [], [], []
    for role in ('fit', 'held'):
        table = mm[role].copy(); table['sog'] = attributes[role+'_sog'][masks[role]]
        for c, group in table.groupby('class_id'):
            coverage.append(dict(role=role, class_id=int(c), class_name=str(group.iloc[0].class_name),
                n=len(group), ships=int(group.mmsi.nunique()), zero=int((group.sog == 0).sum()),
                median=float(group.sog.median()), maximum=float(group.sog.max())))
    for seed in SEEDS:
        for arm, directory in [('B', PREVIOUS), ('C', MOTION), ('D', MOTION)]:
            for snapshot in ('selected', 'last'):
                with np.load(directory/f'{arm}_{seed}_{snapshot}.npz') as f:
                    features = dict(fit=f['z'][fit][masks['fit']].astype(float),
                                    held=f['z'][held][masks['held']].astype(float))
                    auxiliary = {r: f['attribute_prediction'][ix][masks[r]]
                                 for r, ix in [('fit', fit), ('held', held)]} if arm != 'B' else {}
                train = features['fit']; w = weights['fit']; cls = mm['fit'].class_id.to_numpy()
                xm = np.stack([np.average(train[cls == c], axis=0, weights=w[cls == c]) for c in range(8)])
                direct = fit_ridge(train, target['fit'], w)
                conditional = fit_ridge(train-xm[cls], target['fit']-cm[cls], w)
                for role in ('fit', 'held'):
                    cls = mm[role].class_id.to_numpy(); x = features[role]; t = target[role]
                    pred = dict(global_mean=np.zeros_like(t), class_mean=cm[cls],
                        sar=predict_ridge(direct, x), conditional=cm[cls]+predict_ridge(conditional, x-xm[cls]))
                    if auxiliary:
                        pred['trained_auxiliary'] = auxiliary[role]
                    for c in [-1]+list(range(8)):
                        chosen = np.ones(len(cls), bool) if c == -1 else cls == c
                        if not chosen.any():
                            continue
                        w = sample_weights(mm[role].loc[chosen]); w /= w.sum()
                        base = float(w@np.square(cm[cls][chosen]-t[chosen]).ravel())
                        for name, value in pred.items():
                            mse = float(w@np.square(value[chosen]-t[chosen]).ravel())
                            readers.append(dict(seed=seed, arm=arm, snapshot=snapshot, role=role,
                                class_id=c, reader=name, MSE=mse, skill_vs_class=100*(1-mse/base) if base else np.nan))
                    if role == 'held':
                        for name, value in pred.items():
                            delta_readers.append(dict(seed=seed, arm=arm, snapshot=snapshot, reader=name,
                                **pair_delta_error(value, t, mm[role])))
    pd.DataFrame(readers).to_csv(OUT/'motion_readability.csv', index=False)
    pd.DataFrame(delta_readers).to_csv(OUT/'motion_within_ship_delta.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT/'motion_class_coverage.csv', index=False)
    frames, source_frames, class_frames = [], [], []
    for label, folder in [('geometry', GEO), ('near_specificity', GEO/'near_specificity'), ('motion', MOTION)]:
        for file, sink in [('target_results.csv', frames), ('source_results.csv', source_frames), ('target_classes.csv', class_frames)]:
            f = pd.read_csv(folder/file); f['experiment'] = label; sink.append(f)
    target_table = pd.concat(frames, ignore_index=True)
    source_table = pd.concat(source_frames, ignore_index=True)
    class_table = pd.concat(class_frames, ignore_index=True)
    target_table.groupby(['experiment', 'snapshot', 'arm'])[['OA', 'BA', 'raw_OA', 'raw_BA']].mean().to_csv(OUT/'target_summary.csv')
    target_table.groupby(['experiment', 'snapshot', 'arm', 'seed'])[['OA', 'BA']].mean().to_csv(OUT/'target_seeds.csv')
    target_table.groupby(['experiment', 'snapshot', 'arm', 'port'])[['OA', 'BA']].mean().to_csv(OUT/'target_ports.csv')
    source_table.groupby(['experiment', 'snapshot', 'readout', 'arm'])[['OA', 'BA']].mean().to_csv(OUT/'source_summary.csv')
    class_table.groupby(['experiment', 'snapshot', 'arm', 'class_id'])[['recall', 'delta_recall']].mean().to_csv(OUT/'class_changes.csv')
    target_table.groupby(['experiment', 'snapshot', 'arm'])[['rescue', 'harm']].sum().to_csv(OUT/'rescue_harm_exposures.csv')
    write_json(OUT/'COMPLETE.json', dict(completed=True, new_fits=0, no_model_selection=True,
        conditional_reader_requires_true_class=True, event_counts_are_support_draw_exposures=True))
    r = pd.DataFrame(readers)
    print('MOTION READOUT', r[(r.class_id == -1)&(r.snapshot == 'selected')].groupby(['arm', 'role', 'reader'])[['MSE', 'skill_vs_class']].mean().to_string())
    d = pd.DataFrame(delta_readers)
    print('MOTION DELTA', d[d.snapshot == 'selected'].groupby(['arm', 'reader'])[['pairs', 'ships', 'skill']].mean().to_string())
    print('GEOMETRY GRADIENT', pd.DataFrame(gradients).groupby('arm').gradient_change_percent.agg(['mean', 'max']).to_string())


if __name__ == '__main__':
    main()
