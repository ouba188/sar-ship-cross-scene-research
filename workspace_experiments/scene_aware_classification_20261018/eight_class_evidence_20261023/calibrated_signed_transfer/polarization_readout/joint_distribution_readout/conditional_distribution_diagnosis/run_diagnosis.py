"""Frozen additive readout with source-only, label-free matched moment donors."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '2')
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_joint as run


def matching_inputs(base, mean):
    b = base - base.mean(1, keepdims=True)
    return b, np.concatenate([b, mean.reshape(len(mean), -1)], axis=1)


def predict(model, ck, b, mu, moment, valid):
    x, _, _ = run.inputs(b, mu, moment, valid, ck['mu'], ck['scale'])
    with torch.inference_mode():
        return b + model(x).numpy().astype(float)


def main():
    torch.set_num_threads(2)
    table = run.parent.load_table()
    cache = dict(np.load(run.parent.HERE / 'signed_features.npz'))
    valid = cache['valid'].astype(bool)
    saved = pd.read_csv(run.HERE / 'predictions.csv')
    saved = saved[saved.arm.eq('additive')].set_index(['fold', 'seed', 'role', 'instance_id'])
    rows, checks = [], []
    for fold in range(3):
        d, _, _ = run.parent.load_fold(fold, table, cache)
        means, om, moments, od = run.load_features(fold, table, cache, d)
        pool = d['validation']
        for oldseed, seed in run.strong.SEED_PAIRS:
            base, bt, _ = run.strong.calibrated(fold, oldseed, seed, table, d)
            ck, model = run.restore(run.HERE / 'models' / f'fold{fold}_additive_seed{seed}.pt')
            raw = matching_inputs(base, means)
            train = matching_inputs(bt[d['train']], om[d['train']])
            scales = [np.std(x, axis=0) for x in train]
            scales = [np.where(s < 1e-8, 1., s) for s in scales]
            for role in ['validation', 'test']:
                ix = d[role]
                own = predict(model, ck, base[ix], means[ix], moments[ix], valid[ix])
                key = pd.MultiIndex.from_arrays([[fold]*len(ix), [seed]*len(ix),
                    [role]*len(ix), table.iloc[ix].instance_id])
                reference = saved.loc[key, [f's{c}' for c in range(8)]].to_numpy()
                error = float(np.max(abs(own-reference)))
                assert error < 1e-6 and np.array_equal(own.argmax(1), reference.argmax(1))
                legal = ((valid[ix, None] == valid[pool][None]).all(2)
                    & (table.iloc[ix].mmsi.to_numpy()[:, None] != table.iloc[pool].mmsi.to_numpy()[None])
                    & (table.iloc[ix].pass_group.to_numpy()[:, None] != table.iloc[pool].pass_group.to_numpy()[None]))
                distance = [cdist(x[ix]/s, x[pool]/s, 'sqeuclidean')/x.shape[1]
                    for x, s in zip(raw, scales)]
                eligible = legal.any(1)
                for match, dist in zip(['B_only', 'B_and_mean'], distance):
                    nearest = np.where(legal, dist, np.inf).argmin(1)
                    donor = pool[nearest].copy()
                    donor[~eligible] = ix[~eligible]
                    borrowed = moments[donor].copy()
                    assert np.all(abs(borrowed[..., 1]) <= borrowed[..., 0]+1e-9)
                    score = predict(model, ck, base[ix], means[ix], borrowed, valid[ix])
                    part = table.iloc[ix][['instance_id', 'class_id', 'class_name', 'mmsi', 'pass_group', 'port']].copy()
                    part['donor_id'] = table.iloc[donor].instance_id.to_numpy()
                    part['donor_class'] = table.iloc[donor].class_id.to_numpy()
                    part['donor_mmsi'] = table.iloc[donor].mmsi.to_numpy()
                    part['donor_pass'] = table.iloc[donor].pass_group.to_numpy()
                    for k, v in dict(fold=fold, seed=seed, role=role, match=match, epoch=ck['epoch']).items():
                        part[k] = v
                    part['eligible'] = eligible
                    part['B_distance'] = distance[0][np.arange(len(ix)), nearest]
                    part['B_mean_distance'] = distance[1][np.arange(len(ix)), nearest]
                    part.loc[~eligible, ['B_distance', 'B_mean_distance']] = np.nan
                    part['own_prediction'], part['borrowed_prediction'] = own.argmax(1), score.argmax(1)
                    part['net_own_vs_borrowed'] = part.own_prediction.eq(part.class_id).astype(int)-part.borrowed_prediction.eq(part.class_id).astype(int)
                    part['outcome'] = np.select([part.net_own_vs_borrowed.eq(1), part.net_own_vs_borrowed.eq(-1),
                        part.own_prediction.eq(part.class_id)], ['rescued', 'harmed', 'CC'], default='WW')
                    for c in range(8):
                        part[f'b{c}'], part[f'own{c}'], part[f'borrowed{c}'] = base[ix,c], own[:,c], score[:,c]
                    assert not (part.loc[eligible].mmsi == part.loc[eligible].donor_mmsi).any()
                    assert not (part.loc[eligible].pass_group == part.loc[eligible].donor_pass).any()
                    rows.append(part)
                    checks.append(dict(fold=fold, seed=seed, role=role, match=match, rows=len(ix),
                        eligible=int(eligible.sum()), own_replay_error=error, mean_B_valid_preserved=True))
    p = pd.concat(rows, ignore_index=True)
    p.to_csv(HERE/'predictions.csv', index=False)
    metrics, classes = [], []
    for keys, g in p.groupby(['role', 'match', 'seed']):
        meta = dict(zip(['role', 'match', 'seed'], keys))
        y = g.class_id.to_numpy()
        a = run.parent.assessment(y, g[[f'own{c}' for c in range(8)]].to_numpy())
        b = run.parent.assessment(y, g[[f'borrowed{c}' for c in range(8)]].to_numpy())
        metrics.append(dict(**meta, n=len(g), eligible=int(g.eligible.sum()),
            net=int(g.net_own_vs_borrowed.sum()), own_rescues=int(g.net_own_vs_borrowed.eq(1).sum()),
            own_harms=int(g.net_own_vs_borrowed.eq(-1).sum()),
            **{f'own_{k}':v for k,v in a.items() if k!='n'},
            **{f'borrowed_{k}':v for k,v in b.items() if k!='n'},
            **{f'delta_{k}':a[k]-b[k] for k in ['OA','BA','NLL','Brier']},
            median_B_distance=g.B_distance.median(), median_full_distance=g.B_mean_distance.median()))
        for c,h in g.groupby('class_id'):
            classes.append(dict(**meta,class_id=c,n=len(h),net=int(h.net_own_vs_borrowed.sum()),
                same_donor_class_rate=h.loc[h.eligible].class_id.eq(h.loc[h.eligible].donor_class).mean()))
    pd.DataFrame(metrics).to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(HERE/'classes.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'checks.csv',index=False)
    (HERE/'summary.json').write_text(json.dumps(dict(new_training=0, prediction_rows=len(p),
        exact_condition_match=False, target_labels_used_for_donor_selection=False,
        max_replay_error=max(x['own_replay_error'] for x in checks)),indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics)[['role','match','seed','n','eligible','net','delta_OA','delta_BA']].to_string(index=False))


if __name__ == '__main__':
    main()
