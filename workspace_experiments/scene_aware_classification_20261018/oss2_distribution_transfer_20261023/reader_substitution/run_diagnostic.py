"""Change only source-validation local moment reader; no training or target eval."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import train_heads as original


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    table = pd.read_csv(ROOT / 'cohort.csv', dtype={'mmsi': str})
    with np.load(ROOT / 'moments.npz') as z:
        cache = {k: z[k] for k in z.files}
    names = list(cache['reader_names'])
    frames, per_head = [], []
    for fold in range(6):
        d, mean, moments, _, _ = original.load_fold(fold, table, cache)
        ix = d['validation']
        indices = [names.index(f'fold{fold}_inner{k}') for k in range(4)]
        replacement = cache['moments_pol'][indices][:, ix].mean(0)
        for k in range(4):
            with np.load(ROOT / 'readers/models' / f'fold{fold}_inner{k}.npz') as z:
                fit = z['fit']
            for field in ['mmsi', 'pass_group']:
                assert not set(table.iloc[fit][field]) & set(table.iloc[ix][field])
        for seed in [17, 29, 43]:
            with np.load(ROOT / 'calibrated' / f'fold{fold}_seed{seed}.npz') as z:
                base = z['base'][ix]
            for arm in ['local', 'additive']:
                path = ROOT / 'heads' / f'fold{fold}_{arm}_seed{seed}.pt'
                old, ck = original.predict(path, base, mean[ix], moments[ix], cache['valid'][ix])
                new, _ = original.predict(path, base, mean[ix], replacement, cache['valid'][ix])
                if ck['epoch'] == 0:
                    np.testing.assert_array_equal(old, base)
                    np.testing.assert_array_equal(new, base)
                xo, _, _, _ = original.descriptor(arm, base, mean[ix], moments[ix], cache['valid'][ix], ck['mu'], ck['scale'])
                xn, _, _, _ = original.descriptor(arm, base, mean[ix], replacement, cache['valid'][ix], ck['mu'], ck['scale'])
                y = table.iloc[ix].class_id.to_numpy()
                a = original.parent.assessment(y, old)
                b = original.parent.assessment(y, new)
                per_head.append(dict(fold=fold, seed=seed, arm=arm, epoch=ck['epoch'],
                    descriptor_delta_rms=float((xn-xo).square().mean().sqrt()),
                    max_score_change=float(np.max(abs(new-old))),
                    **{f'old_{k}': a[k] for k in ['OA', 'BA', 'NLL', 'balanced_NLL']},
                    **{f'new_{k}': b[k] for k in ['OA', 'BA', 'NLL', 'balanced_NLL']}))
                g = table.iloc[ix][['sample_id', 'class_id', 'class_name', 'mmsi', 'pass_group', 'chip_path_vv', 'chip_path_vh']].copy()
                g['fold'], g['seed'], g['arm'], g['epoch'] = fold, seed, arm, ck['epoch']
                g['old_prediction'], g['new_prediction'] = old.argmax(1), new.argmax(1)
                g['net'] = g.new_prediction.eq(g.class_id).astype(int) - g.old_prediction.eq(g.class_id).astype(int)
                g['outcome'] = np.select([g.net.eq(1),g.net.eq(-1),g.new_prediction.eq(g.class_id)], ['rescued','harmed','CC'], default='WW')
                for c in range(8):
                    g[f'b{c}'], g[f'o{c}'], g[f's{c}'] = base[:,c],old[:,c],new[:,c]
                frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    # Replay must match the original run before interpreting a substitution effect.
    old = pd.read_csv(ROOT / 'predictions.csv')
    old = old[old.role.eq('validation') & old.arm.isin(['local','additive'])]
    joined = out.merge(old, left_on=['sample_id','fold','seed','arm'],
                       right_on=['instance_id','fold','seed','arm'], suffixes=('', '_record'), validate='one_to_one')
    assert len(joined) == len(out)
    replay = max(float(np.max(abs(joined[f'o{c}']-joined[f's{c}_record']))) for c in range(8))
    assert replay < 1e-6
    out.to_csv(HERE / 'predictions.csv', index=False)
    pd.DataFrame(per_head).to_csv(HERE / 'per_head.csv', index=False)
    pooled, classes = [], []
    for (arm,seed), g in out.groupby(['arm','seed']):
        for label, sub in [('all',g)] + [(str(c),v) for c,v in g.groupby('class_id')]:
            y = sub.class_id.to_numpy()
            before = original.parent.assessment(y,sub[[f'o{c}' for c in range(8)]].to_numpy())
            after = original.parent.assessment(y,sub[[f's{c}' for c in range(8)]].to_numpy())
            row = dict(arm=arm,seed=seed,class_id=label,n=len(sub),net=int(sub.net.sum()),
                       rescued=int(sub.net.eq(1).sum()),harmed=int(sub.net.eq(-1).sum()),
                       **{f'delta_{k}':after[k]-before[k] for k in ['OA','BA','NLL','balanced_NLL']})
            (pooled if label=='all' else classes).append(row)
    pd.DataFrame(pooled).to_csv(HERE / 'pooled.csv', index=False)
    pd.DataFrame(classes).to_csv(HERE / 'classes.csv', index=False)
    cases = []
    for arm in ['local','additive']:
        for outcome in ['rescued','harmed','CC','WW']:
            hit = out[out.arm.eq(arm)&out.seed.eq(17)&out.outcome.eq(outcome)].sort_values(['sample_id','fold'])
            if len(hit): cases.append(hit.iloc[0])
    pd.DataFrame(cases).to_csv(HERE / 'review_cases.csv', index=False)
    receipt = dict(heads=36,trained_steps=0,target_test_evaluated=False,
                   validation_rows_per_arm_seed=int(len(out)/6),
                   replay_error=replay,seconds=time.perf_counter()-started)
    (HERE / 'summary.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled).to_string(index=False))
    print(json.dumps(receipt))


if __name__=='__main__':main()
