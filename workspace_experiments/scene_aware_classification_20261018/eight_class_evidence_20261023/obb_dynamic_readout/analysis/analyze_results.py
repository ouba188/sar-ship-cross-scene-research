"""Independent paired accounting; does not select or modify trained models."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCORE = [f's{i}' for i in range(8)]
BASE = [f'b{i}' for i in range(8)]


def assess(y, score):
    pred = score.argmax(1)
    correct = pred == y
    nll = logsumexp(score, axis=1) - score[np.arange(len(y)), y]
    p = softmax(score, axis=1)
    brier = ((p - np.eye(8)[y]) ** 2).sum(1)
    present = np.unique(y)
    return dict(n=len(y), OA=100 * correct.mean(),
                BA=100 * np.mean([correct[y == c].mean() for c in present]),
                classes=len(present), NLL=nll.mean(), Brier=brier.mean())


def paired(y, left, right):
    la, ra = assess(y, left), assess(y, right)
    lc, rc = left.argmax(1) == y, right.argmax(1) == y
    return dict(**{f'delta_{k}': la[k] - ra[k] for k in ('OA', 'BA', 'NLL', 'Brier')},
                rescue=int((lc & ~rc).sum()), harm=int((~lc & rc).sum()),
                net=int(lc.sum() - rc.sum()), n=len(y))


def clustered(y, delta, group, seed, draws=2000):
    codes, levels = pd.factorize(group, sort=True)
    counts = np.zeros((len(levels), 8), float)
    totals = np.zeros_like(counts)
    np.add.at(counts, (codes, y), 1)
    np.add.at(totals, (codes, y), delta)
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(draws):
        pick = rng.integers(0, len(levels), len(levels))
        n, d = counts[pick].sum(0), totals[pick].sum(0)
        if (n == 0).any():
            continue
        records.append([100 * d.sum() / n.sum(), 100 * (d / n).mean()])
    values = np.asarray(records)
    lo, hi = np.quantile(values, [.025, .975], axis=0)
    return dict(groups=len(levels), valid_draws=len(values), attempted_draws=draws,
                OA_lo=lo[0], OA_hi=hi[0], BA_lo=lo[1], BA_hi=hi[1])


def main(stage):
    f = pd.read_csv(ROOT / f'{stage}_predictions.csv', dtype={'mmsi': str},
                    float_precision='round_trip')
    f = f.loc[f.role.eq('test' if stage == 'test' else 'validation')].copy()
    assert not f.duplicated(['role', 'arm', 'seed', 'fold', 'instance_id']).any()
    assert set(f.arm) == {'global', 'fixed', 'dynamic', 'merged'}
    assert set(f.seed) == {17, 29, 43}
    assert np.isfinite(f[SCORE + BASE].to_numpy()).all()
    metrics, contrasts, byclass, byport, intervals, cases, checks = [], [], [], [], [], [], []
    oldpath = ROOT.parent / 'calibrated_signed_transfer/polarization_readout/joint_distribution_readout/matched_attention_readout' / f'{stage}_predictions.csv'
    old = pd.read_csv(oldpath, usecols=['arm', 'role', 'seed', 'fold', 'instance_id'] + SCORE,
                      float_precision='round_trip')
    old = old.loc[old.arm.eq('two_sided_attention') & old.role.eq(f.role.iloc[0])]
    old = old.set_index(['seed', 'fold', 'instance_id'])
    for seed, part in f.groupby('seed'):
        arms = {a: g.sort_values(['fold', 'instance_id']).reset_index(drop=True)
                for a, g in part.groupby('arm')}
        ref = arms['dynamic']
        y = ref.class_id.to_numpy(dtype=int)
        assert len(np.unique(y)) == 8
        if stage == 'test':
            assert len(ref) == 4744 and ref.instance_id.is_unique
        base = ref[BASE].to_numpy()
        keys = pd.MultiIndex.from_frame(ref[['seed', 'fold', 'instance_id']])
        replay = old.loc[keys, SCORE].to_numpy()
        err = float(abs(replay - base).max())
        assert err < 1e-6, err
        checks.append(dict(seed=int(seed), n=len(y), old_attention_error=err))
        scores = {'baseline': base}
        for arm, g in arms.items():
            pd.testing.assert_frame_equal(g[['fold', 'instance_id', 'class_id']], ref[['fold', 'instance_id', 'class_id']])
            np.testing.assert_array_equal(g[BASE].to_numpy(), base)
            scores[arm] = g[SCORE].to_numpy()
        for arm, score in scores.items():
            metrics.append(dict(seed=seed, arm=arm, **assess(y, score)))
        for left, right in [('global', 'baseline'), ('fixed', 'baseline'), ('dynamic', 'baseline'),
                            ('merged', 'baseline'), ('dynamic', 'fixed'), ('dynamic', 'global'),
                            ('dynamic', 'merged')]:
            a, b = scores[left], scores[right]
            meta = dict(seed=seed, left=left, right=right)
            contrasts.append(dict(**meta, **paired(y, a, b)))
            for c in range(8):
                take = y == c
                byclass.append(dict(**meta, class_id=c,
                    class_name=str(ref.loc[take, 'class_name'].iloc[0]), **paired(y[take], a[take], b[take])))
            for port, idx in ref.groupby('port').groups.items():
                ix = np.asarray(idx)
                byport.append(dict(**meta, port=port, classes=len(np.unique(y[ix])),
                                   **paired(y[ix], a[ix], b[ix])))
            if stage == 'test' and left == 'dynamic':
                delta = (a.argmax(1) == y).astype(int) - (b.argmax(1) == y).astype(int)
                passes = ref['product'].map(lambda s: '_'.join([s.split('_')[0], s.split('_')[6], s.split('_')[7]]))
                for groupname, groups in [('mmsi', ref.mmsi), ('pass_group', passes)]:
                    intervals.append(dict(**meta, cluster=groupname,
                        **clustered(y, delta, groups, 20261023 + int(seed))))
            if int(seed) == 17 and left == 'dynamic' and right in ('fixed', 'baseline'):
                ac, bc = a.argmax(1) == y, b.argmax(1) == y
                for name, flag in [('rescued', ac & ~bc), ('harmed', ~ac & bc),
                                   ('correct', ac & bc), ('wrong', ~ac & ~bc)]:
                    candidates = ref.loc[flag].sort_values('obs_id')
                    if not len(candidates):
                        continue
                    row = candidates.iloc[0]
                    i = int(row.name)
                    left_pred, right_pred = int(a[i].argmax()), int(b[i].argmax())
                    rival = right_pred if right_pred != y[i] else left_pred
                    if rival == y[i]:
                        rival = int(np.argmax(np.where(np.arange(8) == y[i], -np.inf, a[i])))
                    cases.append(dict(**meta, outcome=name, selection='lowest obs_id in available group',
                        **{k: row[k] for k in ('fold', 'instance_id', 'obs_id', 'mmsi', 'class_id', 'class_name', 'port', 'product')},
                        rival=rival, pred_left=left_pred, pred_right=right_pred,
                        margin_left=float(a[i, y[i]] - a[i, rival]),
                        margin_right=float(b[i, y[i]] - b[i, rival]),
                        top1_margin_left=float(a[i, y[i]] - np.max(np.delete(a[i], y[i]))),
                        top1_margin_right=float(b[i, y[i]] - np.max(np.delete(b[i], y[i]))),
                        group_n=int(flag.sum())))
    for name, data in [('metrics', metrics), ('contrasts', contrasts), ('by_class', byclass),
                       ('by_port', byport), ('intervals', intervals), ('cases', cases), ('checks', checks)]:
        pd.DataFrame(data).to_csv(HERE / f'{stage}_{name}.csv', index=False)
    m = pd.DataFrame(metrics)
    summary = m.groupby('arm')[['OA', 'BA', 'NLL', 'Brier']].agg(['mean', 'std'])
    summary.to_csv(HERE / f'{stage}_summary.csv')
    (HERE / f'{stage}_receipt.json').write_text(json.dumps(dict(stage=stage,
        rows=len(f), arms=4, seeds=3, new_fits=0, baseline_replay_max_error=max(c['old_attention_error'] for c in checks),
        bootstrap_note='Separate per-seed vessel and pass marginal cluster resampling, not independent seed-expanded observations.'), indent=2), encoding='utf-8')
    print(summary.to_string())
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['source', 'test'])
    main(parser.parse_args().stage)
