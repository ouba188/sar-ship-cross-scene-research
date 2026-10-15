"""Frozen teacher-quality replay: test minmax amplification without refitting."""

import json
import numpy as np
import pandas as pd
from check_translation_teacher_feasibility import BASE, OUT as CACHE, ARMS, SEEDS
from check_motion_teacher_feasibility import assess

ROOT = BASE/'teacher_reliability_gate_20261012'
OUT = ROOT/'weight_strength'


def load_bundle(role, seed):
    with np.load(CACHE/f'{role}_{seed}_all_history.npz') as f:
        predictions = {k: f[k] for k in f.files}
    meta = pd.read_csv(BASE/'observations.csv').iloc[predictions['obs_ids']].reset_index(drop=True)
    cells = pd.read_csv(CACHE/f'{role}_cells_{seed}.csv')
    with np.load(CACHE/f'{role}_quality_{seed}.npz') as f:
        values = {k: f[k] for k in f.files}
    members = [g.index.to_numpy() for _, g in meta.groupby(['mmsi', 'product'], sort=True)]
    assert len(cells) == len(members)
    for i, ix in enumerate(members):
        assert meta.iloc[ix[0]].mmsi == cells.iloc[i].mmsi
        assert meta.iloc[ix[0]]['product'] == cells.iloc[i]['product']
    return dict(meta=meta, cells=cells, members=members, values=values, predictions=predictions)


def local_weights(q, mode):
    gap = q-q.min(); span = gap.max()
    if mode == 'minmax':
        return 1+gap/span if span > 1e-12 else np.ones(len(q))
    if mode == 'absolute_gap':
        return 1+gap/(1+span)
    raise ValueError(mode)


def apply_quality(bundle, quality, mode):
    cells = bundle['cells']; cs = bundle['values']['scores']
    baseline = bundle['predictions']['equal']
    pred = baseline.copy(); active = np.zeros(len(pred), bool); detail = []
    for _, g in cells.groupby('mmsi', sort=True):
        ix = g.index.to_numpy()
        for focal in ix:
            donor = ix[ix != focal]
            known = cells.loc[donor, 'known'].to_numpy(bool)
            if known.sum() < 2:
                continue
            q = quality[donor[known]]
            assert np.isfinite(q).all()
            local = local_weights(q, mode)
            weight = np.ones(len(donor)); weight[known] = local/local.mean()
            ids = bundle['members'][focal]
            pred[ids] = weight@cs[donor]/len(donor)
            active[ids] = True
            detail.append(dict(mmsi=int(cells.loc[focal, 'mmsi']),
                focal_obs_id=int(bundle['predictions']['obs_ids'][ids[0]]),
                known_donors=int(known.sum()), span=float(np.ptp(q)),
                max_min_weight_ratio=float(local.max()/local.min())))
    np.testing.assert_array_equal(active, bundle['predictions']['eligible'])
    return pred, pd.DataFrame(detail)


def evaluate(bundle, pred, arm, role, seed):
    multi = bundle['predictions']['multiview']; mm = bundle['meta']
    y = mm.loc[multi].class_id.to_numpy()
    old = bundle['predictions']['equal'][multi].argmax(1); now = pred[multi].argmax(1)
    rescue = (old != y)&(now == y); harm = (old == y)&(now != y)
    ships = mm.loc[multi].mmsi.to_numpy()
    return dict(role=role, seed=seed, arm=arm, **assess(pred[multi], y, mm.loc[multi]),
        rescue=int(rescue.sum()), harm=int(harm.sum()), rescue_ships=len(np.unique(ships[rescue])),
        harm_ships=len(np.unique(ships[harm])))


def self_test():
    q = np.array([0., 1e-9])
    np.testing.assert_allclose(local_weights(q, 'minmax'), [1., 2.])
    assert np.ptp(local_weights(q, 'absolute_gap')) < 2e-9
    for mode in ('minmax', 'absolute_gap'):
        np.testing.assert_allclose(local_weights(np.array([4., 4.]), mode), [1., 1.])
        np.testing.assert_allclose(local_weights(np.array([.1, .3]), mode),
                                   local_weights(np.array([7.1, 7.3]), mode), atol=1e-12)
        # Availability mass is conserved after normalization.
        w = local_weights(np.array([.1, .3]), mode)
        full = np.r_[w/w.mean(), 1.]/3
        np.testing.assert_allclose(full.sum(), 1.)
        np.testing.assert_allclose(full[-1], 1/3)


def main():
    self_test()
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'PROTOCOL.json').write_text(json.dumps(dict(new_fits=0, neural_fits=0, target_evaluations=0,
        original='w=1+(q-min(q))/span for non-ties',
        comparison='w=1+(q-min(q))/(1+span), in native predicted class-margin units; fixed, no scale search',
        invariant='Same cached crossfit/full readers, all-history donors, availability mask and total known mass',
        purpose='Test whether removing weak-score amplification reduces harm while retaining benefit; not a calibrated confidence claim',
        boundary='Historical source fit/held teachers only, no student or causal claim')), encoding='utf-8')
    rows, events, spans = [], [], []
    for role in ('fit', 'held'):
        for seed in SEEDS:
            bundle = load_bundle(role, seed)
            mm = bundle['meta']; old = bundle['predictions']['equal']
            outputs = {'equal': old}
            rows.append(evaluate(bundle, old, 'equal', role, seed))
            for arm in ARMS:
                for mode in ('minmax', 'absolute_gap'):
                    pred, detail = apply_quality(bundle, bundle['values'][arm], mode)
                    if mode == 'minmax':
                        np.testing.assert_array_equal(pred, bundle['predictions'][arm+'_soft'])
                    name = arm+'_'+mode; outputs[name] = pred
                    rows.append(evaluate(bundle, pred, name, role, seed))
                    detail['role'] = role; detail['seed'] = seed; detail['arm'] = name
                    spans.append(detail)
                    changed = np.flatnonzero(bundle['predictions']['multiview'] & (pred.argmax(1) != old.argmax(1)))
                    for i in changed:
                        y = int(mm.iloc[i].class_id)
                        events.append(dict(role=role, seed=seed, arm=name,
                            obs_id=int(bundle['predictions']['obs_ids'][i]), mmsi=int(mm.iloc[i].mmsi),
                            class_id=y, before=int(old[i].argmax()), after=int(pred[i].argmax()),
                            rescue=bool(old[i].argmax() != y and pred[i].argmax() == y),
                            harm=bool(old[i].argmax() == y and pred[i].argmax() != y)))
            np.savez_compressed(OUT/f'{role}_{seed}.npz', obs_ids=bundle['predictions']['obs_ids'], **outputs)
    pd.DataFrame(rows).to_csv(OUT/'results.csv', index=False)
    pd.DataFrame(events).to_csv(OUT/'changed_predictions.csv', index=False)
    pd.concat(spans).to_csv(OUT/'weight_spans.csv', index=False)
    summary = pd.DataFrame(rows).groupby(['role', 'arm'])[['OA', 'BA', 'CE', 'rescue', 'harm', 'rescue_ships', 'harm_ships']].mean()
    summary.to_csv(OUT/'summary.csv')
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(completed=True, new_fits=0,
        cached_minmax_replay_exact=True, synthetic_checks_passed=True)), encoding='utf-8')
    print(summary.to_string())


if __name__ == '__main__':
    main()
