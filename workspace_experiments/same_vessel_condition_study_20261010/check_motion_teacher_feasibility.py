"""No-training feasibility check: does source-only SOG select better teachers?"""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

from pathlib import Path
import numpy as np
import pandas as pd
from relative_content_condition import vessel_targets
from scene_training import sample_weights, metrics
from probe_knowledge_attributes import write_json

BASE = Path(__file__).resolve().parent
OUT = BASE/'motion_teacher_feasibility_20261011'
ARMS = ('equal', 'near', 'far', 'sham_near', 'sham_far', 'confidence',
        'nearest_only', 'farthest_only', 'label_oracle')


def candidates(scores, meta, sog, matched=False):
    _, baseline, multiview = vessel_targets(scores, meta)
    pred = {a: baseline.copy() for a in ARMS}
    eligible = np.zeros(len(meta), bool)
    edges = []
    rng = np.random.default_rng(51011)
    for ship, group in meta.groupby('mmsi', sort=True):
        assert group.class_id.nunique() == 1
        cells = []
        for product, g in group.groupby('product', sort=True):
            ix = g.index.to_numpy()
            speed = sog[ix]; speed = speed[np.isfinite(speed)]
            angle = g.incidence_deg.to_numpy(float)
            angle = angle[np.isfinite(angle)]
            cells.append(dict(ids=ix, product=product, score=scores[ix].mean(0),
                speed=np.log1p(speed).mean() if len(speed) else np.nan,
                angle=angle.mean() if len(angle) else np.nan, port=str(g.iloc[0].port)))
        if len(cells) < 3:
            continue
        state = np.array([c['speed'] for c in cells])
        shuffled = state.copy(); known = np.flatnonzero(np.isfinite(state))
        shuffled[known] = rng.permutation(state[known])
        for focal, cell in enumerate(cells):
            if not np.isfinite(state[focal]):
                continue
            donor = np.array([j for j in range(len(cells)) if j != focal])
            use = np.isfinite(state[donor])
            if matched:
                use &= np.array([cells[j]['port'] == cell['port'] and
                    np.isfinite(cells[j]['angle']) and np.isfinite(cell['angle']) and
                    abs(cells[j]['angle']-cell['angle']) <= 2 for j in donor])
            if use.sum() < 2 or np.ptp(state[donor[use]]) < 1e-12:
                continue
            eligible[cell['ids']] = True
            means = np.stack([cells[j]['score'] for j in donor])
            nknown = int(use.sum()); count = len(donor)
            truth = int(group.iloc[0].class_id)
            for arm in ARMS:
                a = shuffled if arm.startswith('sham') else state
                gap = np.abs(a[donor[use]]-a[focal])
                distance = np.minimum(gap/np.log(6), 1.)
                local = np.ones(nknown)
                if arm in ('near', 'sham_near'):
                    local = 2-distance
                elif arm in ('far', 'sham_far'):
                    local = 1+distance
                elif arm == 'confidence':
                    ordered = np.sort(means[use], axis=1)
                    margin = ordered[:, -1]-ordered[:, -2]
                    local = 1+margin/(1+margin)
                elif arm in ('nearest_only', 'farthest_only'):
                    selected = gap == (gap.min() if arm == 'nearest_only' else gap.max())
                    local = selected.astype(float)
                elif arm == 'label_oracle':
                    pool = (means[~use].sum(0)[None, :]+nknown*means[use])/count
                    pool = np.vstack([baseline[cell['ids'][0]], pool])
                    rivals = pool.copy(); rivals[:, truth] = -np.inf
                    best = np.argmax(pool[:, truth]-rivals.max(1))
                    pred[arm][cell['ids']] = pool[best]
                    continue
                weights = np.ones(count)
                weights[use] = local/local.mean()
                if arm != 'equal':
                    pred[arm][cell['ids']] = weights@means/count
            for j in donor[use]:
                donor_score = cells[j]['score']
                ordered = np.sort(donor_score)
                rival = donor_score.copy(); rival[truth] = -np.inf
                edges.append(dict(mmsi=int(ship), focal_product=cell['product'],
                    donor_product=cells[j]['product'], class_id=truth,
                    log_speed_gap=float(abs(state[j]-state[focal])),
                    same_port=cells[j]['port'] == cell['port'],
                    incidence_gap=float(abs(cells[j]['angle']-cell['angle'])),
                    confidence=float(ordered[-1]-ordered[-2]),
                    teacher_margin=float(donor_score[truth]-rival.max())))
    return pred, multiview, eligible, edges


def assess(pred, truth, meta):
    weights = sample_weights(meta); weights /= weights.sum()
    score = pred.astype(float)
    maximum = score.max(1)
    ce = maximum+np.log(np.exp(score-maximum[:, None]).sum(1))-score[np.arange(len(score)), truth]
    target = 2*np.eye(8)[truth]-1
    centered = score-score.mean(1, keepdims=True)
    target -= target.mean(1, keepdims=True)
    mse = ((centered-target)**2).mean(1)
    return dict(**metrics(score.argmax(1), truth), ship_balanced_accuracy=float(100*weights@(score.argmax(1) == truth)),
                CE=float(weights@ce), centered_target_MSE=float(weights@mse))


def self_test():
    meta = pd.DataFrame(dict(mmsi=[1]*4, product=['a', 'a', 'b', 'c'],
        incidence_deg=[35.]*4, port=['p']*4, class_id=[0]*4))
    scores = np.zeros((4, 8), np.float32); scores[:, 0] = [100, 100, -2, 4]
    sog = np.array([0., 0., 0., 10.])
    p, _, eligible, edges = candidates(scores, meta, sog)
    assert eligible[:2].all() and not eligible[3]
    assert p['near'][0, 0] < p['equal'][0, 0] < p['far'][0, 0]
    np.testing.assert_array_equal(p['label_oracle'][0], scores[3])
    assert all(e['focal_product'] != e['donor_product'] for e in edges)
    altered = scores.copy(); altered[:2] = -999
    q, _, _, _ = candidates(altered, meta, sog)
    for arm in ARMS:
        np.testing.assert_array_equal(p[arm][:2], q[arm][:2])
    unknown = sog.copy(); unknown[2] = np.nan
    _, _, eligible, _ = candidates(scores, meta, unknown)
    assert not eligible[:2].any()


def main():
    self_test()
    OUT.mkdir(exist_ok=False)
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(BASE/'motion_knowledge_20261011/attributes.npz') as f:
        attributes = {k: f[k] for k in f.files}
    write_json(OUT/'PROTOCOL.json', dict(
        neural_fits=0, target_data_used=False, seeds=[31415, 27182],
        fit='Cached native ship-fold OOF teacher scores on original source fit ships.',
        held='Cached FULL source teacher on disjoint held vessels; historically reused diagnostic cohort.',
        operators='Boundednear/far on log1p SOG gap/log6; confidence1+topgap/(1+topgap); hardselectors diagnostic, not tuning.',
        mass='Knowneligible donor total mass unchanged; all others retain equal mass. Focal wholeproduct excluded.',
        oracle='True-label best margin among baseline and selecting one eligible donor; diagnostic only, not deployable or a student-gain bound.',
        matched='Additional same-port and incidence difference<=2deg eligible pools; availability limitations reported.',
        limitation='Better teacher quality does not guarantee better student recognition; no new deployed model or causal identification.'))
    rows, coverage, edge_rows = [], [], []
    for role in ('fit', 'held'):
        ids = attributes[role+'_obs_ids']
        mm = meta.iloc[ids].reset_index(drop=True)
        sog = attributes[role+'_sog'].copy()
        sog[~attributes[role+'_sog_valid'] | (sog >= 102.2)] = np.nan
        for seed in (31415, 27182):
            if role == 'fit':
                with np.load(BASE/'geometry_train_teacher_20261011'/f'teachers_{seed}.npz') as f:
                    np.testing.assert_array_equal(f['fit_obs_ids'], ids)
                    scores = f['original_oof']
            else:
                with np.load(BASE/'crossfit_evidence_20261011'/f'full_{seed}_outputs.npz') as f:
                    assert not set(meta.iloc[f['fit_obs_ids']].mmsi) & set(mm.mmsi)
                    scores = f['prior'][ids]
            for matched in (False, True):
                pred, multiview, eligible, edges = candidates(scores, mm, sog, matched)
                pool = 'same_port_angle' if matched else 'all_history'
                coverage.append(dict(role=role, seed=seed, pool=pool, observations=len(mm),
                    multiview=int(multiview.sum()), speed_known=int(np.isfinite(sog).sum()),
                    eligible=int(eligible.sum()), ships=int(mm.loc[eligible].mmsi.nunique()),
                    classes=','.join(map(str, sorted(mm.loc[eligible].class_id.unique())))))
                edge_rows.extend(dict(role=role, seed=seed, pool=pool, **e) for e in edges)
                np.savez_compressed(OUT/f'{role}_{seed}_{pool}.npz', **pred, obs_ids=ids,
                                    multiview=multiview, eligible=eligible)
                for cohort, mask in [('multiview', multiview), ('eligible', eligible)]:
                    if not mask.any():
                        continue
                    truth = mm.loc[mask].class_id.to_numpy()
                    old = pred['equal'][mask].argmax(1)
                    for arm in ARMS:
                        score = pred[arm][mask]; current = score.argmax(1)
                        rows.append(dict(role=role, seed=seed, pool=pool, cohort=cohort, arm=arm,
                            **assess(score, truth, mm.loc[mask]),
                            rescue=int(((old != truth)&(current == truth)).sum()),
                            harm=int(((old == truth)&(current != truth)).sum())))
    pd.DataFrame(rows).to_csv(OUT/'results.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv', index=False)
    pd.DataFrame(edge_rows).to_csv(OUT/'teacher_edges.csv', index=False)
    summary = pd.DataFrame(rows).groupby(['role', 'pool', 'cohort', 'arm'])[
        ['OA', 'BA', 'ship_balanced_accuracy', 'CE', 'centered_target_MSE', 'rescue', 'harm']].mean()
    summary.to_csv(OUT/'summary.csv')
    write_json(OUT/'COMPLETE.json', dict(completed=True, neural_fits=0, targets_evaluated=0, synthetic_tests_passed=True))
    print(pd.DataFrame(coverage).drop(columns='seed').drop_duplicates().to_string(index=False))
    print(summary.to_string())


if __name__ == '__main__':
    main()
