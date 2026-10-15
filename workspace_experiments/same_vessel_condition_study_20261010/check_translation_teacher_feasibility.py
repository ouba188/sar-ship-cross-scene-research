"""Bounded source-only motion-direction ranking probe, not a student module."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from incidence_transport import pair_weights
from relative_content_condition import vessel_targets
from check_motion_teacher_feasibility import assess

BASE = Path(__file__).resolve().parent
OUT = BASE/'translation_teacher_feasibility_20261012'/'corrected'
ARMS = ('confidence', 'speed', 'direction', 'capacity')
COLS = dict(confidence=[0], speed=[0, 1, 2], direction=[0, 1, 2, 3, 4],
            capacity=[0, 1, 2, 5, 6])
SEEDS = (31415, 27182)


def motion_features(sog, cog, azimuth, valid):
    result = np.full((len(sog), 6), np.nan)
    zero = valid & (sog == 0)
    moving = valid & (sog > 0) & np.isfinite(cog) & np.isfinite(azimuth)
    result[zero] = 0.
    u = sog[moving]/(1+sog[moving])
    theta = np.deg2rad(cog[moving]-azimuth[moving])
    result[moving] = np.c_[u, u*u, u*u*np.cos(2*theta), u*u*np.sin(2*theta), u**3, u**4]
    return result


def make_cells(meta, scores, motion, fold):
    records, vectors, predictions, members = [], [], [], []
    for (ship, product), g in meta.groupby(['mmsi', 'product'], sort=True):
        ix = g.index.to_numpy()
        assert g.class_id.nunique() == 1
        assert len(np.unique(fold[ix])) == 1
        known = np.isfinite(motion[ix]).all(1)
        vector = motion[ix[known]].mean(0) if known.any() else np.full(6, np.nan)
        score = scores[ix].mean(0)
        ss = np.sort(score); gap = ss[-1]-ss[-2]
        vectors.append(np.r_[gap/(1+gap), vector])
        predictions.append(score); members.append(ix)
        records.append(dict(mmsi=ship, product=product, class_id=int(g.iloc[0].class_id),
            port=g.iloc[0].port, incidence_deg=float(g.incidence_deg.mean()),
            fold=int(fold[ix[0]]), known=bool(known.any()),
            known_rows=int(known.sum()), rows=len(ix)))
    return pd.DataFrame(records), np.array(vectors), np.array(predictions), members


def pairs_for(cells, allowed=None):
    mask = cells.known.to_numpy().copy()
    if allowed is not None:
        mask &= allowed
    pairs = []
    for _, g in cells.loc[mask].groupby('mmsi', sort=True):
        assert g.class_id.nunique() == 1
        pairs.extend(itertools.combinations(g.index, 2))
    return np.array(pairs, int).reshape(-1, 2)


def margins(scores, labels):
    rivals = scores.copy(); rivals[np.arange(len(scores)), labels] = -np.inf
    return scores[np.arange(len(scores)), labels]-rivals.max(1)


def fit_rank(cells, x, scores, pairs, arm):
    a, b = pairs.T
    dx = (x[b]-x[a])[:, COLS[arm]]
    margin = margins(scores, cells.class_id.to_numpy())
    dy = margin[b]-margin[a]
    w = pair_weights(cells, pairs)
    beta = np.linalg.solve(dx.T@(w[:, None]*dx)+.1*np.eye(dx.shape[1]), dx.T@(w*dy))
    return beta


def rank_metrics(cells, scores, pairs, quality, seed, arm, role):
    a, b = pairs.T
    margin = margins(scores, cells.class_id.to_numpy())
    target = margin[b]-margin[a]; prediction = quality[b]-quality[a]
    w = pair_weights(cells, pairs)
    active = np.abs(target) > 1e-12
    concordance = ((prediction*target > 0).astype(float)+.5*(np.abs(prediction) <= 1e-12))[active]
    return dict(seed=seed, arm=arm, role=role, pairs=len(pairs),
        ships=int(cells.iloc[a].mmsi.nunique()), MSE=float(w@(prediction-target)**2),
        zero_MSE=float(w@target**2),
        pair_ranking_accuracy=float(100*np.average(concordance, weights=w[active])) if active.any() else np.nan)


def selection(meta, scores, cells, cell_scores, members, qualities, matched):
    _, baseline, multiview = vessel_targets(scores, meta)
    result = {'equal': baseline.copy(), 'raw_confidence': baseline.copy(), 'label_oracle': baseline.copy()}
    for arm in ARMS:
        for mode in ('soft', 'hard'):
            result[arm+'_'+mode] = baseline.copy()
    eligible = np.zeros(len(meta), bool)
    detail = []
    for _, group in cells.groupby('mmsi', sort=True):
        indexes = group.index.to_numpy()
        if len(indexes) < 3:
            continue
        for focal in indexes:
            donor = indexes[indexes != focal]
            known = cells.loc[donor, 'known'].to_numpy(bool).copy()
            if matched:
                aa = cells.loc[donor, 'incidence_deg'].to_numpy(float)
                af = float(cells.loc[focal, 'incidence_deg'])
                known &= (cells.loc[donor, 'port'].to_numpy() == cells.loc[focal, 'port'])
                known &= np.isfinite(aa) & np.isfinite(af) & (np.abs(aa-af) <= 2)
            if known.sum() < 2:
                continue
            ids = members[focal]; eligible[ids] = True
            means = cell_scores[donor]; n = len(donor); nk = int(known.sum())
            ordered = np.sort(means[known], axis=1); gap = ordered[:, -1]-ordered[:, -2]
            local_weights = {'raw_confidence': 1+gap/(1+gap)}
            for arm, quality in qualities.items():
                q = quality[donor[known]]
                span = np.ptp(q)
                local_weights[arm+'_soft'] = 1+(q-q.min())/span if span > 1e-12 else np.ones(nk)
                local_weights[arm+'_hard'] = np.isclose(q, q.max(), rtol=0, atol=1e-12).astype(float)
            for arm, local in local_weights.items():
                weight = np.ones(n)
                weight[known] = local/local.mean()
                result[arm][ids] = weight@means/n
            # Preserve unavailable-donor mass, including in the finite oracle.
            pool = np.vstack([baseline[ids[0]], (means[~known].sum(0)+nk*means[known])/n])
            y = int(cells.loc[focal, 'class_id'])
            result['label_oracle'][ids] = pool[np.argmax(margins(pool, np.full(len(pool), y)))]
            detail.append(dict(mmsi=cells.loc[focal, 'mmsi'], product=cells.loc[focal, 'product'],
                focal_rows=len(ids), known_donors=nk, total_donors=n))
    return result, multiview, eligible, detail


def self_test():
    x = motion_features(np.array([0., 2., 2., 2.]), np.array([np.nan, 10., 190., 100.]),
                        np.array([np.nan, 10., 10., 10.]), np.ones(4, bool))
    np.testing.assert_array_equal(x[0], np.zeros(6))
    np.testing.assert_allclose(x[1], x[2], atol=1e-12)
    np.testing.assert_allclose(x[1, 2:4], -x[3, 2:4], atol=1e-12)
    meta = pd.DataFrame(dict(mmsi=[1]*5, product=['a', 'a', 'b', 'c', 'd'],
        class_id=[0]*5, port=['p']*5, incidence_deg=[35.]*5))
    scores = np.zeros((5, 8)); scores[:, 0] = [100, 100, -2, 4, 7]
    motion = np.zeros((5, 6)); motion[4] = np.nan
    cells, vectors, cs, members = make_cells(meta, scores, motion, np.zeros(5, int))
    qualities = {a: np.arange(4, dtype=float) for a in ARMS}
    p, _, eligible, _ = selection(meta, scores, cells, cs, members, qualities, False)
    matched, _, _, _ = selection(meta, scores, cells, cs, members, qualities, True)
    for arm in p:
        np.testing.assert_array_equal(p[arm], matched[arm])
    assert eligible[:2].all()
    # Known donor mass is 2/3; unavailable d keeps weight 1/3.
    np.testing.assert_allclose(p['direction_hard'][0, 0], (2*4+7)/3)
    altered = scores.copy(); altered[:2] = -999
    c2, _, s2, m2 = make_cells(meta, altered, motion, np.zeros(5, int))
    q, _, _, _ = selection(meta, altered, c2, s2, m2, qualities, False)
    for arm in p:
        np.testing.assert_allclose(p[arm][:2], q[arm][:2], atol=1e-12)
    assert np.max(np.abs(x)) <= 1


def main():
    self_test()
    OUT.mkdir(exist_ok=False)
    protocol = dict(date='2026-10-12', seeds=list(SEEDS), ridge=.1, neural_fits=0,
        motion='Bounded u=SOG/(1+SOG), u^2*cos2(COG-ground_range_axis), u^2*sin2(...). Not measured radial velocity or angular motion.',
        zero='Reported valid zero speed maps to zero vector even if COG/grid missing; moving rows require valid COG and exact cached native grid.',
        inputs='Same-known-donor mask in every arm. No dimensions/heading requirement. Average valid observation features per vessel/product.',
        models=COLS, response='Within-vessel different-product true-class margin difference; class not a predictor. No centering/scaling or intercept; fixed ridge .1.',
        folds='Existing source vessel3fold, response reader only. Upstream cached teacher not nested refit. All full readers frozen before held scoring.',
        weights='Soft local1..2 minmax quality, equal if tied. Hard diagnostic. Known total mass preserved; unknown equal mass. Entire focal product excluded.',
        pools='All other acquisitions primary; same-port/incidencegap<=2 donor reweighting diagnostic. Not conditional causal identification.',
        decision='Check headroom and held-ship exploitability vs equal/confidence/speed/capacity. Do not promote held-best variant or start student automatically.',
        limits='Repeated historical source development cohorts; metadata used only for teacher feasibility. No target evaluation, new independent confirmation, or student gain claim.')
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    meta = pd.read_csv(BASE/'observations.csv')
    motion = pd.read_csv(BASE/'motion_knowledge_20261011/motion_knowledge.csv').set_index('obs_id')
    grid = pd.read_csv(BASE/'interaction_feasibility_20261012/geometry_join.csv').set_index('obs_id')
    assert motion.index.is_unique and grid.index.is_unique
    with np.load(BASE/'motion_knowledge_20261011/attributes.npz') as f:
        ids_by_role = {r: f[r+'_obs_ids'] for r in ('fit', 'held')}
    assert not set(meta.iloc[ids_by_role['fit']].mmsi) & set(meta.iloc[ids_by_role['held']].mmsi)
    with np.load(BASE/'crossfit_evidence_20261011/source_folds.npz') as f:
        np.testing.assert_array_equal(f['fit_obs_ids'], ids_by_role['fit'])
        source_fold = f['assignment']
    inputs = {}
    for role, ids in ids_by_role.items():
        mm = meta.iloc[ids].reset_index(drop=True)
        m = motion.loc[ids]; g = grid.loc[ids]
        np.testing.assert_array_equal(m.mmsi.to_numpy(), mm.mmsi.to_numpy())
        np.testing.assert_array_equal(m['product'].to_numpy(), mm['product'].to_numpy())
        sog = m.sog_knots.to_numpy(float)
        cog = m.cog_deg.to_numpy(float).copy()
        cog[~m.cog_deg_valid.to_numpy(bool)] = np.nan
        az = g.range_azimuth.to_numpy(float).copy(); az[g.grid_status.to_numpy() != 'ok'] = np.nan
        valid = m.sog_knots_valid.to_numpy(bool) & m.match_status.eq('matched').to_numpy()
        valid &= np.isfinite(sog) & (sog >= 0) & (sog < 102.2)
        feat = motion_features(sog, cog, az, valid)
        inputs[role] = (mm, feat, source_fold if role == 'fit' else np.full(len(ids), -1))
    models, oof_quality, fit_bundles = {}, {}, {}
    ranking_rows = []
    for seed in SEEDS:
        ids = ids_by_role['fit']; mm, feat, fold = inputs['fit']
        with np.load(BASE/'geometry_train_teacher_20261011'/f'teachers_{seed}.npz') as f:
            np.testing.assert_array_equal(f['fit_obs_ids'], ids)
            scores = f['original_oof']
        cells, x, cs, members = make_cells(mm, scores, feat, fold)
        fit_bundles[seed] = (scores, cells, x, cs, members)
        pp = pairs_for(cells)
        for arm in ARMS:
            oq = np.full(len(cells), np.nan)
            for k in range(3):
                train = pairs_for(cells, cells.fold.to_numpy() != k)
                beta = fit_rank(cells, x, cs, train, arm)
                select = cells.fold.to_numpy() == k
                oq[select] = x[select][:, COLS[arm]]@beta
                np.savez_compressed(OUT/f'fold{k}_{arm}_{seed}.npz', beta=beta, train_pairs=train)
            oof_quality[seed, arm] = oq
            ranking_rows.append(rank_metrics(cells, cs, pp, oq, seed, arm, 'source_cv'))
            beta = fit_rank(cells, x, cs, pp, arm)
            models[seed, arm] = beta
            np.savez_compressed(OUT/f'full_{arm}_{seed}.npz', beta=beta, train_pairs=pp)
    (OUT/'FROZEN.json').write_text(json.dumps(dict(all_readers_fitted=True, held_scores_read=False)), encoding='utf-8')
    rows, coverage, event_rows = [], [], []
    for role in ('fit', 'held'):
        ids = ids_by_role[role]; mm, feat, fold = inputs[role]
        for seed in SEEDS:
            if role == 'fit':
                scores, cells, x, cs, members = fit_bundles[seed]
                quality = {a: oof_quality[seed, a] for a in ARMS}
            else:
                with np.load(BASE/'crossfit_evidence_20261011'/f'full_{seed}_outputs.npz') as f:
                    assert not set(meta.iloc[f['fit_obs_ids']].mmsi) & set(mm.mmsi)
                    scores = f['prior'][ids]
                cells, x, cs, members = make_cells(mm, scores, feat, fold)
                quality = {a: x[:, COLS[a]]@models[seed, a] for a in ARMS}
                for arm in ARMS:
                    ranking_rows.append(rank_metrics(cells, cs, pairs_for(cells), quality[arm], seed, arm, 'held'))
            cells.to_csv(OUT/f'{role}_cells_{seed}.csv', index=False)
            np.savez_compressed(OUT/f'{role}_quality_{seed}.npz', x=x, scores=cs, **quality)
            for matched in (False, True):
                pool = 'same_port_angle' if matched else 'all_history'
                pred, multi, eligible, detail = selection(mm, scores, cells, cs, members, quality, matched)
                coverage.append(dict(role=role, seed=seed, pool=pool, observations=len(ids),
                    known_rows=int(np.isfinite(feat).all(1).sum()), multiview=int(multi.sum()),
                    eligible=int(eligible.sum()), eligible_ships=int(mm.loc[eligible].mmsi.nunique()),
                    ranking_pairs=len(pairs_for(cells)),
                    ranking_ships=int(cells.iloc[pairs_for(cells).ravel()].mmsi.nunique())))
                pd.DataFrame(detail).to_csv(OUT/f'{role}_{seed}_{pool}_donors.csv', index=False)
                np.savez_compressed(OUT/f'{role}_{seed}_{pool}.npz', obs_ids=ids, multiview=multi, eligible=eligible, **pred)
                for cohort, mask in (('multiview', multi), ('eligible', eligible)):
                    truth = mm.loc[mask].class_id.to_numpy(); old = pred['equal'][mask].argmax(1)
                    for arm, score in pred.items():
                        now = score[mask].argmax(1)
                        rows.append(dict(role=role, seed=seed, pool=pool, cohort=cohort, arm=arm,
                            **assess(score[mask], truth, mm.loc[mask]),
                            rescue=int(((old != truth)&(now == truth)).sum()),
                            harm=int(((old == truth)&(now != truth)).sum())))
                        if role == 'held' and cohort == 'multiview':
                            changed = np.flatnonzero(mask)[now != old]
                            for i in changed:
                                event_rows.append(dict(seed=seed, pool=pool, arm=arm, obs_id=int(ids[i]),
                                    mmsi=int(mm.iloc[i].mmsi), class_id=int(mm.iloc[i].class_id),
                                    before=int(pred['equal'][i].argmax()), after=int(score[i].argmax()),
                                    old_correct=bool(pred['equal'][i].argmax() == mm.iloc[i].class_id),
                                    new_correct=bool(score[i].argmax() == mm.iloc[i].class_id)))
    pd.DataFrame(rows).to_csv(OUT/'results.csv', index=False)
    pd.DataFrame(ranking_rows).to_csv(OUT/'ranking_results.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv', index=False)
    pd.DataFrame(event_rows).to_csv(OUT/'held_changed_predictions.csv', index=False)
    summary = pd.DataFrame(rows).groupby(['role', 'pool', 'cohort', 'arm'])[['OA', 'BA', 'ship_balanced_accuracy', 'CE', 'rescue', 'harm']].mean()
    summary.to_csv(OUT/'summary.csv')
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(completed=True, neural_fits=0,
        ridge_fits=len(ARMS)*len(SEEDS)*4, synthetic_checks_passed=True, target_evaluations=0)), encoding='utf-8')
    print(pd.DataFrame(coverage).drop(columns='seed').drop_duplicates().to_string(index=False))
    print(pd.DataFrame(ranking_rows).groupby(['role', 'arm']).mean(numeric_only=True).to_string())
    print(summary.loc[('held', 'all_history', 'multiview')].to_string())


if __name__ == '__main__':
    main()
