"""Mine repeated-vessel changes in existing geometry, then diagnose frozen models."""

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import json
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evidence_aware_handoff import predict_gate, route_scores
from run_protected_drift_activation import gate_features
from run_generic_transfer import metrics
from sar_joint_completion import write_json
from task_stable_sensitivity import ARMS, PARENT, ROOT, SOURCE, make_views, direct_scores
from same_vessel_temporal import (collect_observations, make_pairs, pair_covariance,
                                  class_port_centers, fit_temporal_directions)

OUT = ROOT / 'same_vessel_temporal_probe_20261009'
MODES = ('reference', 'temporal', 'class_protected')
STRENGTH = .25
RANK = 8


def latent(bank, g):
    r = g['stable_rank']
    return np.column_stack((bank[:, 192:192+r] * g['stable_scale'] + g['stable_center'],
                            bank[:, 192+r:]))


def jacobians(g):
    basis = np.column_stack((g['qs'], g['qd']))
    r, d = g['stable_rank'], g['drift_rank']
    raw = basis.T * g['feature_scale']
    pc = g['model']['pca'].components_
    xs = (raw @ pc.T) / g['model']['scale']
    comp = (raw - (raw @ pc.T) @ pc) / g['stats']['discarded_scale'][:2048]
    stable = (basis.T @ g['qs']) / g['stable_scale']
    drift = basis.T @ g['qd']
    bank = np.zeros((r+d, 192+r+d))
    bank[:, :128] = xs
    bank[:, 192:192+r] = stable
    bank[:, 192+r:] = drift
    return dict(basis=basis, raw=raw, xs=xs, comp=comp, stable=stable, drift=drift, bank=bank)


def gradient(g, fit, p, sv, j, arm, head, draw):
    coef = fit['heads'][arm+'_'+head]['coef']
    answer = j['bank'] @ coef
    beta = p[arm+'_'+head+'_coefficients'][draw]
    suppress, activate = arm == 'joint', arm != 'matched_baseline'
    dx, dc = j['xs'].copy(), j['comp'].copy()
    if suppress:
        dx -= .5 * j['drift'] @ g['drift_to_xs']
        dc -= .5 * j['drift'] @ g['drift_to_complement']
    if head == 'base':
        answer += (dx / g['base_scale'] / np.sqrt(128)) @ beta[:128]
        if activate:
            answer += (j['stable'] / np.sqrt(g['stable_rank'])) @ beta[128:128+g['stable_rank']]
    else:
        value = sv['common'][suppress]
        dk = .5 * ((dx / g['model']['expert_scale'][:128] / np.sqrt(192))
                   @ value['retained'][:, :128].T
                   + dc / np.sqrt(2560) @ value['complement'][:, :2048].T)
        if activate:
            dk += .25 * (j['stable'] / np.sqrt(g['stable_rank'])) @ sv['stable'].T
        answer += dk @ beta
    return answer


def verify_gradients(g, fit, p, j, tb, raw, extra, sv, gradients):
    use = np.flatnonzero(p['eval_mask'][0])[:3]
    before_bank = np.asarray(tb[use], dtype=float)
    before_raw = np.asarray(raw[p['query_rows'][use]], dtype=float)
    delta = np.linspace(-.3, .3, j['bank'].shape[0])
    after_bank = before_bank + delta @ j['bank']
    after_raw = before_raw + delta @ j['raw']
    before = make_views(before_bank, before_raw, extra[use], g)
    after = make_views(after_bank, after_raw, extra[use], g)
    replay, response = 0., 0.
    for arm in ARMS:
        index = list(p['arm_names']).index(arm)
        for head in ('base', 'expert'):
            a = direct_scores(before_bank, before, sv, fit['heads'], p, arm, 0, head)
            b = direct_scores(after_bank, after, sv, fit['heads'], p, arm, 0, head)
            replay = max(replay, float(np.max(np.abs(a-p[head+'_scores'][0, index, use]))))
            response = max(response, float(np.max(np.abs(b-a-delta @ gradients[arm, head]))))
    if max(replay, response) > 3e-6:
        raise ValueError(f'Raw-feature intervention mismatch: {replay}, {response}')
    return replay, response


def energy_fraction(cov, directions):
    total = np.trace(cov)
    return float(np.trace(directions.T @ cov @ directions) / total) if total > 0 else np.nan


def between_vessel_covariance(obs, z):
    groups = []
    for (c, port), cell in obs.groupby(['class_id', 'port'], sort=True):
        means = []
        for _, ship in cell.groupby('mmsi', sort=True):
            if len(ship) >= 2:
                means.append(z[ship.index].mean(0))
        if len(means) >= 2:
            a = np.asarray(means)
            a -= a.mean(0)
            groups.append((int(c), (a.T @ a) / len(a)))
    classes = sorted({c for c, _ in groups})
    return np.mean([np.mean([cov for c, cov in groups if c == label], axis=0)
                    for label in classes], axis=0)


def source_diagnostics(port, obs, bank, pairs, deltas, weights, bases, center, fit, j):
    records = []
    z = latent(bank, j['geometry'])
    y = obs.class_id.to_numpy()
    ia, ib = pairs.obs_a.to_numpy(dtype=int), pairs.obs_b.to_numpy(dtype=int)
    for arm in ARMS:
        for head in ('base', 'expert'):
            h = fit['heads'][arm+'_'+head]
            original = bank @ h['coef'] + h['intercept']
            g = j['bank'] @ h['coef']
            diff = original[ib] - original[ia]
            centered = diff-diff.mean(1, keepdims=True)
            old_energy = float(np.sum(weights[:, None] * centered**2))
            baseline = metrics(y, original.argmax(1), 8)
            for mode, directions in bases.items():
                shift = ((z-center) @ directions) @ directions.T
                scores = original-STRENGTH*shift @ g
                changed = metrics(y, scores.argmax(1), 8)
                dd = diff-STRENGTH*(deltas @ directions) @ (directions.T @ g)
                dd -= dd.mean(1, keepdims=True)
                new_energy = float(np.sum(weights[:, None] * dd**2))
                records.append(dict(port=port, arm=arm, head=head, mode=mode,
                    held_prior_oa=baseline['acc'], held_prior_ba=baseline['ba'],
                    oa_delta_pp=100*(changed['acc']-baseline['acc']),
                    ba_delta_pp=100*(changed['ba']-baseline['ba']),
                    temporal_centered_score_energy_ratio=new_energy/old_energy if old_energy > 0 else np.nan,
                    pair_prediction_disagreement_before=float(np.sum(weights*(original[ia].argmax(1)!=original[ib].argmax(1)))),
                    pair_prediction_disagreement_after=float(np.sum(weights*(scores[ia].argmax(1)!=scores[ib].argmax(1))))))
    return records


def freeze_port(folder, frame, raw):
    out = OUT/'ports'/folder.name
    out.mkdir(parents=True, exist_ok=True)
    with (folder/'geometry.pkl').open('rb') as f:
        g = pickle.load(f)
    with (folder/'source_models.pkl').open('rb') as f:
        fit = pickle.load(f)
    with np.load(folder/'policies.npz') as f:
        p = {k: f[k] for k in f.files}
    source, target = np.load(folder/'source_rows.npy'), np.load(folder/'target_rows.npy')
    np.testing.assert_array_equal(source, p['source_rows'])
    np.testing.assert_array_equal(target, p['query_rows'])
    sb = np.load(folder/'source_bank.npy', mmap_mode='r')
    tb = np.load(folder/'target_bank.npy', mmap_mode='r')
    obs, obsbank = collect_observations(frame, source, sb)
    target_ids = frame.mmsi.to_numpy()[target]
    if np.intersect1d(obs.mmsi.to_numpy(), target_ids).size:
        raise ValueError('Target vessel identity appears in source observation bank')
    ids = np.asarray(np.sort(obs.mmsi.unique()), dtype=str)
    ids = np.random.default_rng(20261009).permutation(ids)
    held_ids = ids[::3]
    held = obs.mmsi.isin(held_ids).to_numpy()
    train_obs, held_obs = obs.loc[~held].reset_index(drop=True), obs.loc[held].reset_index(drop=True)
    train_bank, held_bank = obsbank[~held], obsbank[held]
    train_z, held_z = latent(train_bank, g), latent(held_bank, g)
    tr_pairs, tr_delta, tr_w = make_pairs(train_obs, train_z)
    ho_pairs, ho_delta, ho_w = make_pairs(held_obs, held_z)
    train_cov, held_cov = pair_covariance(tr_delta, tr_w), pair_covariance(ho_delta, ho_w)
    classes, centers = class_port_centers(train_obs, train_z)
    if len(classes) != 8:
        raise ValueError('Train identities do not cover all eight classes')
    class_center = centers.mean(0)
    cc = centers-class_center
    class_cov = cc.T @ cc / len(cc)
    candidates = fit_temporal_directions(train_cov, centers, rank=RANK)
    bases = {name: candidates[name] for name in ('temporal', 'class_protected')}
    held_between = between_vessel_covariance(held_obs, held_z)
    pd.DataFrame(train_obs.groupby(['class_id','port']).agg(observations=('mmsi','size'),
        ships=('mmsi','nunique'))).to_csv(out/'train_observation_coverage.csv')
    pd.DataFrame(held_obs.groupby(['class_id','port']).agg(observations=('mmsi','size'),
        ships=('mmsi','nunique'))).to_csv(out/'held_observation_coverage.csv')
    tr_pairs.to_csv(out/'train_pairs.csv', index=False)
    ho_pairs.to_csv(out/'held_pairs.csv', index=False)
    eigens, geometric = [], []
    for mode, d in bases.items():
        transform = np.eye(len(train_cov))-STRENGTH*(d @ d.T)
        ratio = np.trace(transform @ held_cov @ transform.T)/np.trace(held_cov)
        geometric.append(dict(port=folder.name, mode=mode, rank=d.shape[1],
            train_temporal_fraction=energy_fraction(train_cov, d),
            held_temporal_fraction=energy_fraction(held_cov, d),
            held_between_vessel_fraction=energy_fraction(held_between, d),
            class_center_contrast_fraction=energy_fraction(class_cov, d),
            held_latent_temporal_energy_ratio=float(ratio),
            stable_span_fraction=float(np.square(d[:g['stable_rank']]).sum()/d.shape[1])))
        for k in range(d.shape[1]):
            direction = d[:, k:k+1]
            eigens.append(dict(port=folder.name, mode=mode, direction=k+1,
                train_temporal_fraction=energy_fraction(train_cov, direction),
                held_temporal_fraction=energy_fraction(held_cov, direction),
                held_between_vessel_fraction=energy_fraction(held_between, direction),
                class_center_contrast_fraction=energy_fraction(class_cov, direction),
                stable_span_fraction=float(np.square(direction[:g['stable_rank']]).sum())))
    # Freeze candidate geometry before reading any target query labels for evaluation.
    np.savez_compressed(out/'directions.npz', center=class_center, classes=classes,
        class_centers=centers, train_cov=train_cov, held_cov=held_cov, held_between_cov=held_between,
        train_mmsi=ids[~np.isin(ids,held_ids)], held_mmsi=held_ids, **bases)
    j = jacobians(g)
    j['geometry'] = g
    source_records = source_diagnostics(folder.name, held_obs, held_bank, ho_pairs,
        ho_delta, ho_w, bases, class_center, fit, j)
    z = latent(tb, g)
    shifts = {mode: ((z-class_center) @ d) @ d.T for mode, d in bases.items()}
    extra = np.load(ROOT/'module_completion_20261008/ports_projection'/folder.name/
                    'matched_joint/joint_extra.npy', mmap_mode='r')
    saved = {key: [] for key in ('predictions', 'base_predictions', 'expert_predictions', 'selected')}
    replay, response, gate_error = 0., 0., 0.
    for draw, ev in enumerate(p['eval_mask']):
        support = np.searchsorted(target, p['support_rows'][draw])
        np.testing.assert_array_equal(target[support], p['support_rows'][draw])
        if ev[support].any():
            raise ValueError('Support/query overlap')
        support_y = frame.class_id.to_numpy()[p['support_rows'][draw]]
        sv = make_views(tb[support], raw[target[support]], extra[support], g)
        grads = {(arm, head): gradient(g, fit, p, sv, j, arm, head, draw)
                 for arm in ARMS for head in ('base', 'expert')}
        if draw == 0:
            replay, response = verify_gradients(g, fit, p, j, tb, raw, extra, sv, grads)
        answers = []
        for arm in ARMS:
            index = list(p['arm_names']).index(arm)
            b0, e0 = p['base_scores'][draw, index], p['expert_scores'][draw, index]
            arm_results = []
            for mode in MODES:
                b, e = b0.copy(), e0.copy()
                if mode != 'reference':
                    b[ev] -= STRENGTH*shifts[mode][ev] @ grads[arm, 'base']
                    e[ev] -= STRENGTH*shifts[mode][ev] @ grads[arm, 'expert']
                score = predict_gate(fit['gates'][arm], gate_features(b, e))
                result = route_scores(b, e, score, support_y, ev, target)
                if mode == 'reference':
                    np.testing.assert_array_equal(result['predictions'], p['predictions'][draw,index])
                    np.testing.assert_array_equal(result['selected'], p['selected'][draw,index])
                    gate_error = max(gate_error, float(np.max(np.abs(score-p['gate_scores'][draw,index]))))
                if result['selected'][~ev].any() or result['selected'].sum() > result['cap']:
                    raise ValueError('Invalid handoff after temporal intervention')
                arm_results.append(result)
            answers.append(arm_results)
        for key in saved:
            saved[key].append(np.stack([[v[key] for v in row] for row in answers]))
    np.savez_compressed(out/'policies.npz', query_rows=target, support_rows=p['support_rows'],
        eval_mask=p['eval_mask'], arms=np.asarray(ARMS), modes=np.asarray(MODES),
        **{k: np.stack(v) for k,v in saved.items()})
    check = dict(port=folder.name, source_raw_rows=len(source), consolidated_observations=len(obs),
        valid_source_ships=len(ids), train_ships=int((~np.isin(ids,held_ids)).sum()), held_ships=len(held_ids),
        train_pairs=len(tr_pairs), held_pairs=len(ho_pairs),
        train_repeat_groups=int(tr_pairs.groupby(['port','mmsi']).ngroups),
        held_repeat_groups=int(ho_pairs.groupby(['port','mmsi']).ngroups),
        train_gap_days_median=float(tr_pairs.gap_days.median()),
        train_gap_days_max=float(tr_pairs.gap_days.max()), class_rank=candidates['class_rank'],
        search_rank=j['bank'].shape[0], max_score_replay_error=replay,
        max_raw_perturbation_response_error=response, max_gate_replay_error=gate_error)
    write_json(out/'FROZEN.json', check)
    print(json.dumps(check), flush=True)
    return geometric, eigens, source_records, check


def evaluate(frame):
    y = frame.class_id.to_numpy()
    records, recall_rows = [], []
    for folder in sorted((OUT/'ports').iterdir()):
        with np.load(folder/'policies.npz') as f:
            p = {k:f[k] for k in f.files}
        truth = y[p['query_rows']]
        for draw, ev in enumerate(p['eval_mask']):
            for a, arm in enumerate(ARMS):
                for m, mode in enumerate(MODES):
                    row = dict(port=folder.name, draw=draw, arm=arm, mode=mode,
                        query_n=int(ev.sum()), selected=int(p['selected'][draw,a,m].sum()))
                    for stage, key in [('full','predictions'),('base','base_predictions'),('expert','expert_predictions')]:
                        score = metrics(truth[ev], p[key][draw,a,m,ev], 8)
                        row[stage+'_oa'], row[stage+'_ba'] = score['acc'], score['ba']
                        for c, recall in enumerate(score['class_recall']):
                            recall_rows.append(dict(port=folder.name, draw=draw, arm=arm, mode=mode,
                                                     stage=stage, class_id=c, recall=recall))
                    records.append(row)
    df = pd.DataFrame(records)
    df.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(recall_rows).to_csv(OUT/'class_recalls.csv', index=False)
    pm = df.groupby(['port','arm','mode']).mean(numeric_only=True).drop(columns='draw').reset_index()
    reference = pm[pm['mode']=='reference'].set_index(['port','arm'])
    for metric in ('full_oa','full_ba','base_ba','expert_ba'):
        pm[metric+'_delta_pp'] = [100*(row[metric]-reference.loc[(row.port,row.arm),metric]) for _,row in pm.iterrows()]
    pm.to_csv(OUT/'port_means.csv', index=False)
    full8 = frame.groupby('port').class_id.nunique()
    summaries = []
    for name, use in [('all23', np.ones(len(pm),dtype=bool)),
                      ('all8_target12', pm.port.map(full8).eq(8).to_numpy())]:
        mean = pm[use].groupby(['arm','mode']).mean(numeric_only=True).reset_index()
        mean.insert(0, 'group', name)
        mean.insert(1, 'ports', pm.loc[use,'port'].nunique())
        summaries.append(mean)
    answer = pd.concat(summaries, ignore_index=True)
    answer.to_csv(OUT/'coverage_group_means.csv', index=False)
    print(answer.to_string(index=False), flush=True)
    return len(records)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed temporal diagnostic')
    write_json(OUT/'PROTOCOL.json', dict(date='2026-10-09', stage='Identify what may be suppressed, not retrain a new model',
        parent=str(PARENT), search_space='Raw source-standardized R50 coordinates in existing orthonormal[Qs,Qd]',
        identity='Source-purged valid9digit MMSI, consistent class within supplied source; no target query labels for mining',
        pairs='Same vessel AND same port, adjacent distinct acquisition times; same-time repeats averaged',
        covariance='.5 E(delta deltaT), equalclass/equalport/equalship/equaladjacentpair weights',
        split='Seed20261009 sorted source MMSI shuffle, every third identity held; same ship cannot enter both',
        axes='Top8 temporal eigenvectors, or top8 after projecting off the span of8 training classcenters',
        rank=RANK, suppression=STRENGTH, arms=list(ARMS), modes=list(MODES),
        source_boundary='Axes/classcenters fit only train identities. Existing geometry and source heads include held source identities; conditional replication, NOT fully nested OOF',
        source_scores='Frozen source-prior heads only, descriptive; not adapted fullmodel validation',
        target='Original64 labels for existing adaptation; query-only readout perturbation, fixed supportcoeff/R18/gate; recompute8class and5percent handoff',
        diagnosis='Frozen-readout intervention, not coherent retraining or proof of pure nuisance/physical factors',
        no_target_mode_selection=True, all_policies_frozen_before_query_readout=True,
        main_model_unchanged=True, independent_confirmation=False))
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id','port','mmsi','product'],
                        dtype={'mmsi':str,'product':str}, keep_default_na=False)
    raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    folders = [f for f in sorted((PARENT/'ports').iterdir()) if (f/'policies.npz').exists()]
    gs, ds, ss, checks = [], [], [], []
    with threadpool_limits(limits=2):
        for folder in folders:
            a,b,c,d = freeze_port(folder,frame,raw)
            gs.extend(a)
            ds.extend(b)
            ss.extend(c)
            checks.append(d)
    for name, records in [('geometry',gs),('directions',ds),('held_source_readout',ss),('verification',checks)]:
        pd.DataFrame(records).to_csv(OUT/(name+'.csv'),index=False)
    pd.DataFrame(gs).groupby('mode').mean(numeric_only=True).to_csv(OUT/'geometry_means.csv')
    pd.DataFrame(ss).groupby(['arm','head','mode']).mean(numeric_only=True).to_csv(OUT/'held_source_readout_means.csv')
    write_json(OUT/'ALL_FROZEN.json',dict(ports=len(folders),target_query_labels_used_for_selection=0))
    with threadpool_limits(limits=2):
        count = evaluate(frame)
    write_json(OUT/'COMPLETE.json',dict(ports=len(folders),draws=5,records=count,
        new_classifier_fits=0,main_model_unchanged=True,
        max_score_replay_error=max(v['max_score_replay_error'] for v in checks),
        max_raw_perturbation_response_error=max(v['max_raw_perturbation_response_error'] for v in checks),
        max_gate_replay_error=max(v['max_gate_replay_error'] for v in checks)))


if __name__ == '__main__':
    main()
