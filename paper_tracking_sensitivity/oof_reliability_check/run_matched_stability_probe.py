"""Separate source discrimination from held-port stability in pair directions."""

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from matched_stability_control import equal_domain_pair_moments, reflect_with_matched_moments
from run_conditional_invariance_probe import weighted_quantiles, pair_readout
from sar_joint_completion import SOURCE, write_json


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
CANDIDATE = ROOT/'conditional_invariance_probe_20261008'
PARENT = ROOT/'module_completion_20261008/ports_projection'
OUT = ROOT/'matched_stability_probe_20261008'
ARMS = ('pooled', 'candidate', 'matched_reflection')
CALIPER = .02


def protocol():
    record = dict(date='2026-10-08', purpose='Distinguish generic discrimination from candidate stability',
        arms=list(ARMS), classes=8, directions=28,
        control='Reflect candidate in covariance metric in [pooled,candidate] plane',
        matching='Exactly equal source class-mean gap and class-balanced total within-class variance',
        covariance='Each class mass1/2; domains with >=5rows equal within class; at least3 per class',
        distinction='Total within-class variance includes within-class port mean variation',
        source_auc_ba_caliper=CALIPER, degenerate='Report but exclude identical/numerically degenerate controls',
        primary_subset='Nondegenerate controls with both source AUC and fixed-midpoint BA within2pp',
        secondary_subset='Primary subset where candidate source normalized between-port drift is smaller',
        source_labels='Outer identity/product-purged source only; original target64 labels unused in matching',
        target='23 held ports, original5 query masks, both pair classes>=20',
        normalization='W1 divided by source TOTAL within-class SD, unlike prior within-port-only denominator',
        readout='Pair AUC, fixed source midpoint BA, normalized W1, pair rescues/harms vs existing Base64',
        pair_reference='Argmax restricted to the two true classes, diagnostic only, not deployable8class accuracy',
        deployment='No module added or main-model change; no target subset chosen for fitting',
        freeze='All23 source choices/thresholds/target projections frozen before target query readout',
        aggregation='Mean draws, then pairs within port, then equal ports; matched subsets report coverage',
        inference='Not a causal effect or proof of joint invariance; higher moments/readout sensitivity remain unmatched',
        independent_confirmation=False)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT/'PROTOCOL.json'
    if dest.exists():
        if json.loads(dest.read_text(encoding='utf-8')) != record:
            raise ValueError('Changed protocol needs a different output directory')
    else:
        write_json(dest, record)


def project_source(raw, rows, center, scale, bases):
    projected = np.empty((len(rows), bases.shape[1]), dtype=float)
    for lo in range(0, len(rows), 512):
        x = np.array(raw[rows[lo:lo+512]], dtype=np.float32)
        x -= center
        x /= scale
        projected[lo:lo+len(x)] = x@bases
    return projected


def source_stats(z, y, domains, c, d, weights):
    means, variances, quantiles, drift = [], [], [], []
    for label in (c, d):
        mask = (y == label) & (weights > 0)
        zz, ww = z[mask], weights[mask]
        mean = np.average(zz, weights=ww)
        means.append(mean)
        variances.append(np.average((zz-mean)**2, weights=ww))
        quantiles.append(weighted_quantiles(zz, ww))
        cell_mean = [zz[domains[mask] == port].mean() for port in np.unique(domains[mask])]
        drift.append(np.mean((np.asarray(cell_mean)-mean)**2))
    threshold = .5*(means[0]+means[1])
    valid = weights > 0
    correct = np.where(y == c, z > threshold, z <= threshold)
    variance = .5*sum(variances)
    return dict(mean=np.asarray(means), variance=np.asarray(variances),
        quantiles=np.stack(quantiles), threshold=float(threshold),
        gap=float((means[0]-means[1])/np.sqrt(max(variance, 1e-16))),
        auc=float(roc_auc_score(y[valid] == c, z[valid], sample_weight=weights[valid])),
        ba=float(np.average(correct[valid], weights=weights[valid])),
        drift=float(np.sqrt(.5*sum(drift)/max(variance, 1e-16))))


def freeze(port, raw, frame):
    folder = OUT/'ports'/port
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'FROZEN.json').exists():
        return
    with np.load(CANDIDATE/'ports'/port/'frozen.npz') as saved:
        rows, query, pairs = saved['source_rows'], saved['query_rows'], saved['pairs']
        masks = saved['eval_mask']
        pooled, candidate = saved['pooled_basis'], saved['conditional_penalty_basis']
        center, scale = saved['feature_center'], saved['feature_scale']
        query_pooled, query_candidate = saved['pooled_target_scores'], saved['conditional_penalty_target_scores']
    y, domains = frame.class_id.to_numpy()[rows], frame.port.to_numpy()[rows]
    if np.any(domains == port) or np.intersect1d(rows, query).size:
        raise ValueError('Outer target enters source fitting')
    z = project_source(raw, rows, center, scale, np.column_stack((pooled, candidate)))
    n = len(pairs)
    coefficients, targets, pair_stats, records, matches, contrasts, degeneracies = [], [], [], [], [], [], []
    for j, (c, d) in enumerate(pairs):
        pair_z = z[:, [j, j+n]]
        moments = equal_domain_pair_moments(pair_z, y, domains, c, d)
        control = reflect_with_matched_moments(moments['mean_c']-moments['mean_d'], moments['covariance'])
        w = control['coefficients']
        views = (pair_z[:, 0], pair_z[:, 1], pair_z@w)
        stats = [source_stats(v, y, domains, c, d, moments['weights']) for v in views]
        # Matching is checked in projected observations, independently of helper diagnostics.
        gaps = [s['mean'][0]-s['mean'][1] for s in stats]
        variances = [.5*s['variance'].sum() for s in stats]
        np.testing.assert_allclose(gaps[1], gaps[2], rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(variances[1], variances[2], rtol=1e-8, atol=1e-9)
        matched = (not control['degenerate'] and abs(stats[1]['auc']-stats[2]['auc']) <= CALIPER
                   and abs(stats[1]['ba']-stats[2]['ba']) <= CALIPER)
        contrast = matched and stats[1]['drift'] < stats[2]['drift']-1e-12
        for arm, stat in zip(ARMS, stats):
            records.append(dict(port=port, class_c=int(c), class_d=int(d), arm=arm,
                source_auc=stat['auc'], source_ba=stat['ba'], source_gap=stat['gap'],
                source_drift=stat['drift'], mean_gap=gaps[ARMS.index(arm)],
                total_variance=variances[ARMS.index(arm)], matched=matched,
                source_stability_contrast=contrast, degenerate=bool(control['degenerate'])))
        qz = np.column_stack((query_pooled[:,j], query_candidate[:,j]))
        targets.append(np.column_stack((qz, qz@w)))
        pair_stats.append(stats)
        coefficients.append(w)
        matches.append(matched)
        contrasts.append(contrast)
        degeneracies.append(control['degenerate'])
    np.savez_compressed(folder/'frozen.npz', source_rows=rows, query_rows=query, pairs=pairs,
        eval_mask=masks, coefficients=np.stack(coefficients), target_scores=np.stack(targets, axis=1),
        matched=np.asarray(matches), source_stability_contrast=np.asarray(contrasts),
        degenerate=np.asarray(degeneracies), arm_names=np.asarray(ARMS),
        **{key: np.asarray([[s[key] for s in pair] for pair in pair_stats])
           for key in ('mean', 'variance', 'quantiles', 'threshold', 'gap', 'auc', 'ba', 'drift')})
    pd.DataFrame(records).to_csv(folder/'source_matching.csv', index=False)
    write_json(folder/'FROZEN.json', dict(port=port, pairs=n, matched=int(sum(matches)),
        source_stability_contrast=int(sum(contrasts)), degenerate=int(sum(degeneracies)),
        target_labels_fitted=0, main_model_unchanged=True))
    print(f'FROZEN {port}: pairs={n}, matched={sum(matches)}, sourcecontrast={sum(contrasts)}, '
          f'degenerate={sum(degeneracies)}', flush=True)


def evaluate(inventory, frame):
    records, matching = [], []
    for item in inventory:
        port = item['port']
        folder = OUT/'ports'/port
        matching.append(pd.read_csv(folder/'source_matching.csv'))
        with np.load(folder/'frozen.npz') as saved:
            p = {k: saved[k] for k in saved.files}
        with np.load(PARENT/port/'matched_joint/joint_policies.npz') as saved:
            np.testing.assert_array_equal(saved['query_rows'], p['query_rows'])
            np.testing.assert_array_equal(saved['eval_mask'], p['eval_mask'])
            base = saved['base_scores']
        y = frame.class_id.to_numpy()[p['query_rows']]
        for draw, ev in enumerate(p['eval_mask']):
            for j, (c, d) in enumerate(p['pairs']):
                maskc, maskd = ev & (y == c), ev & (y == d)
                nc, nd = int(maskc.sum()), int(maskd.sum())
                if min(nc, nd) < 20:
                    continue
                mask = maskc | maskd
                weights = np.where(y[mask] == c, .5/nc, .5/nd)
                original = np.where(base[draw,:,c] > base[draw,:,d], c, d)
                base_ba = float(np.sum(weights*(original[mask] == y[mask])))
                for a, arm in enumerate(ARMS):
                    mu, var, quant = p['mean'][j,a], p['variance'][j,a], p['quantiles'][j,a]
                    z = p['target_scores'][:,j,a]
                    m = pair_readout(z[ev], y[ev], c, d, mu[0], mu[1], var[0], var[1], quant[0], quant[1])
                    pred = np.where(z > p['threshold'][j,a], c, d)
                    rescued = (original[mask] != y[mask]) & (pred[mask] == y[mask])
                    harmed = (original[mask] == y[mask]) & (pred[mask] != y[mask])
                    rescue, harm = float(weights@rescued), float(weights@harmed)
                    np.testing.assert_allclose(rescue-harm, m['fixed_threshold_ba']-base_ba, atol=1e-12)
                    records.append(dict(port=port, draw=draw, class_c=int(c), class_d=int(d), arm=arm,
                        class_c_count=nc, class_d_count=nd, matched=bool(p['matched'][j]),
                        source_stability_contrast=bool(p['source_stability_contrast'][j]),
                        degenerate=bool(p['degenerate'][j]), base_pair_ba=base_ba,
                        rescue_rate=rescue, harm_rate=harm, **m))
    df = pd.DataFrame(records)
    df.to_csv(OUT/'draw_pair_metrics.csv', index=False, encoding='utf-8-sig')
    pd.concat(matching).to_csv(OUT/'source_matching.csv', index=False, encoding='utf-8-sig')
    cols = ['auc', 'fixed_threshold_ba', 'conditional_w1', 'class_c_w1', 'class_d_w1',
            'source_gap', 'target_gap', 'base_pair_ba', 'rescue_rate', 'harm_rate']
    allpairs, allports, summaries, contrasts = [], [], [], []
    for name, mask in (('all', np.ones(len(df), dtype=bool)), ('matched', df.matched),
                       ('source_stability_contrast', df.source_stability_contrast)):
        subset = df[mask]
        if subset.empty:
            continue
        pp = subset.groupby(['port', 'arm', 'class_c', 'class_d'], as_index=False)[cols].mean()
        pm = pp.groupby(['port', 'arm'], as_index=False)[cols].mean()
        overall = pm.groupby('arm')[cols].mean().reset_index()
        for tab in (pp, pm, overall):
            tab['subset'] = name
        allpairs.append(pp)
        allports.append(pm)
        summaries.append(overall)
        a = pm[pm.arm == 'candidate'].set_index('port')
        b = pm[pm.arm == 'matched_reflection'].set_index('port')
        delta = a.fixed_threshold_ba-b.fixed_threshold_ba
        shift = a.conditional_w1-b.conditional_w1
        contrasts.append(dict(subset=name, ports=len(a), pair_ports=int(len(pp)/len(ARMS)),
            delta_ba_pp=float(100*delta.mean()), delta_auc=float((a.auc-b.auc).mean()),
            delta_w1=float(shift.mean()), ba_positive_ports=int((delta>1e-12).sum()),
            ba_negative_ports=int((delta < -1e-12).sum()), lower_shift_ports=int((shift < -1e-12).sum()),
            both_positive_ports=int(((delta>1e-12) & (shift < -1e-12)).sum())))
    pd.concat(allpairs).to_csv(OUT/'port_pair_metrics.csv', index=False, encoding='utf-8-sig')
    pd.concat(allports).to_csv(OUT/'port_means.csv', index=False, encoding='utf-8-sig')
    means = pd.concat(summaries)
    means.to_csv(OUT/'equal_port_means.csv', index=False, encoding='utf-8-sig')
    write_json(OUT/'COMPLETE.json', dict(ports=len(inventory), rows=len(df), contrasts=contrasts,
        means=means.to_dict('records'), main_model_changed=False, causal_identification=False,
        independent_confirmation=False))
    print(means.to_string(index=False), flush=True)
    print(json.dumps(contrasts, indent=2), flush=True)


def main():
    protocol()
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed output')
    inventory = json.loads((ROOT/'oof_base_adapt_results_20261007/POLICIES_SAVED.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id', 'port'], keep_default_na=False)
    raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    with threadpool_limits(limits=2):
        for item in inventory:
            freeze(item['port'], raw, frame)
        write_json(OUT/'ALL_FROZEN.json', dict(ports=len(inventory), target_labels_fitted=0))
        evaluate(inventory, frame)


if __name__ == '__main__':
    main()
