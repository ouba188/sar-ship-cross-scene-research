"""Source-only candidate mining, then held-port conditional shift readout."""

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from conditional_invariant_directions import nuisance_penalized_basis
from stable_port_directions import _project
from sar_joint_completion import SOURCE, write_json


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OLD = ROOT/'stable_port_directions_20261008'
OUT = ROOT/'conditional_invariance_probe_20261008'
ARMS = ('pooled', 'stable', 'conditional_penalty')
QUANTILES = (np.arange(101)+.5)/101


def weighted_quantiles(values, weights):
    order = np.argsort(values, kind='stable')
    values, weights = values[order], weights[order]
    position = (np.cumsum(weights)-.5*weights)/weights.sum()
    return np.interp(QUANTILES, position, values)


def source_statistics(z, labels, domains):
    """Equal-domain summaries within each class, never pooled by class prior."""
    classes = np.arange(8)
    shape = (len(classes), z.shape[1])
    means = np.full(shape, np.nan)
    within = np.full(shape, np.nan)
    between = np.full(shape, np.nan)
    quantiles = np.full((*shape, len(QUANTILES)), np.nan)
    counts = np.zeros(8, dtype=int)
    for c in classes:
        groups = [np.flatnonzero((labels == c) & (domains == d)) for d in np.unique(domains)]
        groups = [r for r in groups if len(r) >= 5]
        if len(groups) < 3:
            continue
        counts[c] = len(groups)
        centers = np.stack([z[r].mean(0) for r in groups])
        means[c] = centers.mean(0)
        between[c] = centers.var(0)
        within[c] = np.stack([z[r].var(0) for r in groups]).mean(0)
        rows = np.concatenate(groups)
        weights = np.concatenate([np.full(len(r), 1./len(r)) for r in groups])
        for j in range(z.shape[1]):
            quantiles[c, j] = weighted_quantiles(z[rows, j], weights)
    return dict(class_mean=means, within_var=within, between_var=between,
                class_quantiles=quantiles, class_domains=counts)


def pair_readout(z, y, c, d, mean_c, mean_d, within_c, within_d, quant_c, quant_d):
    """Fixed source orientation/threshold; finite-sample shift is descriptive."""
    a, b = z[y == c], z[y == d]
    sigma = max(float(np.sqrt(.5*(within_c+within_d))), 1e-8)
    qa, qb = np.quantile(a, QUANTILES), np.quantile(b, QUANTILES)
    w_c, w_d = np.mean(np.abs(qa-quant_c))/sigma, np.mean(np.abs(qb-quant_d))/sigma
    threshold = .5*(mean_c+mean_d)
    return dict(auc=float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])),
        fixed_threshold_ba=float(.5*((a > threshold).mean()+(b <= threshold).mean())),
        class_c_w1=float(w_c), class_d_w1=float(w_d), conditional_w1=float(.5*(w_c+w_d)),
        class_c_mean_shift=float((a.mean()-mean_c)/sigma),
        class_d_mean_shift=float((b.mean()-mean_d)/sigma),
        source_gap=float((mean_c-mean_d)/sigma), target_gap=float((a.mean()-b.mean())/sigma))


def freeze(port, base, frame):
    dest = OUT/'ports'/port
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(OLD/'ports'/port/'source_direction_bank.npz') as saved:
        bank = {k: saved[k] for k in saved.files}
    with np.load(OLD/'ports'/port/'heldout_direction_scores.npz') as saved:
        query_rows, eval_mask = saved['query_rows'], saved['eval_mask']
    rows = bank['source_rows']
    assert not np.intersect1d(rows, query_rows).size
    labels = frame.class_id.to_numpy()[rows]
    domains = frame.port.to_numpy()[rows]
    assert not np.any(domains == port)
    xs = np.array(base[rows], dtype=np.float32)
    xs -= bank['feature_center']
    xs /= bank['feature_scale']
    candidate = nuisance_penalized_basis(xs, labels, domains, bank['pairs'], bank['pooled_basis'])
    bases = (bank['pooled_basis'], bank['stable_basis'], candidate['basis'])
    frozen = dict(pairs=bank['pairs'], source_rows=rows, query_rows=query_rows, eval_mask=eval_mask,
                  feature_center=bank['feature_center'], feature_scale=bank['feature_scale'],
                  retained_norm=candidate['retained_norm'])
    for arm, basis in zip(ARMS, bases):
        frozen[arm+'_basis'] = basis
        z = _project(xs, basis)
        for k, v in source_statistics(z, labels, domains).items():
            frozen[arm+'_'+k] = v
    del xs, z
    gc.collect()
    xq = np.array(base[query_rows], dtype=np.float32)
    xq -= bank['feature_center']
    xq /= bank['feature_scale']
    for arm, basis in zip(ARMS, bases):
        frozen[arm+'_target_scores'] = _project(xq, basis).astype(np.float32)
    np.savez_compressed(dest/'frozen.npz', **frozen)
    write_json(dest/'FROZEN.json', dict(port=port, source_rows=len(rows), target_rows=len(query_rows),
        pairs=len(bank['pairs']), source_only_fit=True, target_support_labels_used=False))
    print(f'FROZEN {port}: {len(bank["pairs"])} pairs; no target labels fitted', flush=True)


def evaluate(inventory, frame):
    records = []
    for item in inventory:
        port = item['port']
        with np.load(OUT/'ports'/port/'frozen.npz') as saved:
            y = frame.class_id.to_numpy()[saved['query_rows']]
            for draw, ev in enumerate(saved['eval_mask']):
                for j, (c, d) in enumerate(saved['pairs']):
                    nc, nd = int(np.sum(ev & (y == c))), int(np.sum(ev & (y == d)))
                    if min(nc, nd) < 20:
                        continue
                    for arm in ARMS:
                        mu, within, between, quant = [saved[arm+'_'+key] for key in
                            ('class_mean', 'within_var', 'between_var', 'class_quantiles')]
                        m = pair_readout(saved[arm+'_target_scores'][ev, j], y[ev], c, d,
                            mu[c, j], mu[d, j], within[c, j], within[d, j], quant[c, j], quant[d, j])
                        m['source_mean_drift_ratio'] = float(np.sqrt(
                            (between[c, j]+between[d, j])/max(within[c, j]+within[d, j], 1e-16)))
                        # This is a predefined descriptive screen, not an equivalence test.
                        m['descriptive_match'] = bool(max(m['class_c_w1'], m['class_d_w1']) <= .25
                            and m['auc'] >= .70 and m['source_gap'] >= .5)
                        records.append(dict(port=port, draw=draw, arm=arm, class_c=int(c),
                                            class_d=int(d), class_c_count=nc, class_d_count=nd, **m))
    df = pd.DataFrame(records)
    df.to_csv(OUT/'draw_pair_metrics.csv', index=False, encoding='utf-8-sig')
    numeric = ['auc', 'fixed_threshold_ba', 'conditional_w1', 'class_c_w1', 'class_d_w1',
               'source_gap', 'target_gap', 'source_mean_drift_ratio', 'descriptive_match']
    pp = df.groupby(['port', 'arm', 'class_c', 'class_d'], as_index=False)[numeric].mean()
    pp.to_csv(OUT/'port_pair_metrics.csv', index=False, encoding='utf-8-sig')
    pm = pp.groupby(['port', 'arm'], as_index=False)[numeric].mean()
    pm.to_csv(OUT/'port_means.csv', index=False, encoding='utf-8-sig')
    overall = pm.groupby('arm')[numeric].mean().reset_index()
    overall.to_csv(OUT/'equal_port_means.csv', index=False, encoding='utf-8-sig')
    # Draws overlap; five support draws are not five independent confirmations.
    all_draws = df.groupby(['port', 'arm', 'class_c', 'class_d']).agg(
        draws=('draw', 'size'), all_draws_match=('descriptive_match', 'all')).reset_index()
    all_draws.to_csv(OUT/'descriptive_matches.csv', index=False, encoding='utf-8-sig')
    summary = dict(status='complete', ports=len(inventory), evaluated_ports=int(df.port.nunique()),
        rows=len(df), metrics=overall.to_dict('records'),
        pair_ports_per_arm=pp.groupby('arm').size().to_dict(),
        all_five_draw_match_counts=all_draws[(all_draws.draws == 5) & all_draws.all_draws_match]
            .groupby('arm').size().to_dict(),
        independent_confirmation=False, causal_identification=False, main_model_changed=False)
    write_json(OUT/'COMPLETE.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Completed output is preserved; use a new directory for a changed experiment')
    protocol = dict(date='2026-10-08', objective='Mine class-conditional invariant information candidates',
        invariant_target='Approximate P(Z|Y,port) stability plus retained label discrimination',
        not_claimed='Neither P(Y|Z,port) invariance nor causal identification follows automatically',
        source='Existing outer identity/product-purged sources; frozen R50 common coordinates',
        arms=list(ARMS), candidate='L2-unit (I+U.T U)^-1 pooled contrasts', strength=1.,
        nuisance='Within-class equal-port centered feature means; equal class then equal port weighting',
        source_min_cell_count=5, source_min_domains_per_class=3,
        target_min_class_count_per_pair=20, target_labels='Readout only, original supports excluded',
        readout='Oriented AUC, fixed source midpoint BA, classwise 101-quantile W1/source within-class SD',
        screen='Both class W1 <=0.25 SD, AUC>=0.70, source gap>=0.5 SD; descriptive only',
        policy='No target-selected directions or parameter search; freeze all ports before target readout',
        aggregation='Draw mean then pair mean within port then equal-port mean',
        limitations='Development corpus; repeated vessels/scenes and finite-sample shifts; no formal equivalence proof',
        main_model_unchanged=True)
    write_json(OUT/'PROTOCOL.json', protocol)
    inventory = json.loads((ROOT/'oof_base_adapt_results_20261007/POLICIES_SAVED.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id', 'class_name', 'port'], keep_default_na=False)
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    with threadpool_limits(limits=2):
        for item in inventory:
            if not (OUT/'ports'/item['port']/'FROZEN.json').exists():
                freeze(item['port'], base, frame)
        write_json(OUT/'ALL_FROZEN.json', dict(ports=[i['port'] for i in inventory]))
        evaluate(inventory, frame)


if __name__ == '__main__':
    main()
