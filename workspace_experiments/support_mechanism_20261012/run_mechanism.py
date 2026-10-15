"""Frozen support-map interventions; CPU only, no new fit or query selection."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import json
import time

OUT = Path(__file__).resolve().parent
BUDGET = OUT.parent/'frozen_candidate_coverage_20261012/budget'
sys.path.insert(0, str(BUDGET))
import run_coverage as c
from strong_external import CachedStrongFeatures
import numpy as np
import pandas as pd
from scipy.special import softmax
from unified_head_core import coefficients, late, class_weights
from case_support_completion import COLS, predict, adapt_support

CACHE = Path('C:/Users/25791/AppData/Local/Temp/SAR_support_mechanism_20261012')
DATASETS = ('f0921', 'oss2')
NEW_ARMS = ('relation_only', 'coefficient_only', 'strength_match', 'class_strength_match')


def maps(raw, changed, prior, labels, ridge):
    a0 = coefficients(raw@raw.T, prior, labels, ridge)
    a1 = coefficients(changed@changed.T, prior, labels, ridge)
    w0, w1 = raw.T@a0, changed.T@a1
    gamma = float(np.sum(w0*w1)/max(np.sum(w0*w0), 1e-30))
    per_class = np.sum(w0*w1, axis=0)/np.maximum(np.sum(w0*w0, axis=0), 1e-30)
    result = dict(raw=w0, initial=w1, relation_only=changed.T@a0,
                  coefficient_only=raw.T@a1, strength_match=gamma*w0,
                  class_strength_match=per_class[None, :]*w0)
    return result, gamma, per_class


def freeze():
    start = time.perf_counter()
    records = []
    for ds in DATASETS:
        directory = BUDGET/'strong_external'/ds/ds
        feat = CachedStrongFeatures(ds)
        initial = c.archive(directory/'initial.npz')
        for draw in range(5):
            for budget in c.BUDGETS:
                old = c.archive(directory/f'{draw}_{budget}.npz')
                s, q = old['support_ids'], old['query_ids']
                raw, prior = feat.target(s)
                delta = predict(initial, raw[:, COLS], np.zeros(budget, int), 'image_history')
                changed = adapt_support(raw, delta, c.DOSES[budget])
                ws, gamma, gammas = maps(raw, changed, prior, old['support_labels'], budget/8.)
                block = raw[:, COLS]
                centered = block-block.mean(0)
                transformed = centered@(np.eye(len(COLS))+c.DOSES[budget]*initial['image_coef'])
                restored = transformed*np.sqrt(np.square(centered).sum()/np.square(transformed).sum())+block.mean(0)
                np.testing.assert_allclose(restored, changed[:, COLS], atol=1e-10, rtol=1e-10)
                centroid_error = float(abs(changed[:, COLS].mean(0)-block.mean(0)).max())
                energy_ratio = float(np.square(changed[:, COLS]-block.mean(0)).sum()/np.square(centered).sum())
                values = {arm: np.empty((len(q), 8), dtype=float) for arm in NEW_ARMS}
                joined = np.concatenate([ws[arm] for arm in ('raw', 'initial')+NEW_ARMS], axis=1)
                errors = dict(raw=0., initial=0.)
                for lo in range(0, len(q), 2048):
                    qi = q[lo:lo+2048]
                    qp, pq = feat.target(qi)
                    scores = (qp@joined).reshape(len(qi), 6, 8)+pq[:, None, :]
                    for j, arm in enumerate(('raw', 'initial')):
                        ref = old['expert_'+arm][lo:lo+len(qi)]
                        errors[arm] = max(errors[arm], float(abs(scores[:, j]-ref).max()))
                        np.testing.assert_allclose(scores[:, j], ref, atol=1e-9, rtol=1e-9)
                    for j, arm in enumerate(NEW_ARMS, start=2):
                        values[arm][lo:lo+len(qi)] = scores[:, j]
                # Label-free fixed query subset for subsequent relationship evaluation.
                rng = np.random.default_rng(20261012)
                pos = np.sort(rng.choice(len(q), min(4096, len(q)), replace=False))
                query_features = feat.target(q[pos])[0]
                cross0, cross1 = query_features@raw.T, query_features@changed.T
                dest = CACHE/ds
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest/f'{draw}_{budget}.npz', **values,
                    retrieval_query_ids=q[pos], retrieval_raw=cross0, retrieval_initial=cross1,
                    query_ids=q, support_ids=s)
                own = OUT/'weights'/ds
                own.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(own/f'{draw}_{budget}.npz', **ws, gamma=gamma, class_gamma=gammas)
                relative = lambda x: float(np.linalg.norm(x)/max(np.linalg.norm(ws['initial']), 1e-30))
                records.append(dict(dataset=ds, draw=draw, budget=budget, query_n=len(q),
                    gamma=gamma, gamma_min=float(gammas.min()), gamma_max=float(gammas.max()),
                    strength_weight_relative_error=relative(ws['strength_match']-ws['initial']),
                    class_strength_weight_relative_error=relative(ws['class_strength_match']-ws['initial']),
                    change_weight_relative_norm=relative(ws['raw']-ws['initial']),
                    centroid_error=centroid_error, energy_ratio=energy_ratio,
                    raw_replay_max_error=errors['raw'], initial_replay_max_error=errors['initial']))
                print('FROZEN', ds, draw, budget, 'replay', errors, flush=True)
    pd.DataFrame(records).to_csv(OUT/'operator_diagnostics.csv', index=False)
    c.write(OUT/'PREDICTIONS_FROZEN.json', dict(datasets=DATASETS, budgets=c.BUDGETS, draws=5,
        new_arms=NEW_ARMS, prediction_units=len(records), cache=str(CACHE),
        seconds=time.perf_counter()-start, query_labels_loaded=False, new_training=0,
        source_protocol=str(OUT/'PROTOCOL_ZH.md'), runtime=sys.executable))


def metrics(y, p, baseline):
    pred, bp = p.argmax(1), baseline.argmax(1)
    losses = np.square(p-np.eye(p.shape[1])[y]).sum(1)
    classes = np.unique(y)
    return dict(oa=100*float(np.mean(pred==y)),
        ba=100*float(np.mean([np.mean(pred[y==k]==k) for k in classes])),
        nll=float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-30)).mean()),
        brier=float(losses.mean()), macro_brier=float(np.mean([losses[y==k].mean() for k in classes])),
        rescue=int(np.sum((bp!=y)&(pred==y))), harm=int(np.sum((bp==y)&(pred!=y))))


def identity_values(frame):
    column = next((key for key in ('mmsi', 'mmsi_or_ship_id') if key in frame), None)
    if column is None:
        return np.full(len(frame), '', dtype=object)
    vals = frame[column].astype('string').fillna('').str.strip().str.replace(r'\.0$', '', regex=True).to_numpy()
    vals[np.isin(vals, ('nan', 'None', '0', '<NA>'))] = ''
    return vals


def retrieval(sim, y, sy, query_mmsi, support_mmsi):
    same = ((query_mmsi[:, None] == support_mmsi[None, :]) & (query_mmsi[:, None] != ''))
    valid = ~same
    relevant = (y[:, None] == sy[None, :]) & valid
    count = relevant.sum(1)
    cover = count > 0
    order = np.argsort(-np.where(valid, sim, -np.inf), axis=1, kind='stable')
    hits = np.take_along_axis(relevant, order, axis=1)
    ranks = np.arange(1, len(sy)+1)[None, :]
    ap = ((hits.cumsum(1)/ranks)*hits*(ranks<=count[:, None])).sum(1)/np.maximum(count, 1)
    def macro(values, mask):
        return float(np.mean([values[mask & (y==k)].mean() for k in np.unique(y[mask])])) if mask.any() else np.nan
    p1 = hits[:, 0].astype(float)
    k = np.minimum(valid.sum(1), 5)
    p5 = (hits*(ranks<=k[:, None])).sum(1)/np.maximum(k, 1)
    return dict(query_n=len(y), covered_n=int(cover.sum()), coverage=float(cover.mean()),
        map_at_r=float(ap[cover].mean()) if cover.any() else np.nan,
        macro_map_at_r=macro(ap, cover), macro_p1=macro(p1, cover), macro_p5=macro(p5, cover),
        p1_all=float(p1.mean()), same_ship_pairs_removed=int(same.sum()),
        query_known_identity=int(np.sum(query_mmsi!='')), support_known_identity=int(np.sum(support_mmsi!='')))


def evaluate():
    receipt = c.read_json(OUT/'PREDICTIONS_FROZEN.json')
    assert receipt['prediction_units'] == 30
    rows, relations, decompositions = [], [], []
    for ds in DATASETS:
        directory = BUDGET/'strong_external'/ds/ds
        table = pd.read_csv(c.read_json(directory/'PREDICTIONS_FROZEN.json')['target_manifest'])
        all_labels = table.class_id.to_numpy(int)
        mmsi = identity_values(table)
        for draw in range(5):
            for budget in c.BUDGETS:
                old = c.archive(directory/f'{draw}_{budget}.npz')
                new = c.archive(CACHE/ds/f'{draw}_{budget}.npz')
                q, s = old['query_ids'], old['support_ids']
                np.testing.assert_array_equal(new['query_ids'], q)
                np.testing.assert_array_equal(new['support_ids'], s)
                y = all_labels[q]
                scores = {arm: old['expert_'+arm] for arm in ('raw', 'initial', 'simple')}
                scores.update({arm: new[arm] for arm in NEW_ARMS})
                bp = dict(expert=softmax(scores['raw'], axis=1), fusion=old['fusion_raw'])
                key = dict(dataset=ds, draw=draw, budget=budget)
                for arm, score in scores.items():
                    for layer, p in [('expert', softmax(score, axis=1)), ('fusion', late(old['base'], score))]:
                        rows.append(dict(**key, arm=arm, layer=layer, **metrics(y, p, bp[layer])))
                # A fixed correct-vs-strongest-wrong margin is additive in expert scores.
                raw = scores['raw']
                wrong = raw.copy()
                wrong[np.arange(len(y)), y] = -np.inf
                rival = wrong.argmax(1)
                at = np.arange(len(y))
                def margin(x):
                    return x[at, y]-x[at, rival]
                changes = {arm: margin(score-raw) for arm, score in scores.items() if arm != 'raw'}
                interaction = changes['initial']-changes['relation_only']-changes['coefficient_only']
                f0, f1 = old['fusion_raw'].argmax(1), old['fusion_initial'].argmax(1)
                groups = {'all': np.ones(len(y), bool), 'fusion_rescue': (f0!=y)&(f1==y),
                          'fusion_harm': (f0==y)&(f1!=y)}
                for group, mask in groups.items():
                    for arm, value in {**changes, 'interaction': interaction}.items():
                        decompositions.append(dict(**key, group=group, arm=arm, n=int(mask.sum()),
                            mean_margin_change=float(value[mask].mean()) if mask.any() else np.nan))
                rq = new['retrieval_query_ids']
                for arm in ('raw', 'initial'):
                    relations.append(dict(**key, arm=arm, **retrieval(new['retrieval_'+arm],
                        all_labels[rq], old['support_labels'], mmsi[rq], mmsi[s])))
                print('EVALUATED', ds, draw, budget, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'metrics.csv', index=False)
    identifiers = ['dataset', 'draw', 'budget', 'layer']
    numbers = ['oa', 'ba', 'nll', 'brier', 'macro_brier']
    base = frame[frame.arm=='raw'][identifiers+numbers]
    paired = frame.merge(base, on=identifiers, suffixes=('', '_raw'), validate='many_to_one')
    for field in numbers:
        paired['delta_'+field] = paired[field]-paired[field+'_raw']
    paired.to_csv(OUT/'paired.csv', index=False)
    group = ['dataset', 'budget', 'layer', 'arm']
    summary = paired.groupby(group).agg(oa=('oa', 'mean'), ba=('ba', 'mean'),
        delta_oa=('delta_oa', 'mean'), delta_ba=('delta_ba', 'mean'),
        delta_ba_sd=('delta_ba', 'std'), positive_ba_draws=('delta_ba', lambda x: int((x>1e-10).sum())),
        delta_macro_brier=('delta_macro_brier', 'mean'), rescue=('rescue', 'mean'), harm=('harm', 'mean')).reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    relation = pd.DataFrame(relations)
    relation.to_csv(OUT/'retrieval.csv', index=False)
    relation.groupby(['dataset', 'budget', 'arm']).mean(numeric_only=True).to_csv(OUT/'retrieval_summary.csv')
    pd.DataFrame(decompositions).to_csv(OUT/'margin_paths.csv', index=False)
    c.write(OUT/'COMPLETE.json', dict(status='complete', frozen_prediction_units=30,
        fits=0, query_labels_used_for_training_or_selection=False,
        support_draws=5, independent_training_seeds=0, tasks=DATASETS,
        budgets=c.BUDGETS, retrieval_cap=4096, source_receipt=str(OUT/'PREDICTIONS_FROZEN.json')))
    print(summary[summary.layer=='fusion'].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('freeze', 'evaluate', 'all'))
    args = parser.parse_args()
    with c.threadpool_limits(limits=2):
        if args.phase in ('freeze', 'all'):
            freeze()
        if args.phase in ('evaluate', 'all'):
            evaluate()
