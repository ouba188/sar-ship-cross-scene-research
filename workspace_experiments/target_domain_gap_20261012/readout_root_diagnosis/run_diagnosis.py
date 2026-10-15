"""Separate source reweighting and target evidence in fixed-feature readouts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment import OUT as PARENT, TASKS, SEEDS, softmax, residual, write
from report import metric
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
SUPPORT = PARENT / 'support_set_evaluation'
DRAWS = (1201, 1202, 1203, 1204, 1205)
ARMS = ('B', 'R', 'T', 'F', 'U', 'UT', 'G', 'C')
CONTRASTS = {
    'full_gain': ('B', 'F'),
    'source_reweight_alone': ('B', 'R'),
    'target_with_original_source': ('B', 'T'),
    'target_given_reweight': ('R', 'F'),
    'source_reweight_given_target': ('T', 'F'),
    'source_mass_only': ('B', 'U'),
    'source_class_weights_at_matched_mass': ('U', 'R'),
    'source_class_weights_with_target': ('UT', 'F'),
    'target_with_matched_source_mass': ('U', 'UT'),
    'target_gram_only': ('R', 'G'),
    'target_cross_stat_only': ('R', 'C'),
}


def weights(y, mass):
    w = 1. / np.bincount(y, minlength=8)[y]
    return w * (mass / w.sum())


def sufficient(h, y, w):
    target = 2. * np.eye(8)[y] - 1.
    return h.T @ (w[:, None] * h), h.T @ (w[:, None] * target)


def solve(g, c):
    penalty = np.eye(g.shape[0])
    penalty[-1, -1] = 0.
    return np.linalg.solve(g + penalty, c)


def scored(y, prob, ships):
    result, recalls = metric(y, prob)
    pred = prob.argmax(1)
    ship = pd.DataFrame(dict(ship=ships, y=y, correct=(pred == y)))
    ship = ship.groupby(['ship', 'y'], as_index=False).correct.mean()
    result.update(
        ship_oa=100. * float(ship.correct.mean()),
        ship_ba=100. * float(ship.groupby('y').correct.mean().mean()),
        nll=float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-15, 1.)).mean()),
        brier=float(np.sum((prob - np.eye(8)[y]) ** 2, axis=1).mean()),
    )
    return result, recalls


def summarize(metrics, paired, factors):
    for name, data, keys, columns in (
        ('metrics', metrics, ['task', 'stage', 'arm'],
         ['oa', 'ba', 'ship_oa', 'ship_ba', 'nll', 'brier']),
        ('contrasts', paired, ['task', 'stage', 'comparison'],
         ['delta_oa', 'delta_ba', 'delta_nll', 'delta_brier']),
        ('factors', factors, ['task', 'stage', 'metric'],
         ['source', 'target', 'interaction', 'total']),
    ):
        records = []
        for group, part in data.groupby(keys, sort=False):
            record = dict(zip(keys, group))
            record['draws'] = len(part)
            for col in columns:
                record[col] = float(part[col].mean())
                variances = part.groupby('split_seed')[col].var(ddof=1)
                record[col + '_support_sd'] = float(np.sqrt(variances.mean()))
                record[col + '_min'] = float(part[col].min())
                record[col + '_max'] = float(part[col].max())
                if name == 'contrasts':
                    record[col + '_positive'] = int((part[col] > 1e-10).sum())
                    record[col + '_negative'] = int((part[col] < -1e-10).sum())
                    record[col + '_ties'] = int((part[col].abs() <= 1e-10).sum())
            records.append(record)
        pd.DataFrame(records).to_csv(OUT / (name + '_summary.csv'), index=False)


def freeze():
    frame = pd.read_csv(PARENT / 'manifest.csv', dtype={'mmsi': str})
    y = frame.class_id.to_numpy()
    audits, inventory, innovations = [], [], []
    replay_max = 0.
    for task in TASKS:
        for seed in SEEDS:
            parent = PARENT / task / str(seed)
            split = dict(np.load(parent / 'split.npz'))
            z = dict(np.load(parent / 'features.npz'))
            loc = pd.Index(z.pop('rows'))
            features = {'r50': z['r50'], 'dual': np.c_[z['r50'], z['r18']] / np.sqrt(2.)}
            train = loc.get_indexer(split['source'])
            query = split['query']
            qi = loc.get_indexer(query)
            ys = y[split['source']]
            w0 = weights(ys, len(ys))
            cache = {}
            for mode, x in features.items():
                h = np.c_[x.astype(np.float64), np.ones(len(x))]
                g0, c0 = sufficient(h[train], ys, w0)
                cache[mode] = (h, g0, c0, solve(g0, c0))
            for draw in DRAWS:
                with np.load(SUPPORT / task / str(seed) / str(draw) / 'predictions.npz') as old:
                    np.testing.assert_array_equal(query, old['query_rows'])
                    support = old['support_rows'].copy()
                    reference = {'B': old['original_residual64'].copy(),
                                 'F': old['global64_residual64'].copy()}
                si = loc.get_indexer(support)
                if min(train.min(), qi.min(), si.min()) < 0:
                    raise ValueError('An existing observation is missing from feature cache')
                yt = y[support]
                joint = weights(np.r_[ys, yt], len(ys))
                ws, wt = joint[:len(ys)], joint[len(ys):]
                ratio = ws.sum() / w0.sum()
                meta = dict(task=task, split_seed=seed, draw_seed=draw)
                inventory.append(dict(**meta, source_mass=float(ws.sum()),
                    target_mass=float(wt.sum()), source_uniform_scale=float(ratio),
                    source_weight_min=float(ws.min()), source_weight_max=float(ws.max()),
                    **{f'support_class_{c}': int(np.sum(yt == c)) for c in range(8)}))
                outputs, head_arrays = {}, {}
                by_mode = {}
                for mode, (h, g0, c0, theta0) in cache.items():
                    gs, cs = sufficient(h[train], ys, ws)
                    gt, ct = sufficient(h[si], yt, wt)
                    heads = {
                        'B': theta0,
                        'R': solve(gs, cs),
                        'T': solve(g0 + gt, c0 + ct),
                        'F': solve(gs + gt, cs + ct),
                        'U': solve(ratio * g0, ratio * c0),
                        'UT': solve(ratio * g0 + gt, ratio * c0 + ct),
                        'G': solve(gs + gt, cs),
                        'C': solve(gs, cs + ct),
                    }
                    target_residual = 2. * np.eye(8)[yt] - 1. - h[si] @ heads['R']
                    update = solve(gs + gt, h[si].T @ (wt[:, None] * target_residual))
                    identity_error = float(np.max(np.abs(heads['F'] - heads['R'] - update)))
                    if identity_error > 1e-10:
                        raise ValueError('Target innovation does not reconstruct the joint update')
                    innovations.append(dict(**meta, mode=mode, identity_error=identity_error,
                        weighted_support_error_before=float(np.sum(wt[:, None] * target_residual**2) / wt.sum()),
                        weighted_support_error_after=float(np.sum(wt[:, None] *
                            (2. * np.eye(8)[yt] - 1. - h[si] @ heads['F'])**2) / wt.sum())))
                    by_mode[mode] = {}
                    for arm, theta in heads.items():
                        head_arrays[mode + '_' + arm] = theta
                        prior = h @ theta
                        adapted = residual(h[:, :-1], prior, si, yt)
                        by_mode[mode]['prior_' + arm] = prior[qi]
                        by_mode[mode]['residual_' + arm] = adapted[qi]
                for name in by_mode['r50']:
                    outputs[name] = .5 * softmax(by_mode['r50'][name]) + .5 * softmax(by_mode['dual'][name])
                for arm, previous in reference.items():
                    current = outputs['residual_' + arm]
                    error = float(np.max(np.abs(previous - current)))
                    same = bool(np.array_equal(previous.argmax(1), current.argmax(1)))
                    audits.append(dict(**meta, arm=arm, max_probability_error=error, same_predictions=same))
                    replay_max = max(replay_max, error)
                    if error > 2e-6 or not same:
                        raise ValueError(f'Endpoint replay failed: {meta}, {arm}, {error}, {same}')
                dest = OUT / task / str(seed) / str(draw)
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest / 'predictions.npz', query_rows=query, support_rows=support, **outputs)
                np.savez_compressed(dest / 'heads.npz', **head_arrays)
            print('FROZEN', task, seed, flush=True)
    pd.DataFrame(audits).to_csv(OUT / 'endpoint_replay.csv', index=False)
    pd.DataFrame(inventory).to_csv(OUT / 'weight_inventory.csv', index=False)
    pd.DataFrame(innovations).to_csv(OUT / 'target_innovation_identity.csv', index=False)
    write(OUT / 'PREDICTIONS_FROZEN.json', dict(support_sets=40, classifier_solutions=576,
        identity_check_solves=80, neural_fits=0, labels_per_draw=64,
        max_endpoint_probability_error=replay_max, all_endpoint_classes_reproduced=True))


def evaluate():
    frame = pd.read_csv(PARENT / 'manifest.csv', dtype={'mmsi': str})
    metrics, classes, paired, factors, events, class_pairs = [], [], [], [], [], []
    for task in TASKS:
        for seed in SEEDS:
            for draw in DRAWS:
                folder = OUT / task / str(seed) / str(draw)
                with np.load(folder / 'predictions.npz') as z:
                    rows, support = z['query_rows'], z['support_rows']
                    outputs = {k: z[k] for k in z.files if k not in ('query_rows', 'support_rows')}
                query = frame.iloc[rows]
                y, ships = query.class_id.to_numpy(), query.mmsi.to_numpy()
                meta = dict(task=task, split_seed=seed, draw_seed=draw)
                counts = np.bincount(frame.iloc[support].class_id, minlength=8)
                for stage in ('prior', 'residual'):
                    result = {}
                    for arm in ARMS:
                        prob = outputs[stage + '_' + arm]
                        mm, recalls = scored(y, prob, ships)
                        result[arm] = mm
                        metrics.append(dict(**meta, stage=stage, arm=arm, **mm))
                        for c in range(8):
                            classes.append(dict(**meta, stage=stage, arm=arm, class_id=c,
                                support_count=int(counts[c]), query_count=int(np.sum(y == c)),
                                recall=100. * recalls[c]))
                    for name, (a, b) in CONTRASTS.items():
                        pa, pb = (outputs[stage + '_' + arm].argmax(1) for arm in (a, b))
                        paired.append(dict(**meta, stage=stage, comparison=name,
                            **{'delta_' + k: result[b][k] - result[a][k] for k in ('oa', 'ba', 'nll', 'brier')},
                            rescues=int(np.sum((pa != y) & (pb == y))),
                            harms=int(np.sum((pa == y) & (pb != y)))))
                    for score in ('oa', 'ba', 'nll', 'brier'):
                        b, r, t, f = (result[arm][score] for arm in ('B', 'R', 'T', 'F'))
                        source = .5 * ((r-b) + (f-t))
                        target = .5 * ((t-b) + (f-r))
                        if abs(source + target - (f-b)) > 1e-10:
                            raise ValueError('Factor contributions do not sum to total gain')
                        factors.append(dict(**meta, stage=stage, metric=score,
                            source=source, target=target, total=f-b, interaction=f-r-t+b))
                a, b = outputs['residual_B'], outputs['residual_F']
                pa, pb = a.argmax(1), b.argmax(1)
                rescue, harm = (pa != y) & (pb == y), (pa == y) & (pb != y)
                for i in np.flatnonzero(pa != pb):
                    other = a[i].copy()
                    other[y[i]] = -np.inf
                    rival = int(other.argmax())
                    events.append(dict(**meta, row=int(rows[i]), mmsi=ships[i],
                        product=query.iloc[i]['product'], true_class=int(y[i]),
                        before=int(pa[i]), after=int(pb[i]),
                        event='rescue' if rescue[i] else ('harm' if harm[i] else 'wrong_to_wrong'),
                        rival=rival, margin_before=float(a[i,y[i]]-a[i,rival]),
                        margin_after=float(b[i,y[i]]-b[i,rival])))
                for c in range(8):
                    denom = int(np.sum(y == c))
                    for k in range(8):
                        if c == k:
                            continue
                        rr = (y == c) & (pa == k) & rescue
                        hh = (y == c) & (pb == k) & harm
                        class_pairs.append(dict(**meta, true_class=c, competitor=k, query_count=denom,
                            rescues=int(rr.sum()), harms=int(hh.sum()),
                            net_ba=100. * (int(rr.sum())-int(hh.sum())) / (8*denom)))
    m, p, f = pd.DataFrame(metrics), pd.DataFrame(paired), pd.DataFrame(factors)
    for name, data in [('metrics', m), ('contrasts', p), ('factors', f),
                       ('class_recalls', pd.DataFrame(classes)), ('events', pd.DataFrame(events)),
                       ('class_pairs', pd.DataFrame(class_pairs))]:
        data.to_csv(OUT / (name + '.csv'), index=False)
    summarize(m, p, f)
    print('RESIDUAL OA/BA', flush=True)
    print(m.loc[m.stage.eq('residual')].groupby(['task', 'arm'], sort=False)[['oa','ba']].mean().to_string(), flush=True)
    print('FACTOR ATTRIBUTION', flush=True)
    print(f.loc[f.stage.eq('residual') & f.metric.isin(['oa','ba'])].groupby(['task','metric'])[
        ['source','target','interaction','total']].mean().to_string(), flush=True)
    write(OUT / 'COMPLETE.json', dict(support_sets=40, classification_metrics=len(m),
        stages=['prior','residual'], arms=list(ARMS), main_model_changed=False,
        query_labels_used_only_for_evaluation=True, new_neural_fits=0))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        freeze()
        evaluate()
