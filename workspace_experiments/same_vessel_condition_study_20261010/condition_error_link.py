"""Identity-held-out prediction of same-ship classification-margin changes."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import itertools
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from relational_factors import BASE, CACHE, FactorNet, encode, source_readout, design
from channel_operator import save_json, SEED

OUT = BASE/'condition_error_link_20261010'


def make_pairs(meta):
    rows = []
    for ship, group in meta.groupby('mmsi'):
        if group.class_id.nunique() != 1:
            raise ValueError('Inconsistent class within ship')
        ordered = group.assign(time=pd.to_datetime(group.acq_time, errors='raise')).sort_values(['time', 'obs_id'])
        for a, b in itertools.combinations(ordered.index, 2):
            ra, rb = meta.loc[a], meta.loc[b]
            if ra['product'] == rb['product'] or pd.Timestamp(ra.acq_time) == pd.Timestamp(rb.acq_time):
                continue
            rows.append(dict(a=a, b=b, mmsi=ship, class_id=ra.class_id,
                same_port=ra.port == rb.port, gap_days=(pd.Timestamp(rb.acq_time)-pd.Timestamp(ra.acq_time)).total_seconds()/86400))
    return pd.DataFrame(rows)


def pair_features(z, pairs, magnitude=False, interaction=True):
    a, b = z[pairs.a], z[pairs.b]
    delta = np.abs(b-a) if magnitude else b-a
    return np.c_[delta, .5*(a+b)*delta] if interaction else delta


def ship_weights(pairs):
    count = pairs.mmsi.value_counts()
    w = 1/pairs.mmsi.map(count).to_numpy(dtype=float)
    return w/w.mean()


def ridge_predict(x, y, w, query, magnitude=False):
    # Odd signed features and no intercept guarantee reversal changes only the sign.
    scale = np.maximum(np.sqrt(np.average(x*x, axis=0, weights=w)), 1e-3)
    x, query = x/scale/np.sqrt(x.shape[1]), query/scale/np.sqrt(x.shape[1])
    if magnitude:
        x, query = np.c_[x, np.ones(len(x))], np.c_[query, np.ones(len(query))]
    penalty = np.eye(x.shape[1])*10.
    if magnitude:
        penalty[-1, -1] = 0.
    beta = np.linalg.solve(x.T@(w[:, None]*x)+penalty, x.T@(w*y))
    return x@beta, query@beta


def summarize(pairs, target, pred, baseline, flips, correct_a, name, task, split):
    w = ship_weights(pairs)
    mse = np.average((pred-target)**2, weights=w)
    base = np.average((baseline-target)**2, weights=w)
    row = dict(split=split, arm=name, task=task, pairs=len(pairs), ships=pairs.mmsi.nunique(),
        MSE=mse, MAE=np.average(np.abs(pred-target), weights=w), skill_vs_constant=1-mse/base)
    if task == 'signed':
        row['direction_accuracy'] = 100*np.average((pred > 0) == (target > 0), weights=w)
        for select, label in [(np.ones(len(pairs), dtype=bool), 'all'), (flips, 'flips')]:
            truth = target[select] > 0
            ww = w[select]
            row[label+'_direction_AUC'] = roc_auc_score(truth, pred[select], sample_weight=ww) if len(np.unique(truth)) == 2 else np.nan
        row['flip_direction_accuracy'] = 100*np.average((pred[flips] > 0) == (target[flips] > 0), weights=w[flips]) if flips.any() else np.nan
        selected = correct_a
        event = flips[selected]
        row['correct_to_wrong_AUC'] = roc_auc_score(event, -pred[selected], sample_weight=w[selected]) if len(np.unique(event)) == 2 else np.nan
    return row


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    obs = pd.read_csv(BASE/'observations.csv')
    saved = np.load(BASE/'relational_factors_20261010/source_split.npz')
    ids = np.r_[saved['train_obs_ids'], saved['diagnostic_obs_ids']]
    meta = obs.iloc[ids].reset_index(drop=True)
    folds = {f: np.flatnonzero(meta.fold.eq(f)) for f in (0, 1, 2)}
    ship_sets = {f: set(meta.iloc[ix].mmsi) for f, ix in folds.items()}
    assert all(not (ship_sets[a]&ship_sets[b]) for a, b in itertools.combinations(folds, 2))
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], dtype=np.float32)
    state = source_readout(raw[folds[1]], meta.iloc[folds[1]].reset_index(drop=True))
    with torch.no_grad():
        _, scores = design(torch.from_numpy(raw), state)
    scores = scores.numpy().astype(float)
    labels = meta.class_id.to_numpy()
    other = scores.copy()
    other[np.arange(len(other)), labels] = -np.inf
    margin = scores[np.arange(len(scores)), labels]-other.max(1)
    correct = scores.argmax(1) == labels
    pairs = {name: make_pairs(meta.iloc[folds[f]]) for name, f in [('train', 2), ('test', 0)]}
    raw_mean = raw[folds[2]].mean(0)
    _, _, vt = np.linalg.svd(raw[folds[2]]-raw_mean, full_matrices=False)
    features = dict(raw512=raw, raw_pca32=(raw-raw_mean)@vt[:32].T)
    norm = ((raw-saved['center'])/saved['scale']).astype(np.float32)
    for arm, name in [('task_only', 'ordinary32'), ('relations', 'condition32')]:
        model = FactorNet()
        model.load_state_dict(torch.load(BASE/'relational_factors_20261010'/f'{arm}.pt', map_location='cpu', weights_only=True))
        parts = encode(model.eval(), norm)
        features[name] = parts[1]
        if arm == 'relations':
            features['stable128'] = parts[0]
    # No true class is supplied to the predictor. Confidence is a decision-only control.
    sorted_scores = np.sort(scores, axis=1)
    features['confidence2'] = np.c_[sorted_scores[:, -1], sorted_scores[:, -1]-sorted_scores[:, -2]]
    results, predictions, split_rows = [], [], []
    for name, pp in pairs.items():
        pp['delta_margin'] = margin[pp.b]-margin[pp.a]
        pp['correct_a'], pp['correct_b'] = correct[pp.a], correct[pp.b]
        pp['flip'] = pp.correct_a != pp.correct_b
        pp['obs_a'], pp['obs_b'] = meta.iloc[pp.a].obs_id.to_numpy(), meta.iloc[pp.b].obs_id.to_numpy()
        pp.to_csv(OUT/f'{name}_pairs.csv', index=False)
        split_rows.append(dict(split=name, chips=len(folds[2 if name == 'train' else 0]), pairs=len(pp), ships=pp.mmsi.nunique(),
            flips=int(pp.flip.sum()), worsening=int((pp.correct_a&~pp.correct_b).sum()), improving=int((~pp.correct_a&pp.correct_b).sum()),
            same_port_pairs=int(pp.same_port.sum())))
    train, test = pairs['train'], pairs['test']
    w = ship_weights(train)
    for task in ('signed', 'magnitude'):
        yt, yv = train.delta_margin.to_numpy(), test.delta_margin.to_numpy()
        if task == 'magnitude':
            yt, yv = np.abs(yt), np.abs(yv)
        constant = 0. if task == 'signed' else np.average(yt, weights=w)
        bases = dict(train=np.full(len(train), constant), test=np.full(len(test), constant))
        arms = {'constant': (bases['train'], bases['test'])}
        for name, z in features.items():
            center, scale = z[folds[2]].mean(0), np.maximum(z[folds[2]].std(0), 1e-3)
            z = (z-center)/scale
            for interaction in (False, True):
                xt, xv = [pair_features(z, p, task == 'magnitude', interaction) for p in (train, test)]
                fitted, predicted = ridge_predict(xt, yt, w, xv, task == 'magnitude')
                if task == 'magnitude':
                    fitted, predicted = np.maximum(fitted, 0), np.maximum(predicted, 0)
                arms[name+('_interaction' if interaction else '_linear')] = (fitted, predicted)
        for arm, (pt, pv) in arms.items():
            for split, pp, yy, pred in [('train', train, yt, pt), ('test', test, yv, pv)]:
                results.append(summarize(pp, yy, pred, bases[split], pp.flip.to_numpy(), pp.correct_a.to_numpy(), arm, task, split))
                for i in range(len(pp)):
                    predictions.append(dict(split=split, task=task, arm=arm, pair=i, mmsi=pp.iloc[i].mmsi,
                        target=yy[i], prediction=pred[i], base=bases[split][i]))
        print(task, pd.DataFrame(results).query("split == 'test' and task == @task").to_string(index=False), flush=True)
    pd.DataFrame(results).to_csv(OUT/'metrics.csv', index=False)
    pred = pd.DataFrame(predictions)
    pred.to_csv(OUT/'predictions.csv', index=False)
    pd.DataFrame(split_rows).to_csv(OUT/'pair_coverage.csv', index=False)
    meta.groupby(['fold', 'class_id']).agg(chips=('mmsi', 'size'), ships=('mmsi', 'nunique')).to_csv(OUT/'class_coverage.csv')
    intervals = []
    for task in ('signed', 'magnitude'):
        sub = pred.query("split == 'test' and task == @task").copy()
        sub['squared_error'] = (sub.prediction-sub.target)**2
        by_ship = sub.groupby(['mmsi', 'arm']).squared_error.mean().unstack()
        rng = np.random.default_rng(SEED)
        draws = rng.integers(len(by_ship), size=(2000, len(by_ship)))
        for mode in ('linear', 'interaction'):
            for reference in ('constant', 'ordinary32_'+mode):
                diff = (by_ship['condition32_'+mode]-by_ship[reference]).to_numpy()
                low, high = np.quantile(diff[draws].mean(1), [.025, .975])
                intervals.append(dict(task=task, model='condition32_'+mode, reference=reference,
                    delta_MSE=diff.mean(), ci_low=low, ci_high=high, ships=len(diff)))
    pd.DataFrame(intervals).to_csv(OUT/'ship_bootstrap.csv', index=False)
    save_json(OUT/'protocol.json', dict(classifier_fold=1, error_predictor_fold=2, evaluation_fold=0,
        ridge=10, classifier_chips=len(folds[1]), classifier_ships=len(ship_sets[1]),
        target_labels_used=False, encoder_trained_on_folds=[1, 2],
        held_test_scenes_seen_in_encoder=int(meta.iloc[folds[0]]['product'].isin(set(meta.iloc[np.r_[folds[1], folds[2]]]['product'])).sum()),
        notes='Diagnostic only; pretrained backbone exposure inherited. Fixed head, temporal pairs, equal ship weighting. Raw-feature controls can reproduce their own classification scores; not causal evidence.'))


if __name__ == '__main__':
    main()
