"""Class-independent, within-vessel condition response calibration on CPU."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from channel_operator import (BASE, OUT as PARENT, CACHE, PORTS, BATCH, SEED,
    build_blocks, fit_classifier, metrics, scene_energy, save_json)

OUT = BASE/'condition_response_20261010'
EPS = 1e-4
ALPHA = .5


def condition_basis(meta):
    angle = meta.incidence_deg.to_numpy(dtype=float)
    passes = meta.pass_name.fillna('').str.lower()
    platforms = meta.platform.fillna('')
    valid = np.isfinite(angle) & passes.isin(['ascending', 'descending']).to_numpy()
    valid &= platforms.isin(['S1A', 'S1C']).to_numpy()
    a = (np.nan_to_num(angle, nan=40.)-40.)/10.
    d = passes.eq('descending').to_numpy(dtype=float)
    c = platforms.eq('S1C').to_numpy(dtype=float)
    basis = np.column_stack([a, a*a, d, a*d, c])
    basis[~valid] = 0.
    return basis, valid


def within_vessel(meta, values):
    meta = meta.reset_index(drop=True)
    result = np.empty_like(values, dtype=float)
    weights = np.zeros(len(meta))
    groups = list(meta.groupby(['port', 'mmsi']).indices.items())
    port_counts = pd.Series([key[0] for key, _ in groups]).value_counts()
    for (port, _), idx in groups:
        result[idx] = values[idx]-values[idx].mean(0)
        weights[idx] = 1./(len(port_counts)*port_counts[port]*len(idx))
    return result, weights


def fit_effect(meta, basis, log_mean):
    bx, weights = within_vessel(meta, basis)
    by, _ = within_vessel(meta, log_mean)
    gram = bx.T@(weights[:, None]*bx)
    penalty = .05*np.trace(gram)/basis.shape[1]
    coef = np.linalg.solve(gram+max(penalty, 1e-10)*np.eye(basis.shape[1]),
                           bx.T@(weights[:, None]*by))
    reference = np.average(basis, axis=0, weights=weights)
    return coef, reference, dict(groups=meta.groupby(['port', 'mmsi']).ngroups,
        vessels=meta.mmsi.nunique(), ports=meta.port.nunique(), observations=len(meta),
        condition_gram_rank=int(np.linalg.matrix_rank(gram)), ridge=float(penalty))


def gains(basis, valid, coef, reference):
    delta = (basis-reference)@coef
    raw_log_gain = -ALPHA*delta
    raw_log_gain[~valid] = 0.
    return np.exp(np.clip(raw_log_gain, -np.log(2.), np.log(2.))).astype(np.float32)


def repeat_source(obs):
    target = obs.port.isin(PORTS)
    source = obs.loc[~target & ~obs.mmsi.isin(set(obs.loc[target, 'mmsi']))].copy()
    _, valid = condition_basis(source)
    source = source.loc[valid]
    counts = source.groupby(['port', 'mmsi']).mmsi.transform('size')
    return source.loc[counts.ge(2)].copy()


def extract_log_means(block, rows):
    import torch
    cached = np.load(CACHE/'source_layer3.npy', mmap_mode='r')
    result = np.empty((len(rows), 512), np.float32)
    with torch.inference_mode():
        for lo in range(0, len(rows), BATCH):
            x = torch.from_numpy(np.array(cached[rows[lo:lo+BATCH]], dtype=np.float32))
            mu = block(x).mean((2, 3)).numpy()
            result[lo:lo+len(x)] = np.log(np.maximum(mu, EPS))
            if lo % 1024 == 0:
                print('REPEATED_SOURCE', lo, '/', len(rows), flush=True)
    return result


def assess_effect(meta, basis, log_mean):
    # Identity-held folds assess relation fit; whole-port folds assess portability.
    records, channels = [], []
    assignments = dict(identity=meta.fold.to_numpy())
    ports = np.array(sorted(meta.port.unique()))
    ordered = np.random.default_rng(SEED).permutation(ports)
    mapping = {port: i % 3 for i, port in enumerate(ordered)}
    assignments['port'] = meta.port.map(mapping).to_numpy()
    for split, folds in assignments.items():
        for fold in range(3):
            held = np.flatnonzero(folds == fold)
            train = np.flatnonzero(folds != fold)
            train = train[~meta.iloc[train].mmsi.isin(set(meta.iloc[held].mmsi)).to_numpy()]
            mt, mq = meta.iloc[train].reset_index(drop=True), meta.iloc[held].reset_index(drop=True)
            coef, _, _ = fit_effect(mt, basis[train], log_mean[train])
            bx, weights = within_vessel(mq, basis[held])
            by, _ = within_vessel(mq, log_mean[held])
            predicted = bx@coef
            zero_error = (weights[:, None]*by*by).sum(0)
            error = (weights[:, None]*(by-predicted)**2).sum(0)
            corrected_error = (weights[:, None]*(by-ALPHA*predicted)**2).sum(0)
            records.append(dict(split=split, fold=fold, train=len(train), test=len(held),
                held_ports=','.join(sorted(mq.port.unique())),
                prediction_R2=float(1-error.sum()/zero_error.sum()),
                half_correction_variation_reduction=float(1-corrected_error.sum()/zero_error.sum()),
                positive_channels=int((error < zero_error).sum()),
                identity_overlap=len(set(mt.mmsi) & set(mq.mmsi))))
            for c in range(512):
                channels.append(dict(split=split, fold=fold, channel=c,
                    prediction_R2=float(1-error[c]/max(zero_error[c], 1e-12)),
                    original_variance=float(zero_error[c])))
    pd.DataFrame(records).to_csv(OUT/'source_generalization.csv', index=False)
    pd.DataFrame(channels).to_csv(OUT/'source_channel_generalization.csv', index=False)
    print('SOURCE_CV', json.dumps(records), flush=True)


def extract_corrected(block, maps, gain):
    import torch
    result = np.empty((len(maps), 512), np.float32)
    with torch.inference_mode():
        for lo in range(0, len(maps), BATCH):
            x = torch.from_numpy(np.array(maps[lo:lo+BATCH]))
            g = torch.from_numpy(gain[lo:lo+len(x)])[:, :, None, None]
            result[lo:lo+len(x)] = block(x*g).mean((2, 3)).numpy()
    return result


def run():
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Completed outputs must not be overwritten')
    started = time.perf_counter()
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(PARENT/'split.npz') as split:
        train, test = split['train_obs_ids'], split['test_obs_ids']
    ids = np.concatenate([train, test])
    tm, qm = obs.iloc[train].reset_index(drop=True), obs.iloc[test].reset_index(drop=True)
    repeats = repeat_source(obs).reset_index(drop=True)
    save_json(OUT/'PROTOCOL.json', dict(
        objective='Retain channels and spatial arrangement; correct only condition-predicted amplitude changes',
        inputs='incidence, incidence squared, pass, incidence*pass, platform; missing condition -> unchanged',
        estimator='Within same MMSI and port demeaned regression on log channel spatial means; common coefficients across classes and ports',
        weighting='Ports equal; vessels equal within port; observations equal within vessel/port',
        labels='No ship-type labels in response estimator or gain generation; same939 labels for all classification heads',
        additional_source_information='Repeated-vessel source observations and acquisition metadata; extra identity information is explicit',
        source_repeats=len(repeats), classification_train=len(train), test=len(test),
        ridge='Fixed0.05*trace(within-vessel condition Gram)/5', alpha=ALPHA,
        gain_bounds=[.5, 2.], basis_reference='Weighted source repeated-vessel condition mean',
        controls=['baseline', 'sample_mean_shrink', 'condition_gain', 'permuted_channel_gain'],
        head_controls='Original fixed head versus source-refit head on each corrected representation',
        permutation='One fixed source-seed channel reassignment of condition coefficients; identical per-sample gain multiset',
        source_cv='Three identity folds and three whole-port folds; purge validation identities from train',
        target_protocol='Same three jointly held-out ports, no target fitting or target-specific hyperparameters',
        boundaries='Geometry-associated response only; not all port/sea-state factors or causal noise identified; no main64-label result',
        seed=SEED, resources='CPU1, cached layer3/boundary, no main files or GPU modified'))
    blocks = build_blocks()
    source_log = extract_log_means(blocks[0], repeats.manifest_row.to_numpy())
    basis, valid = condition_basis(repeats)
    assert valid.all()
    assess_effect(repeats, basis, source_log)
    coef, reference, fit_info = fit_effect(repeats, basis, source_log)
    np.savez_compressed(OUT/'condition_model.npz', coef=coef, reference=reference,
        source_obs_ids=repeats.obs_id.to_numpy(), log_channel_means=source_log)
    save_json(OUT/'FIT.json', fit_info)
    maps = np.load(PARENT/'boundary_maps.npy', mmap_mode='r')
    n = len(train)
    mu = np.asarray(maps).mean((2, 3))
    log_mu = np.log(np.maximum(mu, EPS))
    all_basis, all_valid = condition_basis(obs.iloc[ids])
    dynamic = gains(all_basis, all_valid, coef, reference)
    permutation = np.random.default_rng(SEED+211).permutation(512)
    sample = np.exp(np.clip(-ALPHA*(log_mu-log_mu[:n].mean(0)), -np.log(2.), np.log(2.))).astype(np.float32)
    # Controls share availability so missing metadata is not a between-arm difference.
    sample[~all_valid] = 1.
    variants = dict(baseline=np.ones_like(dynamic), sample_mean_shrink=sample,
                    condition_gain=dynamic, permuted_channel_gain=dynamic[:, permutation])
    np.savez_compressed(OUT/'gains.npz', **variants, available=all_valid, obs_ids=ids,
                        permutation=permutation)
    with np.load(PARENT/'baseline.npz') as saved:
        center, scale, old_w, old_b, original_pred = (saved[k] for k in
            ('center', 'scale', 'coef', 'bias', 'predictions'))
    results, classes, sensitivities, accounting = [], [], [], []
    replay = None
    for variant, gain in variants.items():
        features = extract_corrected(blocks[1], maps, gain)
        w, b = fit_classifier(features[:n], tm, center, scale)
        x = (features-center)/scale/np.sqrt(512)
        for head, ww, bb in [('fixed', old_w, old_b), ('refit', w, b)]:
            score = x@ww+bb
            pred = score[n:].argmax(1)
            if variant == 'baseline':
                disagreements = int((pred != original_pred).sum())
                if disagreements:
                    raise ValueError('Original baseline prediction mismatch')
                replay = disagreements
            arm = variant+'__'+head
            sensitivities.append(dict(arm=arm,
                source_scene_score_energy=scene_energy(score[:n], tm),
                source_BA=metrics(score[:n].argmax(1), tm.class_id.to_numpy())['BA']))
            for port in PORTS:
                q = qm.port.eq(port).to_numpy()
                y = qm.class_id.to_numpy()[q]
                before, after = original_pred[q] == y, pred[q] == y
                results.append(dict(arm=arm, port=port, n=int(q.sum()), **metrics(pred[q], y),
                    rescue=int((~before & after).sum()), harm=int((before & ~after).sum())))
                for c in np.unique(y):
                    cc = y == c
                    classes.append(dict(arm=arm, port=port, class_id=int(c), n=int(cc.sum()),
                        recall=float(100*after[cc].mean()), rescue=int((~before & after & cc).sum()),
                        harm=int((before & ~after & cc).sum())))
            np.savez_compressed(OUT/f'{arm}.npz', predictions=pred, source_scores=score[:n],
                coef=ww, bias=bb, center=center, scale=scale)
            print('RESULT', arm, json.dumps(results[-3:]), flush=True)
        for port in ('source',)+PORTS:
            q = np.arange(len(ids)) < n if port == 'source' else obs.iloc[ids].port.eq(port).to_numpy()
            gs = gain[q]
            accounting.append(dict(variant=variant, port=port, n=int(q.sum()),
                available=int(all_valid[q].sum()), mean_abs_log_gain=float(np.abs(np.log(gs)).mean()),
                gain_q01=float(np.quantile(gs, .01)), gain_q99=float(np.quantile(gs, .99)),
                capped_fraction=float(np.mean((gs <= .500001) | (gs >= 1.999999)))))
        pd.DataFrame(results).to_csv(OUT/'results.csv', index=False)
        pd.DataFrame(classes).to_csv(OUT/'by_class.csv', index=False)
        pd.DataFrame(sensitivities).to_csv(OUT/'sensitivity.csv', index=False)
        pd.DataFrame(accounting).to_csv(OUT/'gain_accounting.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(variants=list(variants), prediction_rows=len(test),
        classification_train=len(train), response_fit_rows=len(repeats),
        original_prediction_disagreements=replay, seconds=time.perf_counter()-started))


if __name__ == '__main__':
    run()
