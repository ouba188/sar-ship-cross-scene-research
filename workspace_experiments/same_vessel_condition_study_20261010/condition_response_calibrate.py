"""Source-held-port diagnosis and one shared out-of-fold response calibration."""

import numpy as np
import pandas as pd
from condition_response import (OUT, BASE, PARENT, PORTS, SEED, ALPHA,
    condition_basis, within_vessel, fit_effect, gains, extract_corrected)
from channel_operator import build_blocks, fit_classifier, metrics, scene_energy, save_json


def run():
    import torch
    torch.set_num_threads(1)
    dest = OUT/'source_calibration'
    dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        raise ValueError('Do not overwrite completed follow-up')
    save_json(dest/'PROTOCOL.json', dict(
        trigger='Uncalibrated condition correction harms; original whole-port relation fit is weak and CV folds uneven in row count',
        folds='Greedy source-port row-count balance into3 folds; purge held MMSIs',
        correction='One global scalar k=clip(sum(weight*predicted*observed)/sum(weight*predicted^2),0,1), estimated exclusively on source-held-port within-vessel residuals',
        no_class_specific=True, no_port_specific=True, no_target_selection=True,
        alpha='Original0.5 retained; multiply source-fitted response coefficients by calibrated k',
        boundary='OOF predictions fit calibration; calibrated OOF error is not independent confirmation'))
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(OUT/'condition_model.npz') as saved:
        meta = obs.iloc[saved['source_obs_ids']].reset_index(drop=True)
        log_mean, full_coef, reference = saved['log_channel_means'], saved['coef'], saved['reference']
    basis, _ = condition_basis(meta)
    counts = meta.groupby('port').size().sort_values(ascending=False)
    totals = np.zeros(3, dtype=int)
    mapping = {}
    for port, count in counts.items():
        fold = int(np.argmin(totals))
        mapping[port] = fold
        totals[fold] += count
    folds = meta.port.map(mapping).to_numpy()
    observed, weights = within_vessel(meta, log_mean)
    predicted = np.empty_like(observed)
    records, per_port = [], []
    for fold in range(3):
        held = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        train = train[~meta.iloc[train].mmsi.isin(set(meta.iloc[held].mmsi)).to_numpy()]
        coef, _, _ = fit_effect(meta.iloc[train].reset_index(drop=True), basis[train], log_mean[train])
        mq = meta.iloc[held].reset_index(drop=True)
        delta, ww = within_vessel(mq, basis[held])
        pp = delta@coef
        predicted[held] = pp
        yy = observed[held]
        zero = float((ww[:, None]*yy*yy).sum())
        error = float((ww[:, None]*(yy-pp)**2).sum())
        half = float((ww[:, None]*(yy-ALPHA*pp)**2).sum())
        records.append(dict(fold=fold, train=len(train), test=len(held),
            test_vessels=mq.mmsi.nunique(), held_ports=','.join(sorted(mq.port.unique())),
            prediction_R2=1-error/zero, half_reduction=1-half/zero))
        for port, g in mq.groupby('port'):
            idx = g.index.to_numpy()
            wt = ww[idx]/ww[idx].sum()
            z = float((wt[:, None]*yy[idx]**2).sum())
            e = float((wt[:, None]*(yy[idx]-pp[idx])**2).sum())
            per_port.append(dict(port=port, fold=fold, n=len(g), vessels=g.mmsi.nunique(),
                                 prediction_R2=1-e/z))
    denominator = float((weights[:, None]*predicted**2).sum())
    numerator = float((weights[:, None]*predicted*observed).sum())
    scalar = float(np.clip(numerator/max(denominator, 1e-12), 0, 1))
    zero = float((weights[:, None]*observed**2).sum())
    raw = float((weights[:, None]*(observed-predicted)**2).sum())
    calibrated = float((weights[:, None]*(observed-ALPHA*scalar*predicted)**2).sum())
    _, wb = within_vessel(meta, basis)
    bx, _ = within_vessel(meta, basis)
    condition_change = (wb[:, None]*bx**2).sum(0)
    info = dict(scalar=scalar, numerator=numerator, denominator=denominator,
        source_port_OOF_prediction_R2=1-raw/zero,
        calibration_fit_half_correction_reduction=1-calibrated/zero,
        within_vessel_condition_variances=condition_change.tolist(),
        basis_names=['incidence', 'incidence_squared', 'descending', 'incidence_x_descending', 'S1C'],
        platform_coef_norm=float(np.linalg.norm(full_coef[-1])))
    save_json(dest/'CALIBRATION.json', info)
    pd.DataFrame(records).to_csv(dest/'balanced_port_cv.csv', index=False)
    pd.DataFrame(per_port).to_csv(dest/'balanced_port_breakdown.csv', index=False)
    np.savez_compressed(dest/'source_oof.npz', observed=observed, predicted=predicted,
                        weights=weights, source_obs_ids=meta.obs_id.to_numpy())
    print('CALIBRATION', info, flush=True)
    with np.load(PARENT/'split.npz') as split:
        train, test = split['train_obs_ids'], split['test_obs_ids']
    ids = np.concatenate([train, test])
    tm, qm = obs.iloc[train].reset_index(drop=True), obs.iloc[test].reset_index(drop=True)
    all_basis, valid = condition_basis(obs.iloc[ids])
    gain = gains(all_basis, valid, full_coef*scalar, reference)
    maps = np.load(PARENT/'boundary_maps.npy', mmap_mode='r')
    features = extract_corrected(build_blocks()[1], maps, gain)
    with np.load(PARENT/'baseline.npz') as saved:
        center, scale, w0, b0, baseline = (saved[k] for k in
            ('center', 'scale', 'coef', 'bias', 'predictions'))
    n = len(train)
    w, b = fit_classifier(features[:n], tm, center, scale)
    x = (features-center)/scale/np.sqrt(512)
    results, sensitivity = [], []
    for head, ww, bb in [('fixed', w0, b0), ('refit', w, b)]:
        score = x@ww+bb
        pred = score[n:].argmax(1)
        for port in PORTS:
            q = qm.port.eq(port).to_numpy()
            y = qm.class_id.to_numpy()[q]
            before, after = baseline[q] == y, pred[q] == y
            results.append(dict(head=head, port=port, n=int(q.sum()), **metrics(pred[q], y),
                rescue=int((~before & after).sum()), harm=int((before & ~after).sum())))
        sensitivity.append(dict(head=head, source_scene_score_energy=scene_energy(score[:n], tm),
            source_BA=metrics(score[:n].argmax(1), tm.class_id.to_numpy())['BA']))
        np.savez_compressed(dest/f'{head}.npz', predictions=pred, coef=ww, bias=bb, gain=gain)
    pd.DataFrame(results).to_csv(dest/'results.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(dest/'sensitivity.csv', index=False)
    save_json(dest/'COMPLETE.json', dict(scalar=scalar, target_labels_used_for_fit=False, heads=2))
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
