"""Correct quantile interpolation bias without refitting any frozen direction."""

import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from exact_stability_readout import exact_pair_shift
from matched_stability_control import equal_domain_pair_moments
from run_matched_stability_probe import OUT, ROOT, CANDIDATE, SOURCE, ARMS, project_source
from sar_joint_completion import write_json


DEST = OUT/'exact_w1'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST/'COMPLETE.json').exists():
        raise ValueError('Preserve completed correction')
    write_json(DEST/'PROTOCOL.json', dict(date='2026-10-08',
        issue='Source midpoint interpolation vs target np.quantile yields nonzero discrepancy on identical empirical distributions',
        correction='scipy.stats.wasserstein_distance with class-normalized source domain weights and uniform target weights',
        denominator='Frozen source pooled TOTAL within-class SD; unchanged',
        fits=0, model_selection=0, labels_fitted=0,
        unchanged='All directions,source thresholds,AUC,BA,pair coverage,source calipers and original policies',
        preservation='Original approximate101quantile outputs remain in parent',
        sequence='Error found after first readout, transparently corrected without selecting a new method'))
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id', 'port'], keep_default_na=False)
    raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    old = pd.read_csv(OUT/'draw_pair_metrics.csv')
    inventory = json.loads((ROOT/'oof_base_adapt_results_20261007/POLICIES_SAVED.json').read_text(encoding='utf-8'))
    records = []
    with threadpool_limits(limits=2):
        for item in inventory:
            port = item['port']
            with np.load(OUT/'ports'/port/'frozen.npz') as saved:
                p = {k: saved[k] for k in saved.files}
            with np.load(CANDIDATE/'ports'/port/'frozen.npz') as saved:
                source_z = project_source(raw, p['source_rows'], saved['feature_center'], saved['feature_scale'],
                    np.column_stack((saved['pooled_basis'], saved['conditional_penalty_basis'])))
            sy = frame.class_id.to_numpy()[p['source_rows']]
            sd = frame.port.to_numpy()[p['source_rows']]
            ty = frame.class_id.to_numpy()[p['query_rows']]
            n = len(p['pairs'])
            for j, (c, d) in enumerate(p['pairs']):
                sz = source_z[:,[j,j+n]]
                moments = equal_domain_pair_moments(sz, sy, sd, c, d)
                views = np.column_stack((sz, sz@p['coefficients'][j]))
                for draw, ev in enumerate(p['eval_mask']):
                    if min(np.sum(ev & (ty == c)), np.sum(ev & (ty == d))) < 20:
                        continue
                    for a, arm in enumerate(ARMS):
                        shift = exact_pair_shift(views[:,a], sy, moments['weights'],
                            p['target_scores'][ev,j,a], ty[ev], c, d, p['variance'][j,a])
                        records.append(dict(port=port, draw=draw, class_c=int(c), class_d=int(d), arm=arm, **shift))
            print('EXACT W1 '+port, flush=True)
    key = ['port', 'draw', 'class_c', 'class_d', 'arm']
    replaced = ['class_c_w1', 'class_d_w1', 'conditional_w1']
    exact = pd.DataFrame(records)
    df = old.drop(columns=replaced).merge(exact, on=key, validate='one_to_one')
    if len(df) != len(old):
        raise ValueError('Exact readout changed pair coverage')
    df.to_csv(DEST/'draw_pair_metrics.csv', index=False, encoding='utf-8-sig')
    compare = old[key+replaced].merge(exact, on=key, suffixes=('_approx', '_exact'))
    compare.to_csv(DEST/'approximation_comparison.csv', index=False, encoding='utf-8-sig')
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
    pd.concat(allpairs).to_csv(DEST/'port_pair_metrics.csv', index=False, encoding='utf-8-sig')
    pd.concat(allports).to_csv(DEST/'port_means.csv', index=False, encoding='utf-8-sig')
    means = pd.concat(summaries)
    means.to_csv(DEST/'equal_port_means.csv', index=False, encoding='utf-8-sig')
    write_json(DEST/'COMPLETE.json', dict(ports=len(inventory), rows=len(df), contrasts=contrasts,
        means=means.to_dict('records'), original_source_choices_unchanged=True,
        classification_metrics_unchanged=True, fits=0, causal_identification=False))
    print(means.to_string(index=False), flush=True)
    print(json.dumps(contrasts, indent=2), flush=True)


if __name__ == '__main__':
    main()
