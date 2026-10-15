"""Oracle diagnostics and margin accounting, never used for model selection."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from frequency_reliability_run import OUT, EXP, ROOT, SOURCE, DATASETS, load_npz, statistics
from frequency_reliability_core import point_ce


def auc(label, value):
    return float(roc_auc_score(label, value)) if len(np.unique(label)) == 2 else np.nan


def average(value, mask):
    return float(np.mean(value[mask])) if np.any(mask) else np.nan


def main():
    source_y = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    records, oracle, margins, pooled = [], [], [], {}
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        folder = marker.parent
        info = json.loads((folder/'PREPARED.json').read_text())
        p = load_npz(OUT/'targets'/info['dataset']/info['task']/'policies.npz')
        y = source_y[p['rows']] if info['dataset'] == 'self24ports' else pd.read_csv(
            ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        with np.load(folder/'policies.npz') as saved:
            ix = [list(saved['modes']).index(m) for m in ('baseline', 'selective')]
            scores = saved['scores'][:, ix].astype(float)
        for draw, ev in enumerate(p['eval_masks']):
            meta = dict(dataset=info['dataset'], task=info['task'], draw=draw)
            gamma = p['local_gamma'][draw]
            s0, s1 = scores[draw]
            gain = point_ce(s0, y)-point_ce(s1, y)
            for stage, key in [('base', 'base_predictions'), ('full', 'predictions')]:
                old, new, local = p[key][draw, 0], p[key][draw, 1], p[key][draw, 4]
                rescue, harm = ev & (old != y) & (new == y), ev & (old == y) & (new != y)
                changed = rescue | harm
                records.append(dict(**meta, stage=stage,
                    gamma_rescue=average(gamma, rescue), gamma_harm=average(gamma, harm),
                    gamma_flip_auc=auc(rescue[changed], gamma[changed]),
                    ce_utility_flip_auc=auc(rescue[changed], gain[changed]),
                    rescue_ce_positive=average((gain > 0).astype(float), rescue),
                    harm_ce_negative=average((gain < 0).astype(float), harm),
                    rescues=int(rescue.sum()), harms=int(harm.sum()),
                    lost_rescues=int(np.sum(rescue & (local != y))),
                    repaired_harms=int(np.sum(harm & (local == y)))))
                pooled.setdefault((info['dataset'], stage), []).append(
                    (rescue[changed], gamma[changed], gain[changed]))
            # Endpoint oracles isolate base score use. Full rerouting is not oracle-fitted.
            for stage, key in [('base', 'base_predictions'), ('frozen_route', 'frozen_predictions')]:
                old, new = p[key][draw, 0], p[key][draw, 1]
                ce_oracle = np.where(gain >= 0, new, old)
                hard_oracle = np.where(new == y, new, old)
                predicted_ce_endpoint = np.where(gamma > .5, new, old)
                ref_oa, ref_ba, _ = statistics(y[ev], new[ev])
                for name, pred in [('learned', new), ('local', p[key][draw, 4]),
                                   ('predicted_ce_endpoint_diagnostic', predicted_ce_endpoint),
                                   ('ce_endpoint_oracle', ce_oracle), ('hard_endpoint_oracle', hard_oracle)]:
                    oa, ba, _ = statistics(y[ev], pred[ev])
                    oracle.append(dict(**meta, stage=stage, mode=name, oa=oa, ba=ba,
                        delta_oa=oa-ref_oa, delta_ba=ba-ref_ba))
            old, new, local = p['base_predictions'][draw, 0], p['base_predictions'][draw, 1], p['base_predictions'][draw, 4]
            rescue = ev & (old != y) & (new == y)
            ids = np.flatnonzero(rescue)
            z0, dz = s0[ids], (s1-s0)[ids]
            yy = y[ids]
            m0 = z0[np.arange(len(ids)), yy, None]-z0
            dm = dz[np.arange(len(ids)), yy, None]-dz
            need = np.divide(-m0, dm, out=np.zeros_like(m0), where=(m0 < 0) & (dm > 0)).max(axis=1)
            lost = local[ids] != y[ids]
            below = gamma[ids] < need
            margins.append(dict(**meta, rescued_n=len(ids), lost_n=int(lost.sum()),
                threshold_mismatch_n=int(np.sum(lost != below)),
                required_gamma_mean=float(need.mean()) if len(ids) else np.nan,
                lost_required_gamma=average(need, lost), lost_actual_gamma=average(gamma[ids], lost),
                lost_ce_positive=average((gain[ids] > 0).astype(float), lost)))
    r, o, m = pd.DataFrame(records), pd.DataFrame(oracle), pd.DataFrame(margins)
    r.to_csv(OUT/'flip_reliability_metrics.csv', index=False)
    o.to_csv(OUT/'oracle_diagnostics.csv', index=False)
    m.to_csv(OUT/'rescue_margin_diagnostics.csv', index=False)
    fields = ['gamma_rescue', 'gamma_harm', 'gamma_flip_auc', 'ce_utility_flip_auc',
              'rescue_ce_positive', 'harm_ce_negative']
    rs = r.groupby(['dataset', 'stage', 'task'])[fields].mean().groupby(['dataset', 'stage']).mean()
    rs.to_csv(OUT/'flip_reliability_summary.csv')
    osummary = o.groupby(['dataset', 'stage', 'mode', 'task'])[['oa', 'ba', 'delta_oa', 'delta_ba']].mean().groupby(['dataset', 'stage', 'mode']).mean()
    osummary.to_csv(OUT/'oracle_summary.csv')
    pooled_rows = []
    for (dataset, stage), chunks in pooled.items():
        label, gamma, gain = [np.concatenate([a[i] for a in chunks]) for i in range(3)]
        pooled_rows.append(dict(dataset=dataset, stage=stage, n=len(label),
            gamma_flip_auc=auc(label, gamma), ce_utility_flip_auc=auc(label, gain),
            rescue_ce_positive=float(np.mean(gain[label] > 0)),
            harm_ce_negative=float(np.mean(gain[~label] < 0))))
    ps = pd.DataFrame(pooled_rows)
    ps.to_csv(OUT/'flip_reliability_pooled.csv', index=False)
    print('EQUAL TASK AUC\n'+rs.to_string(), flush=True)
    print('POOLED EVENTS\n'+ps.to_string(index=False), flush=True)
    print('ORACLE NOT DEPLOYABLE\n'+osummary.to_string(), flush=True)
    print('MARGIN\n'+m.groupby('dataset').mean(numeric_only=True).to_string(), flush=True)
    (OUT/'ROOT_CAUSE_PROTOCOL.json').write_text(json.dumps(dict(
        diagnostic_only=True, target_labels_used_for_analysis_only=True,
        no_deployment_or_selection=True, sign_threshold='.5 algebraic positive vs negative utility mass',
        ce_oracle='Use new endpoint iff true query CE improves, else old; unavailable at deployment',
        predicted_ce_endpoint='Same fixed endpoint action as CE oracle, but use gamma>.5 as predicted CE sign; diagnostic only, not added to selection',
        hard_oracle='Select correct old/new endpoint if either is correct; endpoint family upper bound only',
        precision='CE and margin diagnostics use original saved FP32 scores, original exact hard predictions retained',
        margin_threshold_mismatches=int(m.threshold_mismatch_n.sum()),
        stages='Base and fixed learned-route oracles only; no target-selected rerouting'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
