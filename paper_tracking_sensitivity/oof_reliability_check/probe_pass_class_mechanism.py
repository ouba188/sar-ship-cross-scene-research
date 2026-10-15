"""Isolate query pass effects from refitting; inspect every port/class cell."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import json
import pickle

import numpy as np
import pandas as pd

from run_direct_observation_knowledge import OUT as RUN, SOURCE, PARENT, ROOT, CACHE
from run_conditioned_vessel_drift import SIDECAR
from direct_incidence_conditioning import condition_features, predict_conditioned
from run_protected_drift_activation import make_views, gate_features
from evidence_aware_handoff import predict_gate, route_scores
from sar_joint_completion import write_json
from run_generic_transfer import metrics

OUT = RUN/'class_port_mechanism'


def freeze(frame):
    checks = []
    for folder in sorted((RUN/'ports').iterdir()):
        with np.load(folder/'policies.npz') as f:
            p = {k: f[k] for k in f.files}
        names = list(p['mode_names'])
        incidence, actual = names.index('incidence'), names.index('incidence_pass')
        parent = PARENT/'ports'/folder.name
        with (parent/'geometry.pkl').open('rb') as f:
            geometry = pickle.load(f)
        with (parent/'source_models.pkl').open('rb') as f:
            gate = pickle.load(f)['gates']['joint']
        with (folder/'incidence_pass_heads.pkl').open('rb') as f:
            heads = pickle.load(f)
        tr = p['query_rows']
        raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
        bank = np.load(parent/'target_bank.npy', mmap_mode='r')
        extra = np.load(ROOT/'module_completion_20261008/ports_projection'/folder.name/'matched_joint/joint_extra.npy', mmap_mode='r')
        views = make_views(bank, raw[tr], extra, geometry)
        u = np.c_[condition_features(frame.loc[tr, 'incidence_deg'].to_numpy()), frame.loc[tr, 'pass_value'].to_numpy()]
        arrays = {k: [] for k in ['muted_base', 'muted_expert', 'muted_pred', 'muted_selected', 'pass_effect_base', 'pass_effect_expert']}
        replay = constant_error = 0.
        for draw, ev in enumerate(p['eval_mask']):
            support = np.searchsorted(tr, p['support_rows'][draw])
            np.testing.assert_array_equal(tr[support], p['support_rows'][draw])
            labels = frame.loc[p['support_rows'][draw], 'class_id'].to_numpy()
            original = predict_conditioned(bank, views, heads, support, labels, u, [1/np.sqrt(2), 1/np.sqrt(2), 1])
            replay = max(replay, *(float(np.max(np.abs(s-p[k][draw, actual]))) for s, k in zip(original, ['base_scores', 'expert_scores'])))
            muted_u = u.copy()
            muted_u[ev, 2] = 0.
            muted = predict_conditioned(bank, views, heads, support, labels, muted_u, [1/np.sqrt(2), 1/np.sqrt(2), 1])
            forced_u = muted_u.copy()
            forced_u[ev, 2] = 1.
            forced = predict_conditioned(bank, views, heads, support, labels, forced_u, [1/np.sqrt(2), 1/np.sqrt(2), 1])
            effects = []
            for original_score, muted_score, forced_score in zip(original, muted, forced):
                effect = (forced_score-muted_score)[ev].mean(axis=0)
                constant_error = max(constant_error, float(np.max(np.abs((forced_score-muted_score)[ev]-effect))),
                    float(np.max(np.abs((original_score-muted_score)[ev]-u[ev, 2, None]*effect))))
                effects.append(effect)
            routed = route_scores(*muted, predict_gate(gate, gate_features(*muted)), labels, ev, tr)
            for key, value in zip(arrays, [*muted, routed['predictions'], routed['selected'], *effects]):
                arrays[key].append(value)
        if replay > 3e-6 or constant_error > 1e-8:
            raise ValueError(f'Pass decomposition failed: {folder.name} {replay} {constant_error}')
        np.savez_compressed(OUT/(folder.name+'.npz'), **{k: np.array(v) for k, v in arrays.items()},
            query_rows=tr, support_rows=p['support_rows'], eval_mask=p['eval_mask'],
            reference_pred=p['predictions'][:, incidence], actual_pred=p['predictions'][:, actual],
            actual_base=p['base_scores'][:, actual], actual_expert=p['expert_scores'][:, actual])
        check = dict(port=folder.name, original_score_replay_error=replay, constant_pass_effect_max_error=constant_error)
        checks.append(check)
        print(json.dumps(check), flush=True)
    pd.DataFrame(checks).to_csv(OUT/'verification.csv', index=False)
    write_json(OUT/'ALL_FROZEN.json', dict(ports=len(checks), query_truth_used=0))


def evaluate(frame):
    records, metrics_rows, confusions = [], [], []
    for path in sorted(OUT.glob('*.npz')):
        with np.load(path) as f:
            p = {k: f[k] for k in f.files}
        meta = frame.loc[p['query_rows']]
        truth, orbital = meta.class_id.to_numpy(), meta.pass_value.to_numpy()
        source = frame.loc[np.load(PARENT/'ports'/path.stem/'source_rows.npy')]
        for draw, ev in enumerate(p['eval_mask']):
            predictions = dict(reference=p['reference_pred'][draw], trained_pass_query_muted=p['muted_pred'][draw],
                               actual=p['actual_pred'][draw])
            support = frame.loc[p['support_rows'][draw]]
            for mode, pred in predictions.items():
                score = metrics(truth[ev], pred[ev], 8)
                metrics_rows.append(dict(port=path.stem, draw=draw, mode=mode, oa=score['acc'], ba=score['ba']))
            muted_base = p['muted_base'][draw].copy()
            muted_base[np.arange(len(truth)), truth] = -np.inf
            competitor = muted_base.argmax(axis=1)
            bias = p['pass_effect_base'][draw]
            margin_shift = orbital*(bias[truth]-bias[competitor])
            for cls in np.unique(truth[ev]):
                mask = ev & (truth == cls)
                n = mask.sum()
                old = predictions['reference'][mask] == cls
                new = predictions['actual'][mask] == cls
                mute = predictions['trained_pass_query_muted'][mask] == cls
                source_class = source[source.class_id.eq(cls)]
                support_class = support[support.class_id.eq(cls)]
                same_pass_supported = np.isin(orbital[mask], support_class.pass_value.to_numpy()) & (orbital[mask] != 0)
                item = dict(port=path.stem, draw=draw, class_id=int(cls), class_name=frame.loc[frame.class_id.eq(cls), 'class_name'].iloc[0],
                    n=int(n), reference_recall=float(old.mean()), actual_recall=float(new.mean()), muted_recall=float(mute.mean()),
                    total_delta_pp=float(100*(new.mean()-old.mean())),
                    query_condition_delta_pp=float(100*(new.mean()-mute.mean())),
                    refit_support_delta_pp=float(100*(mute.mean()-old.mean())),
                    rescues=int((~old & new).sum()), harms=int((old & ~new).sum()),
                    known_pass_n=int((orbital[mask] != 0).sum()), ascending_n=int((orbital[mask] == 1).sum()),
                    descending_n=int((orbital[mask] == -1).sum()), support_class_n=len(support_class),
                    support_class_known_pass_n=int(support_class.pass_value.ne(0).sum()),
                    query_same_class_pass_supported_fraction=float(same_pass_supported.mean()),
                    source_class_known_pass_fraction=float(source_class.pass_value.ne(0).mean()),
                    source_class_ascending_fraction=float(source_class.pass_value.eq(1).mean()),
                    base_query_margin_shift_mean=float(margin_shift[mask].mean()))
                records.append(item)
                for phase, before, after in [('total', predictions['reference'], predictions['actual']),
                                              ('query_condition', predictions['trained_pass_query_muted'], predictions['actual'])]:
                    for kind, changes in [('rescue', mask & (before != truth) & (after == truth)),
                                          ('harm', mask & (before == truth) & (after != truth))]:
                        wrong = before[changes] if kind == 'rescue' else after[changes]
                        for other, count in zip(*np.unique(wrong, return_counts=True)):
                            confusions.append(dict(port=path.stem, draw=draw, class_id=int(cls), phase=phase,
                                                   kind=kind, other_class=int(other), count=int(count)))
    cells = pd.DataFrame(records)
    cells.to_csv(OUT/'port_class_draws.csv', index=False)
    means = cells.groupby(['port', 'class_id', 'class_name']).mean(numeric_only=True).drop(columns='draw').reset_index()
    means.to_csv(OUT/'port_class_means.csv', index=False)
    pd.DataFrame(confusions).to_csv(OUT/'confusion_changes.csv', index=False)
    score = pd.DataFrame(metrics_rows)
    score.to_csv(OUT/'metrics.csv', index=False)
    eight = frame.groupby('port').class_id.nunique()
    tables = []
    for group, mask in [('all23', np.ones(len(score), bool)), ('all8_target12', score.port.map(eight).eq(8))]:
        table = score[mask].groupby(['port', 'mode']).mean(numeric_only=True).groupby('mode').mean().reset_index()
        table.insert(0, 'group', group)
        tables.append(table)
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(OUT/'summary.csv', index=False)
    print(summary.to_string(index=False), flush=True)
    large = means[means.n >= 20].sort_values('total_delta_pp')
    print(pd.concat([large.head(4), large.tail(4)])[['port', 'class_name', 'n', 'total_delta_pp',
        'query_condition_delta_pp', 'refit_support_delta_pp', 'base_query_margin_shift_mean']].to_string(index=False), flush=True)
    write_json(OUT/'COMPLETE.json', dict(ports=23, draws=5,
        interpretation='Computational score-effect decomposition, not a physical causal intervention',
        exact_statement='For each fitted port/draw/head, query pass adds p times a constant class-score vector.',
        residual='Refit/support remainder includes changed visual coefficients, incidence coefficients, intercepts and support fitting; not pure visual improvement.',
        target_tuning=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed probe')
    write_json(OUT/'PROTOCOL.json', dict(intervention='Mute pass on eval queries only; original source heads and64support metadata retained',
        forced_plus_one='Numerical identification of additive class-score vector, not physically relabeling orbit',
        target_labels='Original64 only during fitting; query labels only evaluation',
        purpose='Explain port/class gains and harms through direct query term versus refit/support remainder'))
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'product', 'port', 'class_id', 'class_name'])
    side = pd.read_csv(SIDECAR, usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    frame = frame.merge(side, on=['instance_id', 'product'], how='left', sort=False, validate='one_to_one')
    frame.loc[frame.status.ne('ok'), 'incidence_deg'] = np.nan
    cache = json.loads(CACHE.read_text(encoding='utf-8'))
    frame['pass_value'] = frame['product'].map(lambda p: cache.get(p, {}).get('pass')).map(
        {'Ascending': 1., 'Descending': -1.}).fillna(0.)
    freeze(frame)
    evaluate(frame)


if __name__ == '__main__':
    main()
