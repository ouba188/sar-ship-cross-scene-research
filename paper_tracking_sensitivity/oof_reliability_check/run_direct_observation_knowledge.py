"""Fixed direct-condition study: incidence, orbital pass, satellite platform."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import json
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from direct_incidence_conditioning import condition_features, extend_transforms, predict_conditioned
from run_conditioned_vessel_drift import SIDECAR
from run_protected_drift_activation import make_views, make_transforms, gate_features
from protected_drift_activation import fit_source_heads
from evidence_aware_handoff import predict_gate, route_scores
from task_stable_sensitivity import SOURCE, PARENT, ROOT
from sar_joint_completion import write_json
from run_generic_transfer import metrics


OUT = ROOT/'direct_observation_knowledge_20261009'
REFERENCE = ROOT/'direct_incidence_comparison_20261009'
CACHE = ROOT.parent/'s1_product_incidence.json'
MODES = ('incidence', 'incidence_pass', 'incidence_platform', 'incidence_pass_platform')


def metadata_features(frame, cache):
    angle = condition_features(frame.incidence_deg.to_numpy())
    orbital = frame['product'].map(lambda p: cache.get(p, {}).get('pass'))
    platform = frame['product'].str.split('_').str[0]
    u = np.column_stack((angle, orbital.map({'Ascending': 1., 'Descending': -1.}).fillna(0),
                         platform.map({'S1A': -1., 'S1C': 1.}).fillna(0)))
    return u, orbital.notna().to_numpy(), platform.isin(['S1A', 'S1C']).to_numpy()


def columns(mode):
    return [0, 1] + ([2] if 'pass' in mode else []) + ([3] if 'platform' in mode else [])


def freeze_port(folder, frame, raw, conditions):
    out = OUT/'ports'/folder.name
    out.mkdir(parents=True, exist_ok=True)
    with (folder/'geometry.pkl').open('rb') as f:
        geometry = pickle.load(f)
    with (folder/'source_models.pkl').open('rb') as f:
        fitted = pickle.load(f)
    with np.load(folder/'policies.npz') as f:
        old = {k: f[k] for k in f.files}
    with np.load(REFERENCE/'ports'/folder.name/'policies.npz') as f:
        reference = {k: f[k] for k in f.files}
    for key in ('query_rows', 'support_rows', 'eval_mask'):
        np.testing.assert_array_equal(old[key], reference[key])
    sr, tr = old['source_rows'], old['query_rows']
    source = np.load(folder/'source_bank.npy', mmap_mode='r')
    target = np.load(folder/'target_bank.npy', mmap_mode='r')
    extra = np.load(ROOT/'module_completion_20261008/ports_projection'/folder.name/'matched_joint/joint_extra.npy', mmap_mode='r')
    views = make_views(target, raw[tr], extra, geometry)
    transforms = {k: v for k, v in make_transforms(geometry).items() if k.startswith('joint_')}
    index = list(reference['mode_names']).index('direct_condition')
    saved = {k: [reference[k][:, index:index+1]] for k in ('base_scores', 'expert_scores', 'predictions', 'selected')}
    with (REFERENCE/'ports'/folder.name/'direct_heads.pkl').open('rb') as f:
        direct_heads = pickle.load(f)
    support = np.searchsorted(tr, old['support_rows'][0])
    replay_scores = predict_conditioned(target, views, direct_heads, support,
        frame.loc[old['support_rows'][0], 'class_id'].to_numpy(), conditions[tr, :2])
    replay = max(float(np.max(np.abs(a-reference[k][0, index])))
                 for a, k in zip(replay_scores, ('base_scores', 'expert_scores')))
    if replay > 3e-6:
        raise ValueError(f'Incidence reference replay failed: {folder.name} {replay}')
    for mode in MODES[1:]:
        col = columns(mode)
        su, tu = conditions[sr][:, col], conditions[tr][:, col]
        scales = np.array([1/np.sqrt(2), 1/np.sqrt(2)] + [1.]*(len(col)-2))
        heads = fit_source_heads(np.column_stack((source, su)), frame.loc[sr, 'class_id'].to_numpy(),
                                 extend_transforms(transforms, len(col)))
        with (out/(mode+'_heads.pkl')).open('wb') as f:
            pickle.dump(heads, f)
        values = {k: [] for k in saved}
        for draw, ev in enumerate(old['eval_mask']):
            support = np.searchsorted(tr, old['support_rows'][draw])
            np.testing.assert_array_equal(tr[support], old['support_rows'][draw])
            if ev[support].any():
                raise ValueError('Support/query overlap')
            labels = frame.loc[old['support_rows'][draw], 'class_id'].to_numpy()
            base, expert = predict_conditioned(target, views, heads, support, labels, tu, scales)
            gate = predict_gate(fitted['gates']['joint'], gate_features(base, expert))
            routed = route_scores(base, expert, gate, labels, ev, tr)
            if routed['selected'][~ev].any():
                raise ValueError('Support entered handoff budget')
            for key, value in [('base_scores', base), ('expert_scores', expert),
                               ('predictions', routed['predictions']), ('selected', routed['selected'])]:
                values[key].append(value)
        for key in saved:
            saved[key].append(np.stack(values[key])[:, None])
    np.savez_compressed(out/'policies.npz', mode_names=np.array(MODES), query_rows=tr,
        support_rows=old['support_rows'], eval_mask=old['eval_mask'],
        **{key: np.concatenate(value, axis=1) for key, value in saved.items()})
    check = dict(port=folder.name, max_reference_replay_error=replay,
        source_pass_known=int(np.count_nonzero(conditions[sr, 2])), source_rows=len(sr),
        target_pass_known=int(np.count_nonzero(conditions[tr, 2])), target_rows=len(tr))
    write_json(out/'FROZEN.json', check)
    print(json.dumps(check), flush=True)
    return check


def evaluate(frame):
    records, recalls = [], []
    for folder in sorted((OUT/'ports').iterdir()):
        with np.load(folder/'policies.npz') as f:
            p = {k: f[k] for k in f.files}
        truth = frame.loc[p['query_rows'], 'class_id'].to_numpy()
        for draw, ev in enumerate(p['eval_mask']):
            for m, mode in enumerate(MODES):
                row = dict(port=folder.name, draw=draw, mode=mode, query_n=int(ev.sum()),
                           selected=int(p['selected'][draw, m].sum()))
                for stage, pred in [('full', p['predictions'][draw, m]),
                    ('base', p['base_scores'][draw, m].argmax(1)),
                    ('expert', p['expert_scores'][draw, m].argmax(1))]:
                    score = metrics(truth[ev], pred[ev], 8)
                    row[stage+'_oa'], row[stage+'_ba'] = score['acc'], score['ba']
                    for cls in np.unique(truth[ev]):
                        mask = ev & (truth == cls)
                        recalls.append(dict(port=folder.name, draw=draw, mode=mode, stage=stage,
                            class_id=int(cls), n=int(mask.sum()), recall=float((pred[mask] == cls).mean())))
                records.append(row)
    df = pd.DataFrame(records)
    df.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(recalls).to_csv(OUT/'class_recalls.csv', index=False)
    pm = df.groupby(['port', 'mode']).mean(numeric_only=True).drop(columns='draw').reset_index()
    ref = pm[pm['mode'] == 'incidence'].set_index('port')
    names = ['full_oa', 'full_ba', 'base_oa', 'base_ba', 'expert_oa', 'expert_ba']
    for name in names:
        pm[name+'_delta_pp'] = [100*(r[name]-ref.loc[r.port, name]) for _, r in pm.iterrows()]
    pm.to_csv(OUT/'port_means.csv', index=False)
    full8 = frame.groupby('port').class_id.nunique()
    summaries = []
    for group, mask in [('all23', np.ones(len(pm), bool)), ('all8_target12', pm.port.map(full8).eq(8))]:
        part = pm[mask]
        mean = part.groupby('mode').mean(numeric_only=True)
        for name in ['full_oa', 'full_ba']:
            mean[name+'_ports_up'] = part.groupby('mode')[name+'_delta_pp'].apply(lambda x: (x > 1e-9).sum())
            mean[name+'_ports_down'] = part.groupby('mode')[name+'_delta_pp'].apply(lambda x: (x < -1e-9).sum())
        mean = mean.reset_index()
        mean.insert(0, 'group', group)
        summaries.append(mean)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT/'summary.csv', index=False)
    print(summary[['group', 'mode', 'full_oa', 'full_ba', 'full_oa_delta_pp', 'full_ba_delta_pp']].to_string(index=False), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed experiment')
    write_json(OUT/'PROTOCOL.json', dict(date='2026-10-09', modes=MODES,
        question='Can pass/platform directly condition classification even if drift subtraction failed?',
        reference=str(REFERENCE), pass_encoding='Ascending +1, Descending -1, unavailable 0',
        platform_encoding='S1A -1, S1C +1; unknown 0',
        conditions='Existing incidence t,t^2 unchanged; add pass/platform independent coefficients',
        target_scaling='incidence block /sqrt2 unchanged; each binary block scale1',
        expert_kernel='Original visual kernel +.25 times each scaled condition block Gram',
        fixed='23ports,5original64labeldraws,5percenthandoff,source geometry,backbones,gate,ridge1/8',
        source_heads_refit=True, target_support_refit=True, metadata_mask_flag_added=False,
        parameter_search=False, capacity_matched=False, causal_effect_identified=False,
        missing_test_rows_dropped=False, main_model_unchanged=True, independent_confirmation=False))
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'product', 'port', 'class_id'])
    side = pd.read_csv(SIDECAR, usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    frame = frame.merge(side, on=['instance_id', 'product'], how='left', sort=False, validate='one_to_one')
    frame.loc[frame.status.ne('ok'), 'incidence_deg'] = np.nan
    conditions, pass_known, platform_known = metadata_features(frame, json.loads(CACHE.read_text(encoding='utf-8')))
    coverage = frame[['port', 'class_id']].assign(incidence_known=frame.incidence_deg.notna(),
        pass_known=pass_known, platform_known=platform_known)
    coverage.groupby(['port', 'class_id']).agg(['sum', 'count']).to_csv(OUT/'coverage.csv')
    raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    checks = []
    with threadpool_limits(limits=2):
        for folder in sorted((PARENT/'ports').iterdir()):
            if (folder/'policies.npz').exists():
                checks.append(freeze_port(folder, frame, raw, conditions))
    pd.DataFrame(checks).to_csv(OUT/'verification.csv', index=False)
    write_json(OUT/'ALL_FROZEN.json', dict(ports=len(checks), target_query_labels_used=0))
    evaluate(frame)
    write_json(OUT/'COMPLETE.json', dict(ports=len(checks), draws=5, modes=MODES,
        max_reference_replay_error=max(c['max_reference_replay_error'] for c in checks)))


if __name__ == '__main__':
    main()
