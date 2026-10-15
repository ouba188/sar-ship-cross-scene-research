"""Same-width interaction repair: decision coordinates instead of leading PCA."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import json
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from run_relative_observation_trial import (OUT as PREVIOUS, GEOMETRY, SOURCE, PARENT, ROOT, REFERENCE,
    SIDECAR, axis_features, knowledge, evaluate, condition_features, extend_transforms,
    predict_conditioned, make_views, make_transforms, gate_features, fit_source_heads,
    predict_gate, route_scores, write_json)

OUT = ROOT/'relative_decision_interaction_20261009'


def freeze_port(folder, frame, raw, axes):
    out = OUT/'ports'/folder.name
    out.mkdir(parents=True, exist_ok=True)
    with np.load(PREVIOUS/'ports'/folder.name/'policies.npz') as f:
        old = {k: f[k] for k in f.files}
    sr = np.load(folder/'source_rows.npy')
    tr = old['query_rows']
    with (folder/'geometry.pkl').open('rb') as f:
        g = pickle.load(f)
    with (folder/'source_models.pkl').open('rb') as f:
        gate = pickle.load(f)['gates']['joint']
    with (REFERENCE/'ports'/folder.name/'direct_heads.pkl').open('rb') as f:
        reference = pickle.load(f)['joint_base']
    source = np.load(folder/'source_bank.npy', mmap_mode='r')
    target = np.load(folder/'target_bank.npy', mmap_mode='r')
    transforms = {k: v for k, v in make_transforms(g).items() if k.startswith('joint_')}
    decision_map = reference['coef'][:source.shape[1]]
    sv, tv = source @ decision_map, target @ decision_map
    center, scale = sv.mean(0), sv.std(0)+1e-6
    original = source @ transforms['joint_base'][:, :8]
    _, inv, counts = np.unique(frame.loc[sr, 'class_id'].to_numpy(), return_inverse=True, return_counts=True)
    weights = 1./counts[inv]
    weights /= weights.sum()
    centered_pca, centered_score = original-weights @ original, sv-weights @ sv
    coefficient = np.linalg.solve(centered_pca.T @ (weights[:, None]*centered_pca)+1e-8*np.eye(8),
                                  centered_pca.T @ (weights[:, None]*centered_score))
    r2 = 1-np.sum(weights[:, None]*(centered_score-centered_pca @ coefficient)**2)/np.sum(weights[:, None]*centered_score**2)
    sv, tv = (sv-center)/scale, (tv-center)/scale
    np.savez_compressed(out/'interaction_geometry.npz', visual_map=decision_map, center=center, scale=scale)
    sa, ta = ({k: v[rows] for k, v in axes.items()} for rows in [sr, tr])
    su, scales = knowledge('relative_visual_interaction', condition_features(frame.loc[sr, 'incidence_deg'].to_numpy()), sa, sv)
    tu, _ = knowledge('relative_visual_interaction', condition_features(frame.loc[tr, 'incidence_deg'].to_numpy()), ta, tv)
    heads = fit_source_heads(np.c_[source, su], frame.loc[sr, 'class_id'].to_numpy(), extend_transforms(transforms, su.shape[1]))
    with (out/'decision_heads.pkl').open('wb') as f:
        pickle.dump(heads, f)
    extra = np.load(ROOT/'module_completion_20261008/ports_projection'/folder.name/'matched_joint/joint_extra.npy', mmap_mode='r')
    views = make_views(target, raw[tr], extra, g)
    values = {k: [] for k in ['base_scores', 'expert_scores', 'predictions', 'selected']}
    for draw, ev in enumerate(old['eval_mask']):
        support = np.searchsorted(tr, old['support_rows'][draw])
        np.testing.assert_array_equal(tr[support], old['support_rows'][draw])
        labels = frame.loc[old['support_rows'][draw], 'class_id'].to_numpy()
        base, expert = predict_conditioned(target, views, heads, support, labels, tu, scales)
        routed = route_scores(base, expert, predict_gate(gate, gate_features(base, expert)), labels, ev, tr)
        for key, value in [('base_scores', base), ('expert_scores', expert),
                           ('predictions', routed['predictions']), ('selected', routed['selected'])]:
            values[key].append(value)
    np.savez_compressed(out/'policies.npz', query_rows=tr, support_rows=old['support_rows'], eval_mask=old['eval_mask'],
        mode_names=np.append(old['mode_names'], 'relative_decision_interaction'),
        **{k: np.concatenate((old[k], np.stack(v)[:, None]), axis=1) for k, v in values.items()})
    check = dict(port=folder.name, source_balanced_score_r2_from_first8_pca=float(r2),
        interaction_coordinates=8, interaction_terms=16, source_only_decision_basis=True)
    write_json(out/'FROZEN.json', check)
    print(json.dumps(check), flush=True)
    return check


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed repair')
    write_json(OUT/'PROTOCOL.json', dict(parent=str(PREVIOUS),
        question='Is the failed visual interaction caused by using variance-leading instead of task-discriminative coordinates?',
        intervention='Replace8existing retained PCA coordinates by8source base-head visual scores; same16cross terms',
        basis='Frozen source classifier visual coefficients only, excluding incidence coefficients and intercept',
        normalization='Source mean/std; no target-query fitting',
        unchanged='All data,metadata,masks,64labels,5draws,5percentbudget,gate,ridge,blockscales',
        caveat='Equal coordinate/parameter counts, but score coordinates may be rank deficient/correlated; not identical function classes',
        query_tuning=False, repeated_development=True, original_negative_preserved=True))
    frame = pd.read_csv(GEOMETRY/'relative_conditions.csv')
    side = pd.read_csv(SIDECAR, usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    frame = frame.merge(side, on=['instance_id', 'product'], how='left', sort=False, validate='one_to_one')
    frame.loc[frame.status.ne('ok'), 'incidence_deg'] = np.nan
    axes = axis_features(frame)
    raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    checks = []
    with threadpool_limits(limits=2):
        for folder in sorted((PARENT/'ports').iterdir()):
            if (folder/'policies.npz').exists():
                checks.append(freeze_port(folder, frame, raw, axes))
    pd.DataFrame(checks).to_csv(OUT/'basis_diagnosis.csv', index=False)
    write_json(OUT/'ALL_FROZEN.json', dict(ports=len(checks), query_labels_used=0))
    evaluate(frame, OUT)
    write_json(OUT/'COMPLETE.json', dict(ports=len(checks), draws=5))


if __name__ == '__main__':
    main()
