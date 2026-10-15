"""Availability, separate axes, relative axis and explicit visual interaction."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import json
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from extract_ship_axis_proxy import OUT as GEOMETRY
from run_direct_observation_knowledge import SOURCE, PARENT, ROOT, REFERENCE
from run_conditioned_vessel_drift import SIDECAR
from direct_incidence_conditioning import condition_features, extend_transforms, predict_conditioned
from run_protected_drift_activation import make_views, make_transforms, gate_features
from protected_drift_activation import fit_source_heads
from evidence_aware_handoff import predict_gate, route_scores
from sar_joint_completion import write_json
from run_generic_transfer import metrics

OUT = ROOT/'relative_observation_trial_20261009'
MODES = ('incidence', 'availability', 'separate_axes', 'relative_axis', 'relative_visual_interaction')


def axis_features(frame):
    known = frame.relative_status.eq('ok').to_numpy()
    body = np.deg2rad(frame.body_axis_deg.fillna(0).to_numpy())*2
    radar = np.deg2rad(frame.range_axis_deg.fillna(0).to_numpy())*2
    relative = body-radar
    return dict(known=known.astype(float)[:, None],
        separate=np.column_stack((np.cos(body), np.sin(body), np.cos(radar), np.sin(radar)))*known[:, None],
        relative=np.column_stack((np.cos(relative), np.sin(relative)))*known[:, None])


def knowledge(mode, incidence, axis, visual):
    blocks, scales = [incidence, axis['known']], [1/np.sqrt(2), 1/np.sqrt(2), 1.]
    if mode == 'separate_axes':
        blocks.append(axis['separate'])
        scales.extend([.5]*4)
    if mode in ('relative_axis', 'relative_visual_interaction'):
        blocks.append(axis['relative'])
        scales.extend([1/np.sqrt(2)]*2)
    if mode == 'relative_visual_interaction':
        interaction = (axis['relative'][:, :, None]*visual[:, None, :]).reshape(len(visual), -1)
        blocks.append(interaction)
        scales.extend([.25]*16)
    return np.column_stack(blocks), np.asarray(scales)


def freeze_port(folder, frame, raw, axes):
    out = OUT/'ports'/folder.name
    out.mkdir(parents=True, exist_ok=True)
    with np.load(folder/'policies.npz') as f:
        old = {k: f[k] for k in f.files}
    with np.load(REFERENCE/'ports'/folder.name/'policies.npz') as f:
        reference = {k: f[k] for k in f.files}
    for key in ['query_rows', 'support_rows', 'eval_mask']:
        np.testing.assert_array_equal(old[key], reference[key])
    with (folder/'geometry.pkl').open('rb') as f:
        g = pickle.load(f)
    with (folder/'source_models.pkl').open('rb') as f:
        fitted = pickle.load(f)
    sr, tr = old['source_rows'], old['query_rows']
    source = np.load(folder/'source_bank.npy', mmap_mode='r')
    target = np.load(folder/'target_bank.npy', mmap_mode='r')
    transforms = {k: v for k, v in make_transforms(g).items() if k.startswith('joint_')}
    # Eight existing leading retained coordinates, source-only standardization.
    visual_map = transforms['joint_base'][:, :8]
    source_visual, target_visual = source @ visual_map, target @ visual_map
    center, scale = source_visual.mean(0), source_visual.std(0)+1e-6
    source_visual, target_visual = (source_visual-center)/scale, (target_visual-center)/scale
    np.savez_compressed(out/'interaction_geometry.npz', visual_map=visual_map, center=center, scale=scale)
    si = condition_features(frame.loc[sr, 'incidence_deg'].to_numpy())
    ti = condition_features(frame.loc[tr, 'incidence_deg'].to_numpy())
    sa, ta = ({k: v[rows] for k, v in axes.items()} for rows in [sr, tr])
    extra = np.load(ROOT/'module_completion_20261008/ports_projection'/folder.name/'matched_joint/joint_extra.npy', mmap_mode='r')
    views = make_views(target, raw[tr], extra, g)
    index = list(reference['mode_names']).index('direct_condition')
    saved = {k: [reference[k][:, index:index+1]] for k in ['base_scores', 'expert_scores', 'predictions', 'selected']}
    for mode in MODES[1:]:
        su, scales = knowledge(mode, si, sa, source_visual)
        tu, _ = knowledge(mode, ti, ta, target_visual)
        heads = fit_source_heads(np.c_[source, su], frame.loc[sr, 'class_id'].to_numpy(),
                                 extend_transforms(transforms, su.shape[1]))
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
            routed = route_scores(base, expert, predict_gate(fitted['gates']['joint'], gate_features(base, expert)), labels, ev, tr)
            for key, value in [('base_scores', base), ('expert_scores', expert),
                               ('predictions', routed['predictions']), ('selected', routed['selected'])]:
                values[key].append(value)
        for key in saved:
            saved[key].append(np.stack(values[key])[:, None])
    np.savez_compressed(out/'policies.npz', mode_names=np.asarray(MODES), query_rows=tr,
        support_rows=old['support_rows'], eval_mask=old['eval_mask'],
        **{k: np.concatenate(v, axis=1) for k, v in saved.items()})
    check = dict(port=folder.name, source_axis_n=int(sa['known'].sum()), target_axis_n=int(ta['known'].sum()),
                 support_axis_counts=[int(axes['known'][s].sum()) for s in old['support_rows']])
    write_json(out/'FROZEN.json', check)
    print(json.dumps(check), flush=True)
    return check


def evaluate(frame, output=OUT):
    records, classes = [], []
    for folder in sorted((output/'ports').iterdir()):
        with np.load(folder/'policies.npz') as f:
            p = {k: f[k] for k in f.files}
        truth = frame.loc[p['query_rows'], 'class_id'].to_numpy()
        known = frame.loc[p['query_rows'], 'relative_status'].eq('ok').to_numpy()
        for draw, ev in enumerate(p['eval_mask']):
            for m, mode in enumerate(p['mode_names']):
                for population, mask in [('all', ev), ('axis_available', ev & known), ('axis_missing', ev & ~known)]:
                    if not mask.any():
                        continue
                    row = dict(port=folder.name, draw=draw, mode=mode, population=population,
                               n=int(mask.sum()), selected=int(p['selected'][draw, m, mask].sum()))
                    for stage, pred in [('full', p['predictions'][draw, m]), ('base', p['base_scores'][draw, m].argmax(1)),
                                         ('expert', p['expert_scores'][draw, m].argmax(1))]:
                        metric = metrics(truth[mask], pred[mask], 8)
                        row[stage+'_oa'], row[stage+'_ba'] = metric['acc'], metric['ba']
                    records.append(row)
                    for cls in np.unique(truth[mask]):
                        cm = mask & (truth == cls)
                        before = p['predictions'][draw, 0]
                        after = p['predictions'][draw, m]
                        classes.append(dict(port=folder.name, draw=draw, mode=mode, population=population, class_id=int(cls),
                            n=int(cm.sum()), recall=float((after[cm] == cls).mean()),
                            rescues=int(((before[cm] != cls) & (after[cm] == cls)).sum()),
                            harms=int(((before[cm] == cls) & (after[cm] != cls)).sum())))
    records, classes = pd.DataFrame(records), pd.DataFrame(classes)
    records.to_csv(output/'metrics.csv', index=False)
    classes.to_csv(output/'class_recalls.csv', index=False)
    port_means = records.groupby(['port', 'mode', 'population']).mean(numeric_only=True).drop(columns='draw').reset_index()
    port_means.to_csv(output/'port_means.csv', index=False)
    eight = frame.groupby('port').class_id.nunique()
    summaries = []
    for group, mask in [('all23', np.ones(len(port_means), bool)), ('all8_target12', port_means.port.map(eight).eq(8))]:
        part = port_means[mask]
        mean = part.groupby(['population', 'mode']).mean(numeric_only=True)
        mean['ports'] = part.groupby(['population', 'mode']).port.nunique()
        mean = mean.reset_index()
        mean.insert(0, 'group', group)
        summaries.append(mean)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output/'summary.csv', index=False)
    print(summary[['group', 'population', 'mode', 'ports', 'full_oa', 'full_ba']].to_string(index=False), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed trial')
    write_json(OUT/'PROTOCOL.json', dict(modes=MODES, metadata_source=str(GEOMETRY),
        question='Does relative-axis information help beyond availability and separate axes; does visual interaction help?',
        fixed='23ports,5original64draws,5percenthandoff,frozenbackbones/geometry/gate,ridge1/8',
        availability='same final relative-axis availability in all added-information arms',
        axis_encoding='cos2a,sin2a for undirected axes; zero if unavailable',
        interaction='2relative coordinates times8leading existing retained coordinates; source-only standardization',
        target_block_scales='incidence/sqrt2,mask1,separateaxes/2,relative/sqrt2,interaction/4',
        source_heads='Independent augmented coordinates; unequal parameter counts, not capacity matched',
        angle_search=False, query_tuning=False, independent_confirmation=False, main_model_unchanged=True))
    frame = pd.read_csv(GEOMETRY/'relative_conditions.csv')
    manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id'])
    np.testing.assert_array_equal(frame.instance_id, manifest.instance_id)
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
    write_json(OUT/'ALL_FROZEN.json', dict(ports=len(checks), query_labels_used=0))
    evaluate(frame)
    write_json(OUT/'COMPLETE.json', dict(ports=len(checks), draws=5, modes=MODES))


if __name__ == '__main__':
    main()
