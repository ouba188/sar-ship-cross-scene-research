"""Paired C4 event attribution without fitting a new classifier or router."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from receiver_consistent_handoff import confidence_parts
from evidence_aware_handoff import predict_gate, route_scores

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
SOURCE = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
OUT = ROOT/'c4_condition_diagnosis_20261009'
C4 = ROOT/'c4_base_trial_20261009'
EXTERNAL = ROOT/'cross_dataset_c4_20261009'
PHASES = ('reference', 'query_c4', 'support_c4', 'matched_c4')
CONTRASTS = {'query': (0, 1), 'support_update': (1, 2), 'support_total': (0, 2),
             'source_refit': (2, 3), 'matched_total': (0, 3)}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def aggregate_events(predictions, labels, masks, tasks=1):
    """Preserve equal-draw/equal-task OA and present-class BA contributions."""
    p, y, ev = np.asarray(predictions), np.asarray(labels), np.asarray(masks, dtype=bool)
    if p.shape != (len(ev), 4, len(y)) or ev.shape[1] != len(y):
        raise ValueError('Expected draw x four-phases x rows, with matched query masks')
    correct = p == y[None, None, :]
    ow, bw = np.zeros(ev.shape), np.zeros(ev.shape)
    for d, mask in enumerate(ev):
        classes, inverse, counts = np.unique(y[mask], return_inverse=True, return_counts=True)
        ow[d, mask] = 1./(mask.sum()*len(ev)*tasks)
        bw[d, mask] = 1./(len(classes)*counts[inverse]*len(ev)*tasks)
    n = ev.sum(0)
    result = dict(query_draws=n, oa_weight=ow.sum(0), ba_weight=bw.sum(0))
    for i, phase in enumerate(PHASES):
        result[phase+'_correct_rate'] = (correct[:, i]*ev).sum(0)/np.maximum(n, 1)
        result[phase+'_oa_pp'] = 100*(correct[:, i]*ow).sum(0)
        result[phase+'_ba_pp'] = 100*(correct[:, i]*bw).sum(0)
    events = {}
    for name, (a, b) in CONTRASTS.items():
        gain, harm = ~correct[:, a] & correct[:, b], correct[:, a] & ~correct[:, b]
        events[name+'_rescue'], events[name+'_harm'] = gain, harm
        result[name+'_delta_oa_pp'] = 100*((gain.astype(float)-harm)*ow).sum(0)
        result[name+'_delta_ba_pp'] = 100*((gain.astype(float)-harm)*bw).sum(0)
    # This excludes cases already correct in the unmodified reference.
    events['c4_rescue_lost_to_refit'] = ~correct[:, 0] & correct[:, 2] & ~correct[:, 3]
    events['refit_rescue_after_c4_harm'] = correct[:, 0] & ~correct[:, 2] & correct[:, 3]
    for name, mask in events.items():
        result[name+'_draw_events'] = (mask & ev).sum(0)
        result[name+'_rate'] = (mask & ev).sum(0)/np.maximum(n, 1)
        result[name+'_ba_mass_pp'] = 100*(mask*bw).sum(0)
    return pd.DataFrame(result)


def feature_change(zero, average, rows, pca):
    values = []
    for lo in range(0, len(rows), 1024):
        rr = rows[lo:lo+1024]
        z, a = np.asarray(zero[rr], float), np.asarray(average[rr], float)
        delta = z-a
        norm = np.linalg.norm(delta, axis=1)
        retained = np.linalg.norm(delta@pca.components_.T, axis=1)
        values.append(np.c_[norm/np.maximum(np.linalg.norm(a, axis=1), 1e-12),
                            retained/np.maximum(norm, 1e-12)])
    return np.concatenate(values)


def freeze_port_events():
    frame = pd.read_csv(SOURCE/'manifest.csv', low_memory=False, keep_default_na=False)
    folders = sorted((C4/'ports').glob('*/policies.npz'))
    zero = np.load(C4/'pooled_zero.npy', mmap_mode='r')
    average = np.load(C4/'pooled_c4.npy', mmap_mode='r')
    output, checks = [], []
    for path in folders:
        port = path.parent.name
        parent = ROOT/'protected_drift_activation_20261008/ports'/port
        with np.load(path) as f:
            saved = {k: f[k] for k in f.files}
        with np.load(ROOT/'direct_incidence_comparison_20261009/ports'/port/'policies.npz') as f:
            slot = list(f['mode_names']).index('direct_condition')
            expert = f['expert_scores'][:, slot]
        with (parent/'source_models.pkl').open('rb') as f:
            gate = pickle.load(f)['gates']['joint']
        with (parent/'geometry.pkl').open('rb') as f:
            pca = pickle.load(f)['model']['pca']
        tr, masks = saved['query_rows'], saved['eval_mask']
        y = frame.loc[tr, 'class_id'].to_numpy(int)
        modes = [list(saved['base_modes']).index(k) for k in ('archived', 'query_c4', 'support_c4', 'matched_c4')]
        base = saved['base_scores'][:, modes]
        full, selected = [], []
        for d, mask in enumerate(masks):
            em, eh = confidence_parts(expert[d])
            predictions, selections = [], []
            for i in range(4):
                bm, bh = confidence_parts(base[d, i])
                score = predict_gate(gate, np.c_[em, bm, eh, bh])
                result = route_scores(base[d, i], expert[d], score,
                    frame.loc[saved['support_rows'][d], 'class_id'].to_numpy(int), mask, tr)
                predictions.append(result['predictions'])
                selections.append(result['selected'])
            np.testing.assert_array_equal(predictions[0], saved['full_predictions'][d, 0])
            np.testing.assert_array_equal(predictions[3], saved['full_predictions'][d, 1])
            np.testing.assert_array_equal(selections[0], saved['selected'][d, 0])
            np.testing.assert_array_equal(selections[3], saved['selected'][d, 1])
            full.append(predictions)
            selected.append(selections)
        dest = OUT/'port_policies'/port
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest/'policies.npz', phases=np.asarray(PHASES), query_rows=tr,
            predictions=np.asarray(full, dtype=np.int8), base_predictions=base.argmax(3).astype(np.int8),
            selected=np.asarray(selected), eval_masks=masks, support_rows=saved['support_rows'])
        change = feature_change(zero, average, tr, pca)
        for branch, pred in [('base', base.argmax(3)), ('full', np.asarray(full))]:
            result = aggregate_events(pred, y, masks, len(folders))
            result['dataset'], result['task'], result['branch'] = 'self24ports', port, branch
            result['row_id'], result['class_id'] = tr, y
            result['sample_id'] = frame.loc[tr, 'instance_id'].to_numpy(str)
            result['rotation_shift_rel'], result['shift_pca_retention'] = change.T
            output.append(result[result.query_draws.gt(0)])
        checks.append(dict(port=port, reference_and_matched_replay_exact=True, query_rows=len(tr)))
    return pd.concat(output, ignore_index=True), checks


def external_events(name):
    folder = ROOT/('f0921_detector_recrop_20261007' if name == 'f0921' else 'opensarship2_transfer_20261007')
    frame = pd.read_csv(folder/'target_manifest.csv', keep_default_na=False)
    with np.load(EXTERNAL/name/'policies.npz') as f:
        saved = {k: f[k] for k in f.files}
    with np.load(EXTERNAL/name/'support_refit_diagnostic/policies.npz') as f:
        support = {k: f[k] for k in f.files}
    np.testing.assert_array_equal(saved['support_rows'], support['support_rows'])
    np.testing.assert_array_equal(saved['eval_masks'], support['eval_masks'])
    np.testing.assert_array_equal(saved['sample_ids'], frame.sample_id.to_numpy(str))
    indices = [list(saved['arms']).index(a) for a in ('reference', 'c4_query_only', 'c4_matched')]
    provenance = json.loads((EXTERNAL/name/'FROZEN.json').read_text())
    model_path = provenance.get('model_path', str(folder/'source_model.pkl'))
    with open(model_path, 'rb') as f:
        pca = pickle.load(f)['pca']
    rows, y = np.arange(len(frame)), frame.class_id.to_numpy(int)
    change = feature_change(np.load(EXTERNAL/name/'pooled_zero.npy', mmap_mode='r'),
        np.load(EXTERNAL/name/'pooled_c4.npy', mmap_mode='r'), rows, pca)
    records = []
    for branch, key in [('base', 'base_predictions'), ('full', 'predictions')]:
        prediction = np.stack([saved[key][:, indices[0]], saved[key][:, indices[1]],
                               support[key], saved[key][:, indices[2]]], axis=1)
        result = aggregate_events(prediction, y, saved['eval_masks'])
        result['dataset'], result['task'], result['branch'] = name, name, branch
        result['row_id'], result['class_id'], result['sample_id'] = rows, y, frame.sample_id.to_numpy(str)
        result['rotation_shift_rel'], result['shift_pca_retention'] = change.T
        records.append(result[result.query_draws.gt(0)])
    return pd.concat(records, ignore_index=True)


def events():
    if (OUT/'EVENTS_COMPLETE.json').exists():
        print('REUSE EVENTS', flush=True)
        return
    write_json(OUT/'PROTOCOL.json', dict(
        datasets=['self24ports', 'f0921', 'oss2'], phases=PHASES, contrasts=CONTRASTS,
        no_new_model_or_gate=True, no_new_target_labels=True, no_new_image_perturbations=True,
        primary='First locate gains/harms and source refit losses, before residual module design',
        weighting='Equal five draws;24port metrics equal23tasks; BA present-query-class weighting per draw',
        descriptors='Label-free same128inputs; bright morphology and contrast proxies, NOT groundresolution or pure ship segmentation',
        feature_bins='Source24 marginal terciles frozen then reused externally; descriptive not deployed',
        confounding='Show class-conditional strata as well as weighted summaries; no causal inference',
        excluded='No FUSAR/generic; OSS1 auxiliary record only', historical_development=True))
    source, checks = freeze_port_events()
    source.to_csv(OUT/'self24ports_events.csv.gz', index=False, compression='gzip')
    print('EVENTS self24ports '+str(len(source)), flush=True)
    for name in ('oss2', 'f0921'):
        result = external_events(name)
        result.to_csv(OUT/f'{name}_events.csv.gz', index=False, compression='gzip')
        print('EVENTS '+name+' '+str(len(result)), flush=True)
    write_json(OUT/'EVENTS_COMPLETE.json', dict(port_checks=checks, targets=['oss2', 'f0921']))


def summaries():
    records, classes, totals = [], [], []
    for name in ('self24ports', 'oss2', 'f0921'):
        d = pd.read_csv(OUT/f'{name}_events.csv.gz')
        for branch, part in d.groupby('branch'):
            for phase in PHASES:
                totals.append(dict(dataset=name, branch=branch, phase=phase,
                    oa=part[phase+'_oa_pp'].sum(), ba=part[phase+'_ba_pp'].sum()))
            for contrast in CONTRASTS:
                prefix = contrast+'_'
                records.append(dict(dataset=name, branch=branch, contrast=contrast,
                    delta_oa_pp=part[prefix+'delta_oa_pp'].sum(), delta_ba_pp=part[prefix+'delta_ba_pp'].sum(),
                    unique_rescued=int(part[prefix+'rescue_draw_events'].gt(0).sum()),
                    unique_harmed=int(part[prefix+'harm_draw_events'].gt(0).sum()),
                    rescue_draw_events=int(part[prefix+'rescue_draw_events'].sum()),
                    harm_draw_events=int(part[prefix+'harm_draw_events'].sum()),
                    rescue_ba_pp=part[prefix+'rescue_ba_mass_pp'].sum(), harm_ba_pp=part[prefix+'harm_ba_mass_pp'].sum()))
            for c, sub in part.groupby('class_id'):
                row = dict(dataset=name, branch=branch, class_id=c, images=len(sub))
                for contrast in CONTRASTS:
                    row[contrast+'_delta_recall_pp'] = sub[contrast+'_delta_ba_pp'].sum()/sub.ba_weight.sum()
                row['c4_rescue_lost_images'] = int(sub.c4_rescue_lost_to_refit_draw_events.gt(0).sum())
                row['c4_rescue_lost_draw_events'] = int(sub.c4_rescue_lost_to_refit_draw_events.sum())
                classes.append(row)
    pd.DataFrame(records).to_csv(OUT/'event_summary.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_stage_summary.csv', index=False)
    pd.DataFrame(totals).to_csv(OUT/'phase_summary.csv', index=False)
    print(pd.DataFrame(totals).to_string(index=False), flush=True)


def transitions():
    records = []
    support_records = []
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id'])
    items = []
    for path in sorted((OUT/'port_policies').glob('*/policies.npz')):
        with np.load(path) as p:
            items.append(('self24ports', path.parent.name,
                frame.loc[p['query_rows'], 'class_id'].to_numpy(int),
                p['base_predictions'], p['predictions'], p['eval_masks'],
                frame.class_id.to_numpy(int)[p['support_rows']]))
    for name in ('oss2', 'f0921'):
        directory = ROOT/('opensarship2_transfer_20261007' if name == 'oss2' else 'f0921_detector_recrop_20261007')
        y = pd.read_csv(directory/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy(int)
        with np.load(EXTERNAL/name/'policies.npz') as p, np.load(EXTERNAL/name/'support_refit_diagnostic/policies.npz') as q:
            predictions = []
            for key in ('base_predictions', 'predictions'):
                predictions.append(np.stack([p[key][:, 0], p[key][:, 2], q[key], p[key][:, 3]], axis=1))
            items.append((name, name, y, *predictions, p['eval_masks'], y[p['support_rows']]))
    for dataset, task, y, base, full, masks, support_labels in items:
        for branch, p in [('base', base), ('full', full)]:
            for draw, ev in enumerate(masks):
                support_counts = np.bincount(support_labels[draw], minlength=8)
                for c in np.unique(y[ev]):
                    mask = ev & (y == c)
                    recalls = (p[draw][:, mask] == c).mean(1)
                    support_records.append(dict(dataset=dataset, task=task, branch=branch,
                        draw=draw, class_id=int(c), support_count=int(support_counts[c]),
                        query_count=int(mask.sum()), support_total_delta_pp=100*(recalls[2]-recalls[0]),
                        source_refit_delta_pp=100*(recalls[3]-recalls[2])))
                for subset in ('source_refit_harm', 'c4_rescue_lost_to_refit'):
                    selected = ev & (p[draw, 2] == y) & (p[draw, 3] != y)
                    if subset == 'c4_rescue_lost_to_refit':
                        selected &= p[draw, 0] != y
                    pairs, counts = np.unique(np.c_[y[selected], p[draw, 3, selected]], axis=0, return_counts=True)
                    for (truth, wrong), count in zip(pairs, counts):
                        records.append(dict(dataset=dataset, task=task, branch=branch, draw=draw,
                            subset=subset, true_class=int(truth), refit_prediction=int(wrong), count=int(count)))
    pd.DataFrame(records).to_csv(OUT/'refit_harm_transitions.csv', index=False)
    pd.DataFrame(support_records).to_csv(OUT/'support_count_stage_effects.csv', index=False)
    print('TRANSITIONS COMPLETE', flush=True)


def strata(rank=False):
    # Descriptor columns are explicit to keep labels and ancillary metadata out.
    features = ['rotation_shift_rel', 'shift_pca_retention']
    descriptor_config = json.loads((OUT/'descriptors/PROVENANCE.json').read_text())
    features += descriptor_config['analysis_features']
    data = {}
    for name in ('self24ports', 'oss2', 'f0921'):
        events = pd.read_csv(OUT/f'{name}_events.csv.gz')
        desc = pd.read_csv(OUT/'descriptors'/f'{name}.csv')
        if len(desc) != desc.row_id.nunique():
            raise ValueError('Duplicate descriptor rows')
        data[name] = events.merge(desc[['row_id']+descriptor_config['analysis_features']],
            on='row_id', how='left', validate='many_to_one')
    source = data['self24ports'][data['self24ports'].branch.eq('base')]
    bins, records, port_records, rank_bins = {}, [], [], {}
    prefix = 'rank_' if rank else ''
    for feature in features:
        values = pd.to_numeric(source[feature], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        cut = np.unique(np.quantile(values, [1/3, 2/3])).tolist()
        bins[feature] = cut
        for name, frame in data.items():
            frame = frame.copy()
            x = pd.to_numeric(frame[feature], errors='coerce').to_numpy(float)
            local_cut = cut
            if rank:
                unlabelled = x[frame.branch.eq('base').to_numpy() & np.isfinite(x)]
                local_cut = np.unique(np.quantile(unlabelled, [1/3, 2/3])).tolist()
                if len(local_cut) != len(cut):
                    raise ValueError('Tied quantiles changed the number of rank groups')
                rank_bins.setdefault(name, {})[feature] = local_cut
            frame['stratum'] = np.where(np.isfinite(x), np.searchsorted(local_cut, x, side='right'), -1)
            for (branch, c, bucket), group in frame.groupby(['branch', 'class_id', 'stratum']):
                row = dict(dataset=name, branch=branch, feature=feature, class_id=c,
                           stratum=int(bucket), images=len(group), ba_coverage=group.ba_weight.sum())
                for contrast in CONTRASTS:
                    row[contrast+'_delta_pp'] = group[contrast+'_delta_ba_pp'].sum()/group.ba_weight.sum()
                records.append(row)
            if name == 'self24ports':
                for (branch, task, c, bucket), group in frame.groupby(['branch', 'task', 'class_id', 'stratum']):
                    row = dict(branch=branch, task=task, feature=feature, class_id=c,
                               stratum=int(bucket), images=len(group))
                    for contrast in CONTRASTS:
                        row[contrast+'_delta_pp'] = group[contrast+'_delta_ba_pp'].sum()/group.ba_weight.sum()
                    port_records.append(row)
    write_json(OUT/(prefix+'source_feature_bins.json'), bins)
    if rank:
        write_json(OUT/'rank_dataset_feature_bins.json', dict(
            bins=rank_bins, target_labels_used=False, target_unlabelled_distribution_used=True,
            descriptive_only=True, added_after_source_bin_coverage_failure=True))
    result = pd.DataFrame(records)
    result.to_csv(OUT/(prefix+'class_condition_strata.csv'), index=False)
    # Only form low/high comparisons where each class has20images in BOTH bins.
    comparisons = []
    for (name, branch, feature), group in result.groupby(['dataset', 'branch', 'feature']):
        high = len(bins[feature])
        lo = group[group.stratum.eq(0)].set_index('class_id')
        hi = group[group.stratum.eq(high)].set_index('class_id')
        shared = lo.index.intersection(hi.index)
        shared = [c for c in shared if min(lo.loc[c, 'images'], hi.loc[c, 'images']) >= 20]
        for contrast in CONTRASTS:
            col = contrast+'_delta_pp'
            comparisons.append(dict(dataset=name, branch=branch, feature=feature, contrast=contrast,
                common_classes=len(shared), class_ids=','.join(map(str, shared)),
                low_delta_pp=float(lo.loc[shared, col].mean()) if shared else np.nan,
                high_delta_pp=float(hi.loc[shared, col].mean()) if shared else np.nan,
                high_minus_low_pp=float((hi.loc[shared, col]-lo.loc[shared, col]).mean()) if shared else np.nan))
    pd.DataFrame(comparisons).to_csv(OUT/(prefix+'class_matched_condition_contrasts.csv'), index=False)
    shared_records = []
    for branch in ('base', 'full'):
        for feature in features:
            lookups, eligible = {}, []
            high = len(bins[feature])
            for name in data:
                part = result[(result.dataset == name) & (result.branch == branch) & (result.feature == feature)]
                lo = part[part.stratum.eq(0)].set_index('class_id')
                hi = part[part.stratum.eq(high)].set_index('class_id')
                shared = lo.index.intersection(hi.index)
                eligible.append({c for c in shared if min(lo.loc[c, 'images'], hi.loc[c, 'images']) >= 20})
                lookups[name] = (lo, hi)
            common = sorted(set.intersection(*eligible))
            for name, (lo, hi) in lookups.items():
                for contrast in CONTRASTS:
                    col = contrast+'_delta_pp'
                    shared_records.append(dict(dataset=name, branch=branch, feature=feature,
                        contrast=contrast, common_classes=len(common), class_ids=','.join(map(str, common)),
                        low_delta_pp=float(lo.loc[common, col].mean()) if common else np.nan,
                        high_delta_pp=float(hi.loc[common, col].mean()) if common else np.nan,
                        high_minus_low_pp=float((hi.loc[common, col]-lo.loc[common, col]).mean()) if common else np.nan))
    pd.DataFrame(shared_records).to_csv(OUT/(prefix+'shared_class_condition_contrasts.csv'), index=False)
    pc = pd.DataFrame(port_records)
    pc.to_csv(OUT/(prefix+'port_class_condition_strata.csv'), index=False)
    paired_ports = []
    for (branch, feature), group in pc.groupby(['branch', 'feature']):
        lo = group[group.stratum.eq(0)].set_index(['task', 'class_id'])
        hi = group[group.stratum.eq(len(bins[feature]))].set_index(['task', 'class_id'])
        common = lo.index.intersection(hi.index)
        common = [c for c in common if min(lo.loc[c, 'images'], hi.loc[c, 'images']) >= 20]
        for contrast in CONTRASTS:
            column = contrast+'_delta_pp'
            delta = hi.loc[common, column]-lo.loc[common, column]
            paired_ports.append(dict(branch=branch, feature=feature, contrast=contrast,
                comparable_port_class_cells=len(common), high_minus_low_pp=float(delta.mean()),
                positive_cells=int((delta > 0).sum()), negative_cells=int((delta < 0).sum())))
    pd.DataFrame(paired_ports).to_csv(OUT/(prefix+'within_port_class_condition_contrasts.csv'), index=False)
    print('STRATA COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['events', 'summaries', 'transitions', 'strata', 'rank_strata'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {'events': events, 'summaries': summaries, 'transitions': transitions, 'strata': strata,
     'rank_strata': lambda: strata(rank=True)}[args.stage]()
