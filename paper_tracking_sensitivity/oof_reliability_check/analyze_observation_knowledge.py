"""Predeclared stage swaps and paired benefit/harm decomposition, without tuning."""

import json
import pickle

import numpy as np
import pandas as pd

from run_direct_observation_knowledge import OUT as RUN, SOURCE, PARENT, CACHE
from run_conditioned_vessel_drift import SIDECAR
from run_protected_drift_activation import gate_features
from evidence_aware_handoff import predict_gate, route_scores
from sar_joint_completion import write_json
from run_generic_transfer import metrics


OUT = RUN/'mechanism_analysis'
MODES = ('incidence', 'pass_both', 'pass_base_only', 'pass_expert_only', 'platform_both')
CLASS_NAMES = ('bulk_carrier', 'fishing', 'general_cargo', 'product_chemical_tanker',
               'container', 'crude_tanker', 'tug_towing', 'offshore_supply')


def freeze_swaps(frame):
    for folder in sorted((RUN/'ports').iterdir()):
        with np.load(folder/'policies.npz') as f:
            p = {k: f[k] for k in f.files}
        names = list(p['mode_names'])
        reference, both, platform = [names.index(n) for n in ('incidence', 'incidence_pass', 'incidence_platform')]
        with (PARENT/'ports'/folder.name/'source_models.pkl').open('rb') as f:
            gate = pickle.load(f)['gates']['joint']
        swaps, selections = [], []
        for draw, ev in enumerate(p['eval_mask']):
            labels = frame.loc[p['support_rows'][draw], 'class_id'].to_numpy()
            predictions, selected = [], []
            for bm, em in [(reference, reference), (both, both), (both, reference), (reference, both), (platform, platform)]:
                base, expert = p['base_scores'][draw, bm], p['expert_scores'][draw, em]
                route = route_scores(base, expert, predict_gate(gate, gate_features(base, expert)), labels, ev, p['query_rows'])
                if bm == em:
                    np.testing.assert_array_equal(route['predictions'], p['predictions'][draw, bm])
                    np.testing.assert_array_equal(route['selected'], p['selected'][draw, bm])
                predictions.append(route['predictions'])
                selected.append(route['selected'])
            swaps.append(predictions)
            selections.append(selected)
        np.savez_compressed(OUT/(folder.name+'.npz'), predictions=np.array(swaps), selected=np.array(selections),
            query_rows=p['query_rows'], support_rows=p['support_rows'], eval_mask=p['eval_mask'],
            baseline_base_scores=p['base_scores'][:, reference], modes=np.array(MODES))
    write_json(OUT/'ALL_SWAPS_FROZEN.json', dict(ports=len(list(OUT.glob('*.npz'))), query_labels_used=0))


def strata(rows, frame, original_scores, ev):
    meta = frame.loc[rows]
    angle = pd.cut(meta.incidence_deg, [-np.inf, 33, 38, 43, np.inf],
                   labels=['lt33', '33to38', '38to43', 'ge43'], right=False).astype(object).fillna('missing').to_numpy()
    # Label-free descriptive bins from the unchanged reference, not model selection.
    ordered = np.sort(original_scores, axis=1)
    gap = ordered[:, -1]-ordered[:, -2]
    thresholds = np.quantile(gap[ev], [.25, .5, .75])
    confidence = np.array(['Q1_low', 'Q2', 'Q3', 'Q4_high'])[np.searchsorted(thresholds, gap, side='right')]
    return dict(incidence=angle, reference_margin=confidence,
        pass_availability=np.where(meta.pass_name.notna(), 'known', 'missing'),
        orbital_pass=meta.pass_name.fillna('missing').to_numpy(),
        platform=meta['product'].str[:3].to_numpy())


def evaluate(frame):
    all_rows, contributions, breakdown = [], [], []
    for path in sorted(OUT.glob('*.npz')):
        with np.load(path) as f:
            p = {k: f[k] for k in f.files}
        truth = frame.loc[p['query_rows'], 'class_id'].to_numpy()
        for draw, ev in enumerate(p['eval_mask']):
            reference = p['predictions'][draw, 0]
            ref_score = metrics(truth[ev], reference[ev], 8)
            groups = strata(p['query_rows'], frame, p['baseline_base_scores'][draw], ev)
            classes = np.unique(truth[ev])
            for m, mode in enumerate(MODES):
                pred = p['predictions'][draw, m]
                score = metrics(truth[ev], pred[ev], 8)
                good_old, good_new = reference == truth, pred == truth
                rescue, harm = ~good_old & good_new, good_old & ~good_new
                oa_delta, ba_delta = 100*(score['acc']-ref_score['acc']), 100*(score['ba']-ref_score['ba'])
                all_rows.append(dict(port=path.stem, draw=draw, mode=mode, n=int(ev.sum()),
                    oa=score['acc'], ba=score['ba'], oa_delta_pp=oa_delta, ba_delta_pp=ba_delta,
                    rescues=int(rescue[ev].sum()), harms=int(harm[ev].sum()),
                    selected=int(p['selected'][draw, m].sum())))
                oa_sum = ba_sum = 0.
                for cls in classes:
                    mask = ev & (truth == cls)
                    delta = float(good_new[mask].mean()-good_old[mask].mean())
                    oc, bc = 100*delta*mask.sum()/ev.sum(), 100*delta/len(classes)
                    oa_sum += oc
                    ba_sum += bc
                    contributions.append(dict(port=path.stem, draw=draw, mode=mode, class_id=int(cls),
                        n=int(mask.sum()), recall_delta_pp=100*delta, oa_contribution_pp=oc, ba_contribution_pp=bc,
                        rescues=int(rescue[mask].sum()), harms=int(harm[mask].sum())))
                np.testing.assert_allclose([oa_sum, ba_sum], [oa_delta, ba_delta], atol=1e-10)
                for dimension, values in groups.items():
                    for group in np.unique(values[ev]):
                        mask = ev & (values == group)
                        a, b = metrics(truth[mask], reference[mask], 8), metrics(truth[mask], pred[mask], 8)
                        breakdown.append(dict(port=path.stem, draw=draw, mode=mode, dimension=dimension, group=group,
                            n=int(mask.sum()), rescues=int(rescue[mask].sum()), harms=int(harm[mask].sum()),
                            oa_delta_pp=100*(b['acc']-a['acc']), ba_delta_pp=100*(b['ba']-a['ba'])))
    scores, classes, groups = pd.DataFrame(all_rows), pd.DataFrame(contributions), pd.DataFrame(breakdown)
    scores.to_csv(OUT/'stage_metrics.csv', index=False)
    classes.to_csv(OUT/'class_contributions.csv', index=False)
    groups.to_csv(OUT/'strata_metrics.csv', index=False)
    parts = []
    eight = frame.groupby('port').class_id.nunique()
    for name, mask in [('all23', np.ones(len(scores), bool)), ('all8_target12', scores.port.map(eight).eq(8))]:
        mean = scores[mask].groupby(['port', 'mode']).mean(numeric_only=True).groupby('mode').mean()
        mean = mean.reset_index()
        mean.insert(0, 'group', name)
        parts.append(mean)
    summary = pd.concat(parts, ignore_index=True)
    summary.to_csv(OUT/'stage_summary.csv', index=False)
    # Missing classes contribute zero to the all-port mean, not a changed denominator.
    total = classes.groupby(['mode', 'class_id'])[['oa_contribution_pp', 'ba_contribution_pp']].sum()/115
    total.to_csv(OUT/'class_contribution_summary.csv')
    g = groups.groupby(['port', 'mode', 'dimension', 'group']).mean(numeric_only=True).reset_index()
    gs = g.groupby(['mode', 'dimension', 'group']).agg(ports=('port', 'nunique'),
        oa_delta_pp=('oa_delta_pp', 'mean'), ba_delta_pp=('ba_delta_pp', 'mean'),
        mean_n=('n', 'mean'), mean_rescues=('rescues', 'mean'), mean_harms=('harms', 'mean'))
    gs.to_csv(OUT/'strata_summary.csv')
    print(summary[['group', 'mode', 'oa', 'ba', 'oa_delta_pp', 'ba_delta_pp']].to_string(index=False))
    print(total.loc['pass_both'].to_string())
    print(gs.loc['pass_both'].to_string())
    write_json(OUT/'COMPLETE.json', dict(ports=23, draws=5, stage_modes=MODES,
        original_routing_exactly_reproduced=True, class_contributions_sum_to_metric_deltas=True,
        strata='Descriptive paired differences; different strata may contain different ports/classes.',
        mechanism_limit='Additive condition heads/kernels; no condition-by-visual interaction explicitly learned. Not causal attribution.'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed analysis')
    write_json(OUT/'PROTOCOL.json', dict(modes=MODES,
        interventions='Recombine already-fitted base/expert scores, same frozen gate and same64support/5percentbudget',
        refit=False, query_outcome_tuning=False,
        strata=['true class for evaluation', 'reference margin quartiles without labels', 'fixed angle bins', 'pass known/missing', 'pass', 'platform'],
        metrics=['paired OA/BA', 'rescue/harm', 'exact class contributions']))
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'product', 'port', 'class_id'])
    side = pd.read_csv(SIDECAR, usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    frame = frame.merge(side, on=['instance_id', 'product'], how='left', sort=False, validate='one_to_one')
    frame.loc[frame.status.ne('ok'), 'incidence_deg'] = np.nan
    cache = json.loads(CACHE.read_text(encoding='utf-8'))
    frame['pass_name'] = frame['product'].map(lambda p: cache.get(p, {}).get('pass'))
    freeze_swaps(frame)
    evaluate(frame)


if __name__ == '__main__':
    main()
