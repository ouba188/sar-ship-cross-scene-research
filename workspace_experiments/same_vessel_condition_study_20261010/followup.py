"""Exact score accounting and real-observation feature swaps; no model refit."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from study import OUT as BASE, ROOT, SOURCE, GROUPS, BATCH

OUT = BASE/'evidence_followup'


def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def margin(values, truth, rival):
    index = np.arange(len(truth))
    return values[index, truth]-values[index, rival]


def hybrid_scores(part50, part18, bias, a, b):
    return dict(actual_a=part50[a]+part18[a]+bias[a],
                r50_previous=part50[a]+part18[b]+bias[b],
                r18_previous=part50[b]+part18[a]+bias[b],
                actual_b=part50[b]+part18[b]+bias[b])


def transform(raw, rows, model, name):
    rank = model[name+'_components'].shape[0]
    answer = np.empty((len(rows), rank), np.float32)
    components = model[name+'_components']
    offset = model[name+'_pca_mean']@components.T
    for lo in range(0, len(rows), BATCH):
        sl = slice(lo, min(lo+BATCH, len(rows)))
        block = (np.asarray(raw[rows[sl]], np.float32)-model[name+'_center'])/model[name+'_scale']
        answer[sl] = (block@components.T-offset)/np.sqrt(rank)
    return answer


def reconstruct(obs):
    with np.load(BASE/'FROZEN_PREDICTIONS.npz') as z:
        scores, expected_relations = z['scores'], z['relations']
        np.testing.assert_array_equal(z['manifest_rows'], obs.manifest_row)
    raw = dict(r50=np.load(SOURCE/'pooled_features.npy', mmap_mode='r'),
        r18=np.load(ROOT/'sensitivity/neural_complement_pilot_20261007/frozen_replay/source_extra.npy', mmap_mode='r'))
    part50, part18, bias, single_bias = [np.full((len(obs), 8), np.nan) for _ in range(4)]
    nearest = np.full((len(obs), 2, 3), -1, np.int32)
    receipts = []
    labels_by_row = pd.Series(obs.class_id.to_numpy(), index=obs.manifest_row)
    for fold in range(3):
        with np.load(BASE/f'fold_{fold}_models.npz') as z:
            model = {k: z[k] for k in z.files}
        ix = np.flatnonzero(obs.fold.to_numpy() == fold)
        rows = obs.manifest_row.to_numpy()[ix]
        np.testing.assert_array_equal(rows, model['test_rows'])
        features = {}
        errors = {}
        for mi, name in enumerate(('r50', 'r18')):
            x = transform(raw[name], rows, model, name)
            features[name] = x
            replay = x@model[name+'_head_coef'].T+model[name+'_head_intercept']
            np.testing.assert_allclose(replay, scores[ix, mi], atol=1e-5, rtol=2e-5)
            np.testing.assert_array_equal(replay.argmax(1), scores[ix, mi].argmax(1))
            errors[name] = float(np.max(abs(replay-scores[ix, mi])))
            ref = transform(raw[name], model['reference_rows'], model, name)
            similarity = x@ref.T
            ref_labels = labels_by_row.loc[model['reference_rows']].to_numpy()
            relation = np.column_stack([similarity[:, ref_labels == c].mean(1) for c in range(8)])
            np.testing.assert_allclose(relation, expected_relations[ix, mi], atol=1e-5, rtol=2e-5)
            errors[name+'_relations'] = float(np.max(abs(relation-expected_relations[ix, mi])))
            top = np.argsort(-similarity, axis=1, kind='stable')[:, :3]
            nearest[ix, mi] = model['reference_rows'][top]
        w = np.asarray(model['dual_head_coef'], float)
        part50[ix] = (features['r50']/np.sqrt(2.)).astype(float)@w[:, :128].T
        part18[ix] = (features['r18']/np.sqrt(2.)).astype(float)@w[:, 128:].T
        bias[ix] = model['dual_head_intercept']
        single_bias[ix] = model['r50_head_intercept']
        replay = part50[ix]+part18[ix]+bias[ix]
        np.testing.assert_allclose(replay, scores[ix, 2], atol=1e-5, rtol=2e-5)
        np.testing.assert_array_equal(replay.argmax(1), scores[ix, 2].argmax(1))
        errors['dual'] = float(np.max(abs(replay-scores[ix, 2])))
        receipts.append(dict(fold=fold, n=len(ix), errors=errors, predictions_exact=True))
    np.savez_compressed(OUT/'score_parts.npz', part50=part50, part18=part18, bias=bias, single_bias=single_bias,
                        scores=scores, nearest_reference_rows=nearest,
                        manifest_rows=obs.manifest_row.to_numpy())
    write('REPLAY.json', receipts)
    return scores, part50, part18, bias, single_bias, nearest


def ship_class_mean(frame, columns, minimum_ships=1):
    by_ship = frame.groupby(['class_id', 'mmsi'])[columns].mean()
    counts = by_ship.groupby('class_id').size()
    retained = counts[counts >= minimum_ships].index
    keep = by_ship.index.get_level_values('class_id').isin(retained)
    reduced = by_ship.loc[keep]
    values = reduced.groupby('class_id').mean().mean()
    return values, dict(ships=len(reduced), classes=len(retained),
                       min_class_ships=int(counts.loc[retained].min()) if len(retained) else 0)


def account_pairs(pairs, scores, p50, p18, bias, single_bias):
    t = pairs.copy()
    a, b, y = t.a.to_numpy(), t.b.to_numpy(), t.class_id.to_numpy()
    np.testing.assert_allclose(bias[a], bias[b], atol=0, rtol=0)
    other = scores[b, 0].copy()
    other[np.arange(len(t)), y] = -np.inf
    rival = other.argmax(1)
    harmed = (scores[b, 0].argmax(1) == y) & (scores[b, 2].argmax(1) != y)
    rival[harmed] = scores[b[harmed], 2].argmax(1)
    t['rival'] = rival
    t['single_margin_b_fixed_rival'] = margin(scores[b, 0], y, rival)
    t['dual_margin_b_fixed_rival'] = margin(scores[b, 2], y, rival)
    t['r50_reweight_term'] = margin(p50[b]-(scores[b, 0]-single_bias[b]), y, rival)
    t['intercept_term'] = margin(bias[b]-single_bias[b], y, rival)
    t['old_route_and_intercept_term'] = t.r50_reweight_term+t.intercept_term
    t['r18_added_term'] = margin(p18[b], y, rival)
    delta = t.old_route_and_intercept_term+t.r18_added_term
    np.testing.assert_allclose(delta, t.dual_margin_b_fixed_rival-t.single_margin_b_fixed_rival,
                               atol=1e-5, rtol=2e-5)
    t['r50_temporal_term'] = margin(p50[b]-p50[a], y, rival)
    t['r18_temporal_term'] = margin(p18[b]-p18[a], y, rival)
    true_change = margin(scores[b, 2]-scores[a, 2], y, rival)
    np.testing.assert_allclose(t.r50_temporal_term+t.r18_temporal_term, true_change,
                               atol=1e-5, rtol=2e-5)
    variants = hybrid_scores(p50, p18, bias, a, b)
    for name, value in variants.items():
        t[name+'_prediction'] = value.argmax(1)
        t[name+'_correct'] = (value.argmax(1) == y).astype(float)*100
    np.testing.assert_array_equal(t.actual_a_prediction, t.dual_pred_a)
    np.testing.assert_array_equal(t.actual_b_prediction, t.dual_pred_b)
    for name in ('r50_previous', 'r18_previous'):
        t[name+'_net'] = t[name+'_correct']-t.actual_b_correct
        t[name+'_rescue'] = 100.*((t[name+'_correct'] == 100) & (t.actual_b_correct == 0))
        t[name+'_harm'] = 100.*((t[name+'_correct'] == 0) & (t.actual_b_correct == 100))
    t['damaged'] = t.r50_pred_a.eq(y) & t.r50_pred_b.ne(y)
    t['rescued_damage'] = t.damaged & t.dual_pred_b.eq(y)
    t['r18_also_correct_b'] = t.r18_pred_b.eq(y)
    t['r50_relation_worsened'] = t.r50_relation_margin_b.lt(t.r50_relation_margin_a)
    t['r50_relation_lost'] = t.r50_relation_margin_a.gt(0) & t.r50_relation_margin_b.le(0)
    t['r18_relation_positive_b'] = t.r18_relation_margin_b.gt(0)
    t['r18_evidence_retained'] = t.r18_relation_margin_a.gt(0) & t.r18_relation_margin_b.gt(0)
    t['r18_term_supports_true'] = t.r18_added_term.gt(0)
    no_r18 = p50[b]+bias[b]
    t['r18_removal_breaks_rescue'] = t.rescued_damage & (no_r18.argmax(1) != y)
    t['r18_temporal_offsets_r50'] = t.r50_temporal_term.lt(0) & t.r18_temporal_term.gt(0)
    t.to_csv(OUT/'pair_accounting.csv', index=False)
    return t


def summarize(t):
    fields = ['r50_relation_worsened', 'r50_relation_lost', 'r18_relation_positive_b',
              'r18_evidence_retained', 'r18_also_correct_b', 'rescued_damage',
              'r18_term_supports_true', 'r18_temporal_offsets_r50']
    rows = []
    for group in GROUPS:
        all_pairs = t[t.group.eq(group)]
        for cohort, subset in [('all', all_pairs), ('damaged', all_pairs[all_pairs.damaged]),
                               ('rescued_damage', all_pairs[all_pairs.rescued_damage])]:
            q = subset.copy()
            q[fields] = 100*q[fields].astype(float)
            for min_ships in (1, 5):
                values, count = ship_class_mean(q, fields, min_ships)
                rows.append(dict(group=group, cohort=cohort, pairs=len(q),
                    minimum_ships_per_class=min_ships, **count, **values.to_dict(),
                    raw_rescued_pairs=int(subset.rescued_damage.sum()),
                    raw_removal_breaks=int(subset.r18_removal_breaks_rescue.sum())))
    funnel = pd.DataFrame(rows)
    funnel.to_csv(OUT/'transition_summary.csv', index=False)
    swaps, supported_swaps = [], []
    sf = [n+'_'+m for n in ('r50_previous', 'r18_previous') for m in ('net', 'rescue', 'harm')]
    for group, part in t.groupby('group'):
        if group not in GROUPS:
            continue
        values, count = ship_class_mean(part, sf)
        swaps.append(dict(group=group, **count, **values.to_dict()))
        values, count = ship_class_mean(part, sf, 5)
        supported_swaps.append(dict(group=group, **count, **values.to_dict()))
    swaps = pd.DataFrame(swaps)
    swaps.to_csv(OUT/'swap_summary.csv', index=False)
    pd.DataFrame(supported_swaps).to_csv(OUT/'swap_supported_summary.csv', index=False)
    class_rows = t[t.group.isin(GROUPS)].groupby(['group', 'class_id', 'mmsi'])[sf].mean()
    class_rows.groupby(['group', 'class_id']).mean().to_csv(OUT/'swap_class_summary.csv')
    relation = []
    for group in GROUPS:
        sub = t[t.group.eq(group) & t.damaged]
        for retained, part in sub.groupby('r18_relation_positive_b'):
            q = part.copy()
            q['rescue_rate'] = q.rescued_damage.astype(float)*100
            values, count = ship_class_mean(q, ['rescue_rate'])
            relation.append(dict(group=group, r18_relation_positive_b=bool(retained),
                pairs=len(q), **count, rescue_rate=values['rescue_rate']))
    relation = pd.DataFrame(relation)
    relation.to_csv(OUT/'relation_rescue_association.csv', index=False)
    association_classes, standardized = [], []
    for group in GROUPS:
        sub = t[t.group.eq(group) & t.damaged].copy()
        sub['rescue_rate'] = sub.rescued_damage.astype(float)*100
        values = sub.groupby(['r18_relation_positive_b', 'class_id', 'mmsi']).rescue_rate.mean()
        means = values.groupby(['r18_relation_positive_b', 'class_id']).agg(['mean', 'size'])
        for (flag, cls), record in means.iterrows():
            association_classes.append(dict(group=group, r18_relation_positive_b=bool(flag),
                class_id=int(cls), ships=int(record['size']), rescue_rate=record['mean']))
        counts = means['size'].unstack('r18_relation_positive_b').reindex(columns=[False, True]).fillna(0)
        common = counts.index[counts.min(axis=1) >= 5]
        for flag in (False, True):
            if not len(common):
                continue
            subset = means.loc[flag].loc[common]
            standardized.append(dict(group=group, r18_relation_positive_b=flag,
                classes=len(common), class_ids=','.join(map(str, common)),
                ships=int(subset['size'].sum()), rescue_rate=float(subset['mean'].mean()),
                minimum_ships_per_class_and_condition=5))
    pd.DataFrame(association_classes).to_csv(OUT/'relation_rescue_by_class.csv', index=False)
    pd.DataFrame(standardized).to_csv(OUT/'relation_rescue_common_classes.csv', index=False)
    # These subsets name measured differences, not isolated physical factors.
    conditions = dict(incidence_only_far=t.angle_only_far,
        pass_change_near_incidence=t.same_platform & ~t.same_pass & t.delta_incidence.le(2),
        relative_near_observed=t.delta_relative_direction.le(10),
        relative_far_observed=t.delta_relative_direction.ge(25))
    condition_rows, supported_conditions = [], []
    for name, mask in conditions.items():
        for group in GROUPS:
            q = t[mask & t.group.eq(group)].copy()
            if q.empty:
                continue
            q['damage_rate'] = q.damaged.astype(float)*100
            values, count = ship_class_mean(q, ['damage_rate', 'dual_net', *sf])
            condition_rows.append(dict(condition=name, group=group, pairs=len(q), **count, **values.to_dict()))
            values, count = ship_class_mean(q, ['damage_rate', 'dual_net', *sf], 5)
            supported_conditions.append(dict(condition=name, group=group, candidate_pairs=len(q),
                **count, **values.to_dict(), minimum_ships_per_class=5))
    pd.DataFrame(condition_rows).to_csv(OUT/'condition_subsets.csv', index=False)
    pd.DataFrame(supported_conditions).to_csv(OUT/'condition_supported_subsets.csv', index=False)
    return funnel, swaps, relation


def main():
    OUT.mkdir(exist_ok=True)
    write('PROTOCOL.json', dict(date='2026-10-10', parent=str(BASE), no_refitting=True,
        score_accounting='Exact fixed-rival logit difference; old-route plus intercept change, and added R18 evidence',
        temporal_accounting='Same fold/bias: dual(b)-dual(a)=R50 path change+R18 path change',
        swaps='Four real-feature combinations: (a50,a18),(a50,b18),(b50,a18),(b50,b18). Computational intervention only; hybrids need not be physically realizable.',
        source_mean_removal='Set R18 PCA coordinates to0 under the existing dual head; not a newly trained fair single-backbone baseline',
        relation='Existing64 case class-mean dot similarities; descriptive relation, not the actual ridge coefficient contribution',
        subgroup='Condition and class summaries plus >=5 ships per class; no changed threshold or outcome-based model choice',
        cases='Fixed median-distance selection inside rescued-damage and harm strata; illustrative, not prevalence estimates',
        limitation='Same inherited identity-held frozen probes, not main model LOPO/64-label performance or physical causality'))
    obs = pd.read_csv(BASE/'observations.csv', dtype={'mmsi': str}, parse_dates=['acq_time'])
    pairs = pd.read_csv(BASE/'pair_metrics.csv', dtype={'mmsi': str})
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        scores, p50, p18, bias, single_bias, nearest = reconstruct(obs)
        table = account_pairs(pairs, scores, p50, p18, bias, single_bias)
        funnel, swaps, relation = summarize(table)
        from followup_report import report
        report(obs, table, funnel, swaps, relation, nearest)
    write('COMPLETE.json', dict(seconds=time.perf_counter()-started, observations=len(obs),
        all_pairs=len(table), primary_pairs=int(table.group.isin(GROUPS).sum()),
        score_and_relation_replay_verified=True, no_new_training=True))
    print(funnel[(funnel.cohort == 'damaged') & (funnel.minimum_ships_per_class == 1)].to_string(index=False), flush=True)
    print(swaps.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
