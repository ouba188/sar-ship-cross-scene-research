"""Analyze frozen target predictions; no fitting or prediction regeneration.

Run with python -B analyze_support_completion_target.py after all inputs freeze.
Metrics and contributions are percentages / percentage points, respectively.
"""

from full_teacher_data import BASE, SOURCE  # Sets BLAS environment before numpy.

import json
import numpy as np
import pandas as pd


OUT = BASE / 'support_completion_target_20261012'
PORTS = ('Los Angeles', 'Port Klang', 'Santos')
READOUTS = ('base', 'expert_raw', 'expert_corrected',
            'fusion_raw', 'fusion_corrected')
DRAWS = 5
CONTEXTS = len(PORTS) * DRAWS
OUTCOMES = ('improved', 'unchanged_correct', 'unchanged_wrong', 'harmed')


def outcome(before_correct, after_correct):
    """Partition correctness transitions, including wrong-to-different-wrong."""
    return np.select(
        [~before_correct & after_correct, before_correct & after_correct,
         ~before_correct & ~after_correct],
        OUTCOMES[:3], default='harmed')


def recalls(truth, pred):
    counts = np.bincount(truth, minlength=8)
    hits = np.bincount(truth[pred == truth], minlength=8)
    recall = np.divide(100. * hits, counts, out=np.full(8, np.nan),
                       where=counts > 0)
    return counts, hits, recall


def metrics(truth, pred):
    counts, _, recall = recalls(truth, pred)
    return dict(oa=100. * float(np.mean(pred == truth)) if len(truth) else np.nan,
                ba=float(recall[counts > 0].mean()) if len(truth) else np.nan)


def event_counts(frame):
    return dict(events=len(frame), unique_rows=int(frame.manifest_row.nunique()),
                unique_ships=int(frame.mmsi.nunique()),
                missing_identity_events=int(frame.mmsi.isna().sum()),
                missing_identity_rows=int(frame.loc[frame.mmsi.isna(),'manifest_row'].nunique()))


def records(frame):
    # Pandas emits JSON null for undefined recalls in absent classes.
    return json.loads(frame.to_json(orient='records'))


def main():
    manifest = pd.read_csv(SOURCE / 'manifest.csv',
                           usecols=['class_id', 'mmsi', 'port'], dtype={'mmsi': str})
    metric_rows, class_rows, event_frames, contribution_frames = [], [], [], []
    support_rows, contexts, coverage_rows = [], [], []
    for port in PORTS:
        with np.load(OUT / port / 'predictions.npz') as data:
            rows = data['rows']
            meta = manifest.iloc[rows].reset_index(drop=True)
            truth_all = meta.class_id.to_numpy(dtype=int)
            ships_all = meta.mmsi.to_numpy()
            for draw in range(DRAWS):
                support = data['support'][draw]
                for local in support:
                    support_rows.append(dict(port=port, draw=draw, local_row=int(local),
                        manifest_row=int(rows[local]), mmsi=ships_all[local],
                        class_id=int(truth_all[local])))
                mask = data['eval_masks'][draw]
                truth, ships = truth_all[mask], ships_all[mask]
                pred = {name: data[name][draw, mask].argmax(1) for name in READOUTS}
                correct = {name: value == truth for name, value in pred.items()}
                contexts.append((port, draw, truth, ships, pred))
                counts = np.bincount(truth, minlength=8)
                weight = 100. / ((counts > 0).sum() * counts[truth])
                coverage_rows.append(dict(port=port, draw=draw, eval_rows=len(truth),
                    eval_ships=int(pd.Series(ships).nunique()),
                    present_classes=int((counts > 0).sum()), support_rows=len(support)))
                current_metrics = {name: metrics(truth, value) for name, value in pred.items()}
                current_recalls = {name: recalls(truth, value) for name, value in pred.items()}
                for name in READOUTS:
                    reference = name.replace('_corrected', '_raw')
                    met, ref, base = current_metrics[name], current_metrics[reference], current_metrics['base']
                    metric_rows.append(dict(port=port, draw=draw, readout=name,
                        reference=reference, **met,
                        delta_oa=met['oa']-ref['oa'], delta_ba=met['ba']-ref['ba'],
                        delta_oa_vs_base=met['oa']-base['oa'], delta_ba_vs_base=met['ba']-base['ba']))
                    _, hits, recall = current_recalls[name]
                    for c in range(8):
                        class_rows.append(dict(port=port, draw=draw, readout=name,
                            reference=reference, class_id=c, denominator=int(counts[c]),
                            correct=int(hits[c]), recall=recall[c],
                            delta_recall=recall[c]-current_recalls[reference][2][c],
                            delta_recall_vs_base=recall[c]-current_recalls['base'][2][c]))
                expert_outcome = outcome(correct['expert_raw'], correct['expert_corrected'])
                fusion_outcome = outcome(correct['fusion_raw'], correct['fusion_corrected'])
                frame = pd.DataFrame(dict(port=port, draw=draw,
                    local_row=np.flatnonzero(mask), manifest_row=rows[mask],
                    mmsi=ships, class_id=truth, manifest_port=meta.port.to_numpy()[mask],
                    base_correct=correct['base'], expert_outcome=expert_outcome,
                    fusion_outcome=fusion_outcome, class_denominator=counts[truth],
                    ba_row_weight_pp=weight))
                for name in READOUTS:
                    frame[name + '_pred'] = pred[name]
                    frame[name + '_correct'] = correct[name]
                for branch in ('expert', 'fusion'):
                    delta = correct[branch + '_corrected'].astype(int) - correct[branch + '_raw'].astype(int)
                    frame[branch + '_ba_contribution_pp'] = delta * weight / CONTEXTS
                    con = pd.DataFrame(dict(mmsi=ships, manifest_row=rows[mask],
                        rescue=(delta > 0).astype(int), harm=(delta < 0).astype(int),
                        ba_delta_context_pp=delta * weight))
                    con = con.groupby('mmsi', sort=True, dropna=False).agg(
                        eval_rows=('manifest_row', 'size'), rescue_events=('rescue', 'sum'),
                        harm_events=('harm', 'sum'), ba_delta_context_pp=('ba_delta_context_pp', 'sum')).reset_index()
                    con['port'], con['draw'], con['branch'] = port, draw, branch
                    con['ba_contribution_pp'] = con.ba_delta_context_pp / CONTEXTS
                    contribution_frames.append(con)
                changed = ((pred['expert_raw'] != pred['expert_corrected']) |
                           (pred['fusion_raw'] != pred['fusion_corrected']))
                event_frames.append(frame.loc[changed].copy())

    dest = OUT / 'diagnostics'
    dest.mkdir(parents=True, exist_ok=True)
    metric_table = pd.DataFrame(metric_rows)
    class_table = pd.DataFrame(class_rows)
    events = pd.concat(event_frames, ignore_index=True)
    contributions = pd.concat(contribution_frames, ignore_index=True)
    metric_table.to_csv(dest / 'metrics.csv', index=False)
    class_table.to_csv(dest / 'class_recalls.csv', index=False)
    events.to_csv(dest / 'raw_events.csv', index=False)
    pd.DataFrame(support_rows).to_csv(dest / 'support_rows.csv', index=False)
    pd.DataFrame(coverage_rows).to_csv(dest / 'coverage.csv', index=False)
    contributions.to_csv(dest / 'vessel_context_contributions.csv', index=False)
    metric_cols = ['oa', 'ba', 'delta_oa', 'delta_ba', 'delta_oa_vs_base', 'delta_ba_vs_base']
    metric_summary = pd.concat([
        metric_table.groupby(['port', 'readout'])[metric_cols].mean().reset_index(),
        metric_table.groupby('readout')[metric_cols].mean().reset_index().assign(port='ALL')], ignore_index=True)
    metric_summary.to_csv(dest / 'metric_summary.csv', index=False)
    recall_cols = ['recall', 'delta_recall', 'delta_recall_vs_base']
    class_summary = pd.concat([
        class_table.groupby(['port', 'readout', 'class_id'])[recall_cols].mean().reset_index(),
        class_table.groupby(['readout', 'class_id'])[recall_cols].mean().reset_index().assign(port='ALL')], ignore_index=True)
    class_summary.to_csv(dest / 'class_summary.csv', index=False)
    vessel_summary = contributions.groupby(['branch', 'port', 'mmsi'], dropna=False).agg(
        ba_contribution_pp=('ba_contribution_pp', 'sum'),
        rescue_events=('rescue_events', 'sum'), harm_events=('harm_events', 'sum'),
        evaluated_contexts=('draw', 'size')).reset_index()
    vessel_summary = vessel_summary.sort_values(
        ['branch', 'ba_contribution_pp', 'port', 'mmsi'], ascending=[True, False, True, True])
    vessel_summary.to_csv(dest / 'vessel_contributions.csv', index=False)
    vessel_summary.groupby(['branch', 'mmsi'], dropna=False)[['ba_contribution_pp', 'rescue_events', 'harm_events']].sum().to_csv(
        dest / 'vessel_contributions_all_ports.csv')

    counts_rows, category_rows, ship_event_frames = [], [], []
    for port in (*PORTS, 'ALL'):
        selected = events if port == 'ALL' else events[events.port == port]
        for branch in ('expert', 'fusion'):
            for kind, transition in (('rescue', 'improved'), ('harm', 'harmed')):
                subset = selected[selected[branch + '_outcome'] == transition]
                counts_rows.append(dict(port=port, branch=branch, event=kind, **event_counts(subset)))
                if port != 'ALL':
                    ship_events = subset.groupby('mmsi', dropna=False).agg(
                        events=('manifest_row', 'size'), unique_rows=('manifest_row', 'nunique'),
                        draws=('draw', 'nunique')).reset_index()
                    ship_events['port'], ship_events['branch'], ship_events['event'] = port, branch, kind
                    ship_event_frames.append(ship_events)
                if branch == 'expert':
                    for base_correct in (False, True):
                        for fusion_state in OUTCOMES:
                            cell = subset[(subset.base_correct == base_correct) &
                                          (subset.fusion_outcome == fusion_state)]
                            category_rows.append(dict(port=port, expert_event=kind,
                                base_correct=base_correct, fusion_outcome=fusion_state,
                                **event_counts(cell)))
    event_summary = pd.DataFrame(counts_rows)
    categories = pd.DataFrame(category_rows)
    event_summary.to_csv(dest / 'event_counts.csv', index=False)
    categories.to_csv(dest / 'expert_event_categories.csv', index=False)
    pd.concat(ship_event_frames, ignore_index=True).to_csv(dest / 'ship_events.csv', index=False)

    # A fixed-prediction cohort sensitivity: remove one port/vessel, never refit.
    top = vessel_summary[(vessel_summary.branch == 'fusion') & vessel_summary.mmsi.notna()].iloc[0]
    removal_rows, removal_classes = [], []
    for port, draw, truth, ships, pred in contexts:
        keep = ships != top.mmsi if port == top.port else np.ones(len(truth), dtype=bool)
        before = metrics(truth, pred['fusion_raw'])
        after = metrics(truth, pred['fusion_corrected'])
        raw = metrics(truth[keep], pred['fusion_raw'][keep])
        corrected = metrics(truth[keep], pred['fusion_corrected'][keep])
        removal_rows.append(dict(port=port, draw=draw, removed_rows=int((~keep).sum()),
            remaining_rows=int(keep.sum()), raw_oa=raw['oa'], corrected_oa=corrected['oa'],
            raw_ba=raw['ba'], corrected_ba=corrected['ba'],
            original_delta_oa=after['oa']-before['oa'], original_delta_ba=after['ba']-before['ba'],
            delta_oa=corrected['oa']-raw['oa'], delta_ba=corrected['ba']-raw['ba']))
        old_counts, _, _ = recalls(truth, pred['fusion_raw'])
        counts, raw_hits, raw_recall = recalls(truth[keep], pred['fusion_raw'][keep])
        _, new_hits, new_recall = recalls(truth[keep], pred['fusion_corrected'][keep])
        for c in range(8):
            removal_classes.append(dict(port=port, draw=draw, class_id=c,
                original_denominator=int(old_counts[c]), denominator=int(counts[c]),
                raw_correct=int(raw_hits[c]), corrected_correct=int(new_hits[c]),
                raw_recall=raw_recall[c], corrected_recall=new_recall[c],
                delta_recall=new_recall[c]-raw_recall[c]))
    removal = pd.DataFrame(removal_rows)
    removal.to_csv(dest / 'remove_top_vessel_metrics.csv', index=False)
    pd.DataFrame(removal_classes).to_csv(dest / 'remove_top_vessel_classes.csv', index=False)
    removal_summary = pd.concat([
        removal.groupby('port')[['delta_oa', 'delta_ba']].mean().reset_index(),
        pd.DataFrame([dict(port='ALL', delta_oa=removal.delta_oa.mean(), delta_ba=removal.delta_ba.mean())])], ignore_index=True)
    removal_summary.to_csv(dest / 'remove_top_vessel_summary.csv', index=False)
    summary = dict(
        protocol=dict(ports=list(PORTS), draws_per_port=DRAWS, contexts=CONTEXTS,
            support_size=64, fusion='Frozen supplied 0.5 softmax fusion; not recomputed',
            evaluation='Supplied eval_masks, manifest row positions, argmax readouts',
            aggregation='Equal context means, not pooled rows; BA averages present classes',
            reference='Corrected readout minus corresponding raw readout; base deltas separate',
            contribution='Sum of signed row correctness changes * 100 / (present classes * class count * 15)',
            unique_counts='Distinct global rows / known MMSIs across overlapping draws; missing identities counted separately, not treated as one vessel; categories overlap after deduplication',
            events='All expert or fusion top-1 changes; unchanged_wrong includes wrong-to-different-wrong',
            limitations='Descriptive numerical attribution, not causal identification or independent-draw uncertainty'),
        metrics=records(metric_summary), event_counts=records(event_summary),
        expert_event_categories=records(categories),
        top_fusion_vessel=dict(port=top.port, mmsi=str(top.mmsi),
            ba_contribution_pp=float(top.ba_contribution_pp),
            selection='Largest signed final-fusion port/vessel contribution; port then MMSI breaks ties',
            removal='Only this vessel within its selected port; fixed supports and predictions'),
        remove_top_vessel=records(removal_summary),
        removal_defined_contexts=int(removal.delta_ba.notna().sum()),
        original_mean_fusion_delta_ba=float(removal.original_delta_ba.mean()),
        fusion_contribution_sum_pp=float(vessel_summary.loc[vessel_summary.branch == 'fusion', 'ba_contribution_pp'].sum()))
    np.testing.assert_allclose(summary['fusion_contribution_sum_pp'],
        summary['original_mean_fusion_delta_ba'],atol=1e-10,rtol=0)
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(metric_summary.to_string(index=False))
    print('Diagnostics:', dest)


if __name__ == '__main__':
    main()
