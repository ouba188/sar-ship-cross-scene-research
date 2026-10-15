"""Post-freeze standalone readout; descriptive tables, no policy reselection."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import frequency_specialist_report as specialist
from frequency_band_core import BANDS, unit_bands
from frequency_band_report import STAGES, markdown_table
from frequency_reliability_core import coefficients
from frequency_reliability_run import statistics
from prepare_learned_frequency import ROOT, OUT as EXP, load_npz

OUT = ROOT/'frequency_standalone_20261009'
FAMILIES = ('old', 'independent', 'standalone', 'pair')
MODES = ('learned', *(f'{f}_{b}' for f in FAMILIES for b in ('equal', *BANDS)),
         *(f'{f}_support_{s}' for f in FAMILIES for s in ('static', 'local')), 'source_selected')
SUFFIXES = ('equal', 'support_static', 'support_local')
VALUES = ['oa', 'ba', 'macro_f1_8']
INDEX = ['dataset', 'task', 'draw', 'stage']
GROUP = ['dataset', 'stage', 'suffix', 'comparison']
DELTAS = ['delta_'+v for v in VALUES]


def matched_report(output):
    metrics = pd.read_csv(output/'metrics.csv', dtype={'dataset': str, 'task': str})
    records = []
    for suffix in SUFFIXES:
        frames = {f: metrics.loc[metrics['mode'].eq(f'{f}_{suffix}'), INDEX+['n', *VALUES]]
                  .set_index(INDEX, verify_integrity=True).sort_index() for f in FAMILIES}
        reference = frames['independent']
        for family in ('old', 'standalone', 'pair'):
            frame = frames[family]
            if not frame.index.equals(reference.index) or not frame['n'].equals(reference['n']):
                raise ValueError(f'Unmatched task/draw/stage: {family}/{suffix}')
            delta = (frame[VALUES]-reference[VALUES]).rename(columns=dict(zip(VALUES, DELTAS)))
            records.append(delta.reset_index().assign(
                suffix=suffix, comparison=family+'-independent'))
    paired = pd.concat(records, ignore_index=True)
    task = paired.groupby([*GROUP, 'task'])[DELTAS].mean()
    summary = task.groupby(GROUP)[DELTAS].mean()
    draws = paired.groupby([*GROUP, 'draw'])[DELTAS].mean()
    summary['tasks'] = task.groupby(GROUP).size()
    summary['task_draws'] = paired.groupby(GROUP).size()
    summary['draws'] = draws.groupby(level=GROUP).size()
    draws['tasks'] = paired.groupby([*GROUP, 'draw']).size()
    for value in VALUES:
        positive = 'positive_'+value
        paired[positive] = paired['delta_'+value] > 0
        draws['positive_tasks_'+value] = paired.groupby([*GROUP, 'draw'])[positive].sum()
        summary['positive_task_draws_'+value] = paired.groupby(GROUP)[positive].sum()
        summary['positive_draws_'+value] = (draws['delta_'+value] > 0).groupby(level=GROUP).sum()
    paired.to_csv(output/'matched_metrics.csv', index=False)
    task.reset_index().to_csv(output/'matched_task_means.csv', index=False)
    draws.reset_index().to_csv(output/'matched_draw_metrics.csv', index=False)
    summary = summary.reset_index()
    summary.to_csv(output/'matched_summary.csv', index=False)
    return summary


def source_probe(folders, output):
    records = []
    for folder in folders:
        info = json.loads((folder/'PREPARED.json').read_text())
        meta = dict(dataset=info['dataset'], task=info['task'])
        source = output/'source'/info['dataset']/info['task']
        paths = sorted(source.glob('episode_*.npz'))
        if not (source/'ENCODED.json').is_file() or not paths:
            raise ValueError(f'Missing encoded source episodes: {source}')
        episodes = sorted((folder/'episodes').glob('*.npz'))
        for path in paths:
            episode = int(path.stem.split('_')[-1])
            z, ep = load_npz(path), load_npz(episodes[episode])
            ix = z['indices']
            sy, y = ep['labels'][ix[:64]], ep['labels'][ix[64:]]
            conditions = ep['conditions'][ix]
            condition = np.exp(-np.sum((conditions[:, None]-conditions[None, :64])**2, axis=2)/.5)
            prior = np.zeros((64, 8), dtype=float)
            for family in FAMILIES[1:]:
                bands = unit_bands(z[family])
                values = []
                for b, band in enumerate(BANDS):
                    gram = .25*(bands[:, b]@bands[:64, b].T)*condition
                    score = gram@coefficients(gram[:64], sy, prior)
                    oa, ba, _ = statistics(y, score[64:].argmax(1))
                    values.append((oa, ba))
                    records.append(dict(**meta, episode=episode, family=family, band=band,
                                        n=len(y), oa=oa, ba=ba))
                oa, ba = np.mean(values, axis=0)
                records.append(dict(**meta, episode=episode, family=family, band='mean',
                                    n=len(y), oa=oa, ba=ba))
        print(f'SOURCE STANDALONE PROBE {info["dataset"]}/{info["task"]}: {len(paths)} episodes', flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output/'source_standalone_probe.csv', index=False)
    keys = ['dataset', 'task', 'family', 'band']
    task = frame.groupby(keys)[['oa', 'ba']].mean()
    task['episodes'] = frame.groupby(keys).size()
    task['query_events'] = frame.groupby(keys)['n'].sum()
    task.reset_index().to_csv(output/'source_standalone_probe_task_means.csv', index=False)
    keys = ['dataset', 'family', 'band']
    summary = task.groupby(keys)[['oa', 'ba']].mean()
    summary['tasks'] = task.groupby(keys).size()
    summary[['episodes', 'query_events']] = task.groupby(keys)[['episodes', 'query_events']].sum()
    summary = summary.reset_index()
    summary.to_csv(output/'source_standalone_probe_summary.csv', index=False)
    return summary


def report(folders=None, output=OUT):
    output = Path(output)
    folders = ([p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
               if folders is None else [Path(p) for p in folders])
    # The reused report gates all 25 predictions and source selections before label reads.
    saved = specialist.OUT, specialist.FAMILIES, specialist.MODES
    try:
        specialist.OUT, specialist.FAMILIES, specialist.MODES = output, FAMILIES, MODES
        specialist.report(folders=folders, output=output)
    finally:
        specialist.OUT, specialist.FAMILIES, specialist.MODES = saved
    notes = [
        '# Frozen frequency standalone results',
        'Families: old, independent, standalone, pair. Training definitions belong to the main '
        'experiment records. All 30 modes and the original specialist CSV statistics are retained.',
        'All 25 matching target tasks must be COMPLETE with policies present, and all source '
        'SELECTION.json rules are loaded before target-label reporting. Frozen source_selected '
        'predictions, row IDs, supports and support labels are checked by the reused report. '
        'No policies, predictions, training outputs or source selections are changed.',
        'OA/BA/macro_f1_8 are percentages; deltas are percentage points. BA uses classes present '
        'in each evaluation set; macro F1 uses eight classes. Target summaries average five '
        'draws within task, then tasks equally within dataset. Base has no expert handoff; '
        'full uses each mode\'s route; frozen_route uses learned admissions.',
        'Original metrics.csv, task_means.csv, summary.csv, draw_metrics.csv, class_metrics.csv '
        'and events.csv cover all stages. Their deltas are relative to same-stage learned. '
        'Complementarity, band_events and rescue_overlaps CSVs retain base/frozen_route '
        'details and summaries. Oracles use target labels and are nondeployable; '
        'union_minus_best_single_oracle measures the union against the best single-band oracle. '
        'Rescue/overlap counts accumulate query-draw events, including repeated samples. '
        'source_selection.csv retains frozen rules and source validation deltas.',
        'This report is descriptive and makes no causal or training-effectiveness conclusions.'
    ]
    # Replace the reused report's hardcoded old-experiment family notes immediately.
    markdown = output/'REPORT_ZH.md'
    markdown.write_text('\n\n'.join(notes)+'\n', encoding='utf-8')
    matched = matched_report(output)
    probe = source_probe(folders, output)
    summary = pd.read_csv(output/'summary.csv')
    controls = ['learned', *(f'{f}_equal' for f in FAMILIES), 'source_selected']
    show = summary[summary['mode'].isin(controls)].pivot(
        index=['dataset', 'mode'], columns='stage', values='ba').reindex(
        columns=[s for s, _ in STAGES]).reset_index()
    comparison = matched[matched.stage.eq('full')][['dataset', 'suffix', 'comparison',
        'delta_oa', 'delta_ba', 'positive_draws_oa', 'positive_draws_ba', 'draws']]
    notes.extend([
        '## Target BA', markdown_table(show),
        '## Matched differences against independent (full)', markdown_table(comparison),
        'matched_metrics.csv subtracts independent for the same dataset/task/draw/stage and '
        'the same equal, support_static or support_local suffix before aggregation. '
        'matched_task_means.csv, matched_draw_metrics.csv and matched_summary.csv retain '
        'all three stages and OA/BA/macro F1. positive_draws counts strictly positive dataset '
        'means over the five support draws; positive_task_draws counts positive individual '
        'task-draw differences. Counts are descriptive, not independent replications or tests.',
        '## Source standalone ridge probe',
        markdown_table(probe[probe.band.eq('mean')]),
        'Every available encoded source episode for all 25 tasks is used for independent, '
        'standalone and pair. Labels come only from the original source episodes: the first '
        '64 encoded indices fit the ridge and remaining indices supply evaluation labels. '
        'The kernel is 0.25 times unit-band similarity times exp(-squared condition distance/0.5). '
        'The existing class-balanced ridge helper fixes strength 8; the prior is zero and '
        'there is no shared kernel. OA/BA use all frozen source queries and eight-class argmax; '
        'the training-only eligibility filter is not applied to this readout. '
        'This diagnostic never selects a model or rule.',
        'source_standalone_probe.csv contains episode-level OA/BA for LL/LH/HL/HH and mean. '
        'The mean row is the arithmetic mean of four band accuracies, not a fused predictor. '
        'source_standalone_probe_task_means.csv averages episodes within task; '
        'source_standalone_probe_summary.csv averages tasks equally within dataset and '
        'retains every band plus mean. Episode/query-event counts include reuse across tasks; '
        'these are source diagnostics, not independent representation-held evaluations.'
    ])
    markdown.write_text('\n\n'.join(notes)+'\n', encoding='utf-8')
    print(f'STANDALONE REPORT: {output}', flush=True)


def main():
    report()


if __name__ == '__main__':
    main()
