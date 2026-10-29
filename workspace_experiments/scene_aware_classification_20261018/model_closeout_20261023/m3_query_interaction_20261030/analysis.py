"""CPU development analysis. Run only after the main worker sends READY.

python -B analysis.py --dataset all --seeds 17
Add --gallery for at most four interaction-vs-CE cases per dataset.
Final all-order statistics: --dataset all --seeds 17 29 43 (omit --gallery).
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUT = HERE / 'analysis'
CE_ROOT = HERE.parent / 'm3_pairwise_repair_20261030'
spec = importlib.util.spec_from_file_location('interaction_pairwise_helpers', CE_ROOT / 'analysis.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
replay = previous.replay
read, dump, metric, load_pack = previous.read, previous.dump, previous.metric, previous.load_pack
METRICS, STATES, SEEDS = previous.METRICS, previous.STATES, previous.SEEDS
ARMS = ('query_only', 'ref_only', 'interaction')
COMPARISONS = tuple((a, b) for a in ARMS for b in ('saved_baseline', 'ce'))
COMPARISONS += (('interaction', 'query_only'), ('interaction', 'ref_only'))


def load_dataset(dataset, requested):
    if dataset not in ('private', 'oss2'):
        raise ValueError('Only private development and OSS2 validation are supported')
    manifest = replay.previous.manifest_dir(dataset)
    group = replay.prior.rows(manifest / ('development.json' if dataset == 'private' else 'validation.json'))
    refs = replay.prior.rows(manifest / 'references.json')
    classes = replay.prior.PRIVATE if dataset == 'private' else replay.prior.PUBLIC
    y = np.array([r['label'] for r in group])
    scores, receipts, pending, paths = {}, {}, [], []
    for seed in requested:
        for arm in ('ce', *ARMS):
            root = CE_ROOT if arm == 'ce' else HERE
            rp = root / dataset / 'training' / f'{arm}_s{seed}.json'
            pp = root / dataset / 'predictions' / f'{arm}_s{seed}.npz'
            missing = [str(p) for p in (rp, pp) if not p.is_file()]
            if missing:
                pending.append(dict(arm=arm, seed=seed, missing_files=missing))
                continue
            report = read(rp)
            pack, absent = load_pack(pp, group, classes)
            if absent:
                pending.append(dict(arm=arm, seed=seed, **absent))
                continue
            for key, value in (('dataset', dataset), ('arm', arm), ('seed', seed), ('query_count', len(group))):
                if report.get(key) != value:
                    raise ValueError(f'{rp}: unmatched {key}')
            if report.get('evaluation_used', False):
                raise ValueError(f'{rp}: final evaluation used')
            if list(map(str, report.get('reference_ids', []))) != [str(r['id']) for r in refs]:
                raise ValueError(f'{rp}: actual reference bank differs')
            for field, phase in (('score', 'after'), ('baseline', 'baseline')):
                actual = metric(pack[field], y)
                for key in METRICS:
                    saved = report.get(phase, {}).get(key.lower())
                    if saved is not None and not np.isclose(actual[key], saved, rtol=0, atol=1e-6):
                        raise ValueError(f'{rp}: {phase}/{key} mismatch')
            if ('saved_baseline', seed) in scores and not np.array_equal(scores['saved_baseline', seed], pack['baseline']):
                raise ValueError(f'{dataset} s{seed}: unmatched saved original')
            if arm != 'ce' and ('ce', seed) in receipts:
                for key in ('lr', 'steps', 'batch_size', 'source_count', 'reference_ids'):
                    if report.get(key) != receipts['ce', seed].get(key):
                        raise ValueError(f'{rp}: CE comparison has unmatched {key}')
            scores[arm, seed] = pack['score']
            scores['saved_baseline', seed] = pack['baseline']
            receipts[arm, seed] = report
            paths.extend(map(str, (rp, pp)))
    return dict(dataset=dataset, group=group, refs=refs, classes=classes, y=y,
                scores=scores, receipts=receipts, pending=pending, input_paths=paths,
                manifest=str(manifest), requested=list(requested), seeds=sorted({s for _, s in receipts}))


def summarize(d):
    out = OUT / d['dataset']
    out.mkdir(parents=True, exist_ok=True)
    endpoints, paired, tradeoffs, events = previous.paired_tables(d, ('ce', *ARMS), COMPARISONS)
    means = []
    for arm, reference in COMPARISONS:
        rows = [r for r in paired if (r['arm'], r['reference']) == (arm, reference)]
        selected = [e for e in events if (e['arm'], e['reference']) == (arm, reference)]
        seeds = [r['seed'] for r in rows]
        result = dict(arm=arm, reference=reference, seeds=seeds,
                      requested_complete=seeds == d['requested'], three_seed_complete=set(seeds) == set(SEEDS))
        for key in METRICS:
            values = [r['delta_' + key] for r in rows]
            result['delta_' + key] = float(np.mean(values)) if values else None
            result['sd_' + key] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            result['improved_' + key] = sum(v > 0 if key in ('OA', 'BA') else v < 0 for v in values)
        for state in STATES:
            subset = [e for e in selected if e['state'] == state]
            result[state + '_events'] = len(subset)
            result[state + '_queries'] = len({e['id'] for e in subset})
        means.append(result)
    endpoint_means = []
    for arm in ('saved_baseline', 'ce', *ARMS):
        rows = [r for r in endpoints if r['arm'] == arm]
        result = dict(arm=arm, seeds=[r['seed'] for r in rows])
        for key in METRICS:
            values = [r[key] for r in rows]
            result[key] = float(np.mean(values)) if values else None
            result['sd_' + key] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        endpoint_means.append(result)
    summary = dict(dataset=d['dataset'], scope='development', requested=d['requested'],
        query_count=len(d['y']), available_by_arm={a: previous.paired_seeds(d, a, 'saved_baseline') for a in ('ce', *ARMS)},
        pending=d['pending'], input_paths=d['input_paths'], paired_means=means, endpoint_means=endpoint_means,
        requested_complete=all(r['requested_complete'] for r in means),
        three_seed_complete=all(r['three_seed_complete'] for r in means),
        notes=[
            'CE is the unchanged previous pairwise round endpoint; new arms retain the frozen original M3.',
            'The saved original logits must match exactly across CE and available new arms after ID alignment.',
            'Receipt LR, steps, batch size, source count and reference IDs are compared with CE. The main implementation reads the previous pairwise exposures/schedule_s{seed}.json directly; no new source coverage audit is performed. See analysis/SOURCE_CONTEXT.md.',
            'OA/BA and differences are percent/percentage points; NLL/Brier are unscaled, lower is better.',
            'BA uses present query classes. Missing arms/seeds stay pending; one-seed SD is null.',
            'Repeated query events across seeds are not independent observations or vessels.',
            'Mean and sample SD summarize available matched training orders sharing the original initialization source; they are not variation across independently pretrained models.',
            'Arm comparisons measure trained-system differences; supplied dependence interventions require separate paired analysis.',
            'Cases are outcome-selected diagnostics and require main-worker visual review; no physical cause is inferred.'])
    dump(out / 'summary.json', summary)
    for name, rows in (('metrics', endpoints), ('endpoint_means', endpoint_means), ('paired', paired), ('paired_means', means),
                       ('class_tradeoffs', tradeoffs), ('events', events)):
        replay.previous.write_csv(out / (name + '.csv'), rows)
    lines = [f"# {d['dataset']} query interaction development analysis", '',
             f"Queries: {len(d['y'])}; available seeds by arm: {summary['available_by_arm']}.", '']
    replay.prior.table(lines, ['Arm - reference', 'Orders', 'Mean delta OA', 'SD OA', 'Mean delta BA', 'SD BA', 'Mean delta NLL', 'Mean delta Brier'],
        [[r['arm'] + ' - ' + r['reference'], r['seeds'], r['delta_OA'], r['sd_OA'], r['delta_BA'], r['sd_BA'],
          r['delta_NLL'], r['delta_Brier']] for r in means])
    replay.prior.table(lines, ['Arm - reference', 'Seed', 'Delta OA', 'Delta BA', 'Delta NLL', 'Delta Brier', 'Rescue', 'Harm', 'CC', 'WW'],
        [[r['arm'] + ' - ' + r['reference'], r['seed'], *[r['delta_' + k] for k in METRICS],
          *[r[k] for k in STATES]] for r in paired])
    lines += ['', *summary['notes'], '', 'Pending:', json.dumps(d['pending'], indent=2)]
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def gallery(d, arm='interaction', reference='ce'):
    seeds = previous.paired_seeds(d, arm, reference)
    if not seeds:
        return dict(status='pending', arm=arm, reference=reference)
    replay.OUT = OUT
    replay.ARMS = tuple(a for a in ('ce', *ARMS) if all((a, s) in d['scores'] for s in seeds))
    # Existing renderer selects one distinct query per state, native VV/VH and actual legal refs.
    return replay.gallery(dict(d, seeds=seeds), arm, reference)


def dependence_summary(d):
    """Analyze supplied writer controls without importing the model or retraining."""
    g = dict(d, scores=dict(d['scores']))
    diagnostics, pending, paths = [], [], []
    controls = tuple(a + '_' + c for a in ARMS for c in ('off', 'uniform'))
    comparisons = tuple((a + '_' + c, b) for a in ARMS for c in ('off', 'uniform')
                        for b in (a, 'saved_baseline'))
    for arm in ARMS:
        for seed in previous.paired_seeds(d, arm, 'saved_baseline'):
            normal, baseline = d['scores'][arm, seed], d['scores']['saved_baseline', seed]
            for control in ('off', 'uniform'):
                name = arm + '_' + control
                pp = HERE / d['dataset'] / 'predictions' / f'{name}_s{seed}.npz'
                report = d['receipts'][arm, seed].get('controls', {}).get(control)
                if not pp.is_file() or report is None:
                    pending.append(dict(arm=arm, control=control, seed=seed,
                        prediction=str(pp), prediction_missing=not pp.is_file(), receipt_missing=report is None))
                    continue
                pack, absent = load_pack(pp, d['group'], d['classes'])
                if absent:
                    pending.append(dict(arm=arm, control=control, seed=seed, **absent))
                    continue
                if not np.array_equal(pack['baseline'], baseline):
                    raise ValueError(f'{pp}: unmatched control baseline')
                score = pack['score']
                values = metric(score, d['y'])
                for key in METRICS:
                    if not np.isclose(values[key], report['metrics'][key.lower()], rtol=0, atol=1e-6):
                        raise ValueError(f'{pp}: control receipt {key} mismatch')
                difference = float(np.abs(score - normal).max())
                changed = int((score.argmax(1) != normal.argmax(1)).sum())
                if not np.isclose(difference, report['max_logit_difference'], rtol=0, atol=1e-6):
                    raise ValueError(f'{pp}: control receipt logit difference mismatch')
                if changed != report['changed']:
                    raise ValueError(f'{pp}: control receipt changed count mismatch')
                restore_error = float(np.abs(score - baseline).max())
                if control == 'off' and restore_error >= 1e-4:
                    raise ValueError(f'{pp}: off does not restore saved original within 1e-4')
                diagnostics.append(dict(arm=arm, control=control, seed=seed, query_count=len(d['y']),
                    max_logit_difference_vs_normal=difference, changed_vs_normal=changed,
                    max_logit_difference_vs_baseline=restore_error,
                    changed_vs_baseline=int((score.argmax(1) != baseline.argmax(1)).sum()),
                    off_restoration_passed=restore_error < 1e-4 if control == 'off' else None,
                    **values))
                g['scores'][name, seed] = score
                paths.append(str(pp))
    endpoints, paired, tradeoffs, events = previous.paired_tables(g, (*ARMS, *controls), comparisons)
    means = []
    for arm, reference in comparisons:
        rows = [r for r in paired if (r['arm'], r['reference']) == (arm, reference)]
        seeds = [r['seed'] for r in rows]
        row = dict(arm=arm, reference=reference, seeds=seeds,
                   requested_complete=seeds == d['requested'], three_seed_complete=set(seeds) == set(SEEDS))
        for key in METRICS:
            values = [r['delta_' + key] for r in rows]
            row['delta_' + key] = float(np.mean(values)) if values else None
            row['sd_' + key] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        row.update({s + '_events': sum(r[s] for r in rows) for s in STATES})
        means.append(row)
    out = OUT / d['dataset'] / 'dependence'
    out.mkdir(parents=True, exist_ok=True)
    result = dict(dataset=d['dataset'], scope='development_writer_interventions',
        diagnostics=diagnostics, paired_means=means, pending=pending, input_paths=paths,
        requested_complete=all(r['requested_complete'] for r in means),
        three_seed_complete=all(r['three_seed_complete'] for r in means),
        notes=[
            'Every delta is intervention minus reference. Rescue/harm states follow that same direction.',
            'off disables only the added writer; logits must recover the saved original within 1e-4. Any top-1 differences are reported separately.',
            'uniform replaces only the added writer reference weights with legal uniform weights, retaining transport and the frozen original M3.',
            'uniform versus normal tests sensitivity and utility of the added writer reference weighting. It does not remove reference content, transport dependence, or query dependence.',
            'query_only uniform is a negative control for the changed reference weights; numerical differences and decision changes are reported without inferring a physical mechanism.',
            'New-arm versus old CE comparisons belong in the main analysis. These controls do not update parameters or constitute separately trained candidates.',
            'Missing controls remain pending. Available-seed means are provisional; repeated query events are not independent samples.'])
    dump(out / 'summary.json', result)
    for name, rows in (('diagnostics', diagnostics), ('metrics', endpoints), ('paired', paired),
                       ('paired_means', means), ('class_tradeoffs', tradeoffs), ('events', events)):
        replay.previous.write_csv(out / (name + '.csv'), rows)
    lines = ['# Added-writer dependence controls', '', *result['notes'], '']
    replay.prior.table(lines, ['Intervention - reference', 'Orders', 'Mean delta OA', 'SD OA', 'Mean delta BA', 'SD BA', 'Mean delta NLL', 'Mean delta Brier'],
        [[r['arm'] + ' - ' + r['reference'], r['seeds'], r['delta_OA'], r['sd_OA'], r['delta_BA'], r['sd_BA'],
          r['delta_NLL'], r['delta_Brier']] for r in means])
    replay.prior.table(lines, ['Arm', 'Control', 'Seed', 'Max logit change vs normal', 'Changed vs normal',
                              'Max logit change vs baseline', 'Changed vs baseline'],
        [[r['arm'], r['control'], r['seed'], r['max_logit_difference_vs_normal'], r['changed_vs_normal'],
          r['max_logit_difference_vs_baseline'], r['changed_vs_baseline']] for r in diagnostics])
    replay.prior.table(lines, ['Intervention - reference', 'Seed', 'Delta OA', 'Delta BA', 'Delta NLL', 'Delta Brier',
                              'Rescue', 'Harm', 'CC', 'WW'],
        [[r['arm'] + ' - ' + r['reference'], r['seed'], *[r['delta_' + k] for k in METRICS],
          *[r[k] for k in STATES]] for r in paired])
    lines += ['', 'Pending:', json.dumps(pending, indent=2)]
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return dict(available_controls=len(diagnostics), pending=pending, output=str(out))


def shared_shift_summary(d):
    rows = []
    for arm in ARMS:
        for seed in previous.paired_seeds(d, arm, 'saved_baseline'):
            result = previous.shared_shift(d['scores']['saved_baseline', seed], d['scores'][arm, seed], d['y'])
            result['common_energy'] = float(len(d['y']) * np.square(result['common_class_offset']).sum())
            total = result['centered_delta_energy']
            result['residual_energy_fraction'] = result['residual_energy'] / total if total else None
            for key in METRICS:
                result['full_minus_common_' + key] = result['actual_' + key] - result['common_only_' + key]
            rows.append(dict(arm=arm, reference='saved_baseline', seed=seed, **result))
    means = []
    keys = ('centered_delta_energy', 'common_energy', 'residual_energy',
            'common_energy_fraction', 'residual_energy_fraction')
    keys += tuple(phase + '_' + key for phase in ('reference', 'actual', 'common_only', 'residual_only', 'full_minus_common')
                  for key in METRICS)
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        seeds = [r['seed'] for r in selected]
        row = dict(arm=arm, seeds=seeds, three_seed_complete=set(seeds) == set(SEEDS))
        for key in keys:
            values = [r[key] for r in selected if r[key] is not None]
            row[key] = float(np.mean(values)) if values else None
            row['sd_' + key] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        for key in ('count', 'actual_changed', 'common_changed', 'same_prediction', 'same_among_actual_changed'):
            row[key + '_events'] = sum(r[key] for r in selected)
        means.append(row)
    out = OUT / d['dataset']
    result = dict(rows=rows, means=means, three_seed_complete=all(r['three_seed_complete'] for r in means), notes=[
        'For each query, subtract the across-class mean from new-minus-original logits; common is the across-query mean of that centered increment.',
        'Residual is each query centered increment minus common. All energies are sums of squared logit increments, not averages or physical energy.',
        'Absolute total/common/residual energies accompany fractions: a larger residual fraction alone does not establish a larger useful update.',
        'Means and sample SD are over available orders; energy fractions are averaged per order, not pooled. Summed decision counts are repeated query-order events.',
        'Common-only and residual-only scores are descriptive decompositions of development predictions, not independently trained or deployable interventions.',
        'These fractions are not causal information fractions or proof of useful query/reference dependence.'])
    dump(out / 'shared_shift.json', result)
    replay.previous.write_csv(out / 'shared_shift.csv', rows)
    replay.previous.write_csv(out / 'shared_shift_means.csv', means)
    lines = ['# Centered logit increment decomposition', '', *result['notes'], '']
    replay.prior.table(lines, ['Arm', 'Orders', 'Mean total energy', 'Mean common energy', 'Mean residual energy', 'Mean common fraction',
                              'Mean full-common OA', 'Mean full-common BA'],
        [[r['arm'], r['seeds'], r['centered_delta_energy'], r['common_energy'], r['residual_energy'],
          r['common_energy_fraction'], r['full_minus_common_OA'], r['full_minus_common_BA']] for r in means])
    replay.prior.table(lines, ['Arm', 'Seed', 'Total energy', 'Common energy', 'Residual energy', 'Common fraction', 'Residual fraction'],
        [[r['arm'], r['seed'], r['centered_delta_energy'], r['common_energy'], r['residual_energy'],
          r['common_energy_fraction'], r['residual_energy_fraction']] for r in rows])
    (out / 'SHARED_SHIFT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return dict(rows=len(rows), output=str(out / 'shared_shift.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('private', 'oss2', 'all'), default='all')
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=[17])
    parser.add_argument('--gallery', action='store_true')
    args = parser.parse_args()
    if args.gallery and sorted(set(args.seeds)) != [17]:
        parser.error('Existing pilot17 galleries are preserved. Run all-order statistics without --gallery.')
    status = []
    for dataset in (('private', 'oss2') if args.dataset == 'all' else (args.dataset,)):
        d = load_dataset(dataset, sorted(set(args.seeds)))
        summary = summarize(d)
        item = dict(dataset=dataset, available_by_arm=summary['available_by_arm'], pending=d['pending'])
        item['dependence'] = dependence_summary(d)
        item['shared_shift'] = shared_shift_summary(d)
        if args.gallery:
            item['gallery'] = gallery(d)
        status.append(item)
    assert 'torch' not in sys.modules, 'CPU analysis imported model runtime'
    receipt = dict(status=status, cpu_only=True, torch_imported=False,
                   final_test_manifests_read=False, final_test_predictions_read=False,
                   source_predictions_read=False)
    dump(OUT / f'run_{args.dataset}_receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
