"""Exact two-view union head versus separate heads, with source-only selection."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import time
import numpy as np
import pandas as pd

from prepare_learned_frequency import ROOT, SOURCE, DATASETS, load_npz
from frequency_band_core import old_kernel
from frozen_comparison import folders, write, ba, OUT as REFERENCE, HEADS
from unified_head_core import GRID, STRENGTHS, score, combined, late, fit_diagnostics

OUT = ROOT/'unified_head_20261010'
PAIR = ROOT/'compact_fusion_20261010'
FAMILIES = ('unified', 'late_softmax', 'late_logits', 'scratch')
MODES = ('base', 'expert', 'late_softmax_fixed', 'late_logits_fixed', 'unified_fixed',
         'unified_selected', 'late_softmax_selected', 'late_logits_selected', 'scratch_selected')


def metadata(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    return info, f"{info['dataset']}/{info['task']}"


def base_kernel(data, embedding, support, eta):
    kernel, cond = old_kernel(data, support, eta)
    return kernel+.25*(embedding@embedding[support].T)*cond


def candidate(family, kb, ke, pb, pe, support, labels, ratio, strength, separate):
    if family in ('unified', 'scratch'):
        result, _ = combined(kb, ke, pb, pe, support, labels, ratio, strength, family != 'scratch')
        if family == 'scratch':
            result[:, ~np.isin(np.arange(8), labels)] = -np.inf
        return result
    b, e = separate[strength]
    return late(b, e, ratio, family == 'late_softmax')


def source_fit(folder):
    from unified_head_expert import ExpertContext
    info, suffix = metadata(folder)
    dst = OUT/'source'/suffix
    if (dst/'SELECTION.json').exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    context = ExpertContext(folder)
    previous = load_npz(PAIR/'source'/suffix/'source_pairs_final.npz')
    records, checks = [], []
    for item in context.source_components():
        ei, ix = item['episode'], item['indices']
        ep = load_npz(folder/'episodes'/f'{ei:02d}.npz')
        z = load_npz(HEADS/'source'/suffix/f'episode_{ei:02d}.npz')
        np.testing.assert_array_equal(ix, z['indices'])
        data = {k: ep[k][ix] for k in ('phi', 'mean', 'conditions', 'prior')}
        support = np.arange(64)
        sy, y = ep['labels'][ix[:64]], ep['labels'][ix[64:]]
        pb, pe, ke = data['prior'], item['prior'], item['kernel']
        kb = base_kernel(data, z['learned'].astype(float), support, info['eta'])
        separate = {strength: (score(kb, pb, support, sy, strength)[0],
                               score(ke, pe, support, sy, strength)[0]) for strength in STRENGTHS}
        expected = previous['expert'][previous['episode'] == ei]
        np.testing.assert_array_equal(previous['rows'][previous['episode'] == ei], ep['rows'][ix[64:]])
        np.testing.assert_allclose(separate[8.][0][64:], z['expected'], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(separate[8.][1][64:], expected, atol=3e-5, rtol=1e-6)
        np.testing.assert_array_equal(separate[8.][1][64:].argmax(1), expected.argmax(1))
        checks.append(dict(episode=ei, source_expert_max_error=float(np.max(abs(separate[8.][1][64:]-expected))),
                            **fit_diagnostics(kb, ke, pb, pe, support, sy)))
        for ratio, strength in GRID:
            for family in FAMILIES:
                pred = candidate(family, kb, ke, pb, pe, support, sy, ratio, strength, separate)[64:].argmax(1)
                records.append(dict(episode=ei, family=family, ratio=ratio, strength=strength,
                                    oa=100*np.mean(pred == y), ba=ba(y, pred)))
        print(f'SOURCE {suffix} ep{ei}: original experts replayed; unified and separate fitted', flush=True)
    frame = pd.DataFrame(records)
    means = frame.groupby(['family', 'ratio', 'strength']).ba.mean()
    selected = {}
    for family in FAMILIES:
        best = max(GRID, key=lambda pair: means.loc[(family,)+pair])
        selected[family] = dict(ratio=best[0], strength=best[1], source_ba=float(means.loc[(family,)+best]))
    frame.to_csv(dst/'validation.csv', index=False)
    pd.DataFrame(checks).to_csv(dst/'fit_diagnostics.csv', index=False)
    write(dst/'SELECTION.json', dict(dataset=info['dataset'], task=info['task'], selected=selected,
        source_episodes=int(frame.episode.nunique()), criterion='Equal-source-episode BA; fixed grid order ties',
        expert_source_replay=True, no_target_query_labels=True,
        source_prior='Same original two linear source priors, combined as one concatenated feature-space initialization',
        limitation='Geometry/encoders share source; not independent representation-held validation'))
    print(f'SOURCE CHOSEN {suffix}: {selected}', flush=True)


def target_predict(folder):
    from unified_head_expert import ExpertContext
    info, suffix = metadata(folder)
    dst = OUT/'targets'/suffix
    if (dst/'COMPLETE.json').exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    selected = json.loads((OUT/'source'/suffix/'SELECTION.json').read_text())['selected']
    data = load_npz(folder/'target.npz')
    embed = load_npz(folder/'selective_embedding.npz')
    old = load_npz(REFERENCE/'targets'/suffix/'policies.npz')
    np.testing.assert_array_equal(data['rows'], embed['rows'])
    np.testing.assert_array_equal(data['ids'], old['ids'])
    np.testing.assert_array_equal(data['support'], old['support'])
    np.testing.assert_array_equal(data['eval_masks'], old['eval_masks'])
    context = ExpertContext(folder)
    predictions = np.empty((5, len(MODES), len(data['rows'])), np.int8)
    diagnostics = []
    started = time.perf_counter()
    for draw, support in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        if len(np.unique(support)) != 64 or ev[support].any():
            raise ValueError('Support/query overlap')
        pe, ke = context.target_components(data, support)
        pb = data['prior']
        kb = base_kernel(data, embed['embedding'].astype(float), support, info['eta'])
        needed = set([8., *(v['strength'] for v in selected.values())])
        separate = {strength: (score(kb, pb, support, sy, strength)[0],
                               score(ke, pe, support, sy, strength)[0]) for strength in needed}
        b, e = separate[8.]
        np.testing.assert_allclose(e, data['expert'][draw], atol=3e-5, rtol=1e-6)
        result = dict(base=b, expert=e, late_softmax_fixed=late(b, e),
                      late_logits_fixed=late(b, e, probability=False),
                      unified_fixed=combined(kb, ke, pb, pe, support, sy)[0])
        for family, params in selected.items():
            result[family+'_selected'] = candidate(family, kb, ke, pb, pe, support, sy,
                                                   params['ratio'], params['strength'], separate)
        for mode, reference in [('base', 'ours_base'), ('expert', 'expert_all'),
                                ('late_softmax_fixed', 'softmax_average_all')]:
            expected = old['predictions'][draw, list(old['modes']).index(reference)]
            np.testing.assert_array_equal(result[mode].argmax(1), expected)
        for mi, mode in enumerate(MODES):
            predictions[draw, mi] = result[mode].argmax(1)
        # Exact decomposition distinguishes a shared fit from a mere change in readout.
        diagnostics.append(dict(draw=draw, expert_max_error=float(np.max(abs(e-data['expert'][draw]))),
            **fit_diagnostics(kb, ke, pb, pe, support, sy),
            unified_vs_late_logits_changes=int(np.sum(result['unified_fixed'][ev].argmax(1) !=
                                                       result['late_logits_fixed'][ev].argmax(1)))))
        print(f'TARGET {suffix} draw{draw}: frozen; old base/expert/fusion exact', flush=True)
    np.savez_compressed(dst/'policies.npz', modes=np.array(MODES), predictions=predictions,
        rows=data['rows'], ids=data['ids'], support=data['support'], eval_masks=data['eval_masks'])
    pd.DataFrame(diagnostics).to_csv(dst/'fit_diagnostics.csv', index=False)
    write(dst/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], draws=5,
        target_query_labels_read=False, old_predictions_exact=True, seconds=time.perf_counter()-started))


def report():
    tasks = folders()
    if len(tasks) != 25 or not all((OUT/'targets'/metadata(f)[1]/'COMPLETE.json').exists() for f in tasks):
        raise RuntimeError('Complete every target prediction before query truth readout')
    records, classes, fits, source_records, choices = [], [], [], [], []
    for folder in tasks:
        info, suffix = metadata(folder)
        p = load_npz(OUT/'targets'/suffix/'policies.npz')
        data = load_npz(folder/'target.npz')
        fits.append(pd.read_csv(OUT/'targets'/suffix/'fit_diagnostics.csv').assign(dataset=info['dataset'], task=info['task']))
        source_records.append(pd.read_csv(OUT/'source'/suffix/'validation.csv').assign(dataset=info['dataset'], task=info['task']))
        selection = json.loads((OUT/'source'/suffix/'SELECTION.json').read_text())['selected']
        choices += [dict(dataset=info['dataset'], task=info['task'], family=family, **params)
                    for family, params in selection.items()]
        manifest, idcol = ((SOURCE/'manifest.csv', 'instance_id') if info['dataset'] == 'self24ports'
            else (ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', 'sample_id'))
        frame = pd.read_csv(manifest, usecols=[idcol, 'class_id'], keep_default_na=False).iloc[p['rows']]
        np.testing.assert_array_equal(frame[idcol].to_numpy(str), p['ids'])
        truth = frame.class_id.to_numpy()
        np.testing.assert_array_equal(truth[p['support']], data['support_labels'])
        for draw, ev in enumerate(p['eval_masks']):
            y = truth[ev]
            ref = p['predictions'][draw, MODES.index('late_softmax_fixed'), ev]
            for mi, mode in enumerate(MODES):
                pred = p['predictions'][draw, mi, ev]
                records.append(dict(dataset=info['dataset'], task=info['task'], draw=draw, mode=mode,
                    oa=100*np.mean(pred == y), ba=ba(y, pred),
                    delta_oa=100*np.mean((pred == y).astype(float)-(ref == y)),
                    delta_ba=ba(y, pred)-ba(y, ref), rescues=int(np.sum((pred == y) & (ref != y))),
                    harms=int(np.sum((pred != y) & (ref == y))), query_n=len(y)))
                present = np.unique(y)
                for c in present:
                    mask = y == c
                    delta = 100*np.mean((pred[mask] == c).astype(float)-(ref[mask] == c))
                    classes.append(dict(dataset=info['dataset'], task=info['task'], draw=draw, mode=mode,
                        class_id=int(c), n=int(mask.sum()), support_n=int(np.sum(data['support_labels'][draw] == c)),
                        recall=100*np.mean(pred[mask] == c), delta_recall=delta, ba_contribution=delta/len(present)))
    m = pd.DataFrame(records)
    m.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_metrics.csv', index=False)
    fields = ['oa', 'ba', 'delta_oa', 'delta_ba']
    task = m.groupby(['dataset', 'task', 'mode'])[fields].mean().reset_index()
    task.to_csv(OUT/'task_means.csv', index=False)
    summary = task.groupby(['dataset', 'mode'])[fields].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    m.groupby(['dataset', 'draw', 'mode'])[fields].mean().to_csv(OUT/'draw_metrics.csv')
    pd.concat(fits).to_csv(OUT/'fit_diagnostics.csv', index=False)
    pd.concat(source_records).to_csv(OUT/'source_validation.csv', index=False)
    pd.DataFrame(choices).to_csv(OUT/'source_choices.csv', index=False)
    previous = pd.read_csv(REFERENCE/'summary.csv')
    for mode, reference in [('base', 'ours_base'), ('expert', 'expert_all'), ('late_softmax_fixed', 'softmax_average_all')]:
        actual = summary[summary['mode'] == mode].sort_values('dataset')
        expected = previous[previous['mode'] == reference].sort_values('dataset')
        np.testing.assert_allclose(actual[['oa', 'ba']], expected[['oa', 'ba']], atol=1e-10, rtol=0)
    write(OUT/'COMPLETE.json', dict(tasks=25, draws=125, modes=MODES, rows=len(m),
        all_predictions_frozen_before_readout=True, original_base_expert_fusion_exact=True,
        source_only_selection=True, independent_confirmation=False))
    print(summary[['dataset', 'mode', 'oa', 'ba']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('source', 'target', 'report', 'all'), default='all')
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    if not (OUT/'PROTOCOL.json').exists():
        write(OUT/'PROTOCOL.json', dict(date='2026-10-10', tasks=25, draws=5, target_labels=64,
            modes=MODES, source_grid=GRID, no_target_query_selection=True,
            combined_kernel='(1-rho)*K_base+rho*K_expert; exact concatenated feature-map inner product',
            combined_prior='(1-rho)*source_base_prior+rho*source_expert_prior',
            unified_head='One shared64x8 dual coefficient solve, using both full information kernels',
            prior_boundary='Preserves both old source priors as a single concatenated head initialization; no new fully joint source backbone training',
            scratch='Same feature union, zero source prior; diagnostic only, absent support classes masked',
            grid_selection='Equal source-episode BA, same nine candidates for unified and late families; fixed order ties',
            no_neural_training=True, independent_confirmation=False,
            cost='Both feature extractors retained; fewer adaptation solves does not establish end-to-end acceleration'))
    tasks = [f for f in folders() if not args.tasks or metadata(f)[0]['task'] in args.tasks]
    if args.stage in ('source', 'all'):
        for folder in tasks:
            source_fit(folder)
    if args.stage in ('target', 'all'):
        if not all((OUT/'source'/metadata(f)[1]/'SELECTION.json').exists() for f in folders()):
            raise RuntimeError('Freeze all source choices before target prediction')
        for folder in tasks:
            target_predict(folder)
    if args.stage in ('report', 'all'):
        report()


if __name__ == '__main__':
    main()
