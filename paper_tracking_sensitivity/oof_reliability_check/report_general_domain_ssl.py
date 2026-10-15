"""Fixed comparisons and prior/kernel interventions for the G/D SSL pilot."""

import json
import shutil

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from general_domain_readout import (CACHE, OUT, PRIMARY_MODES, SOURCE_MANIFEST,
                                   design_block, fit_head, load_episodes, metrics)
from unified_head_core import coefficients


CONTRASTS = {
    'single_SAR_adaptation': ('A_G50', 'B_D50'),
    'R18_SAR_adaptation_in_dual': ('C_G50_G18', 'D_G50_D18'),
    'preserve_general_R50': ('E_D50_D18', 'D_G50_D18'),
    'dual_vs_domain_single': ('B_D50', 'D_G50_D18'),
    'second_general_backbone': ('A_G50', 'C_G50_G18'),
    'dual_vs_general_single': ('A_G50', 'D_G50_D18'),
    'R18_alone_SAR_adaptation': ('diag_G18', 'diag_D18'),
    'dual_vs_domain_R18': ('diag_D18', 'D_G50_D18'),
}
CROSSOVERS = {k: CONTRASTS[k] for k in
              ('single_SAR_adaptation', 'R18_SAR_adaptation_in_dual', 'preserve_general_R50')}
CELLS = ((0, 0), (1, 0), (0, 1), (1, 1))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def error_flow(y, before, after):
    """Class-balanced rescue minus harm is exactly delta BA, in points."""
    classes, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
    w = 100. / (len(classes) * counts[inv])
    rescued = (before != y) & (after == y)
    harmed = (before == y) & (after != y)
    result = dict(rescued=int(rescued.sum()), harmed=int(harmed.sum()),
                  rescue_ba=float(w[rescued].sum()), harm_ba=float(w[harmed].sum()))
    result['delta_ba'] = result['rescue_ba'] - result['harm_ba']
    result['delta_oa'] = 100. * (result['rescued'] - result['harmed']) / len(y)
    return result


def fixed_contrasts(root):
    table = pd.read_csv(root / 'metrics.csv')
    records = []
    for ds, group in table.groupby('dataset', sort=False):
        for name, (left, right) in CONTRASTS.items():
            a = group[group['mode'] == left].set_index('draw')
            b = group[group['mode'] == right].set_index('draw')
            for draw in a.index:
                records.append(dict(dataset=ds, contrast=name, before=left, after=right,
                                    draw=int(draw), delta_oa=b.loc[draw, 'oa']-a.loc[draw, 'oa'],
                                    delta_ba=b.loc[draw, 'ba']-a.loc[draw, 'ba']))
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / 'contrasts.csv', index=False)
    summary = frame.groupby(['dataset', 'contrast'], sort=False).agg(
        delta_oa=('delta_oa', 'mean'), delta_ba=('delta_ba', 'mean'),
        support_draw_sd_ba=('delta_ba', 'std'),
        positive_ba_draws=('delta_ba', lambda x: int((x > 0).sum())))
    summary.to_csv(OUT / 'contrast_summary.csv')


def target_flow(receipt, root):
    ds, modes = receipt['dataset'], receipt['modes']
    folder = root / ds
    with np.load(folder / 'design.npz') as design:
        rows, masks = design['rows'], design['eval_masks']
    y = pd.read_csv(receipt['target_manifest'], usecols=['class_id']).class_id.to_numpy(int)[rows]
    pred = np.load(folder / 'predictions.npy', mmap_mode='r')
    flow, classes, adaptation = [], [], []
    for name, (left, right) in CONTRASTS.items():
        for draw, mask in enumerate(masks):
            a, b = pred[draw, modes.index(left), mask], pred[draw, modes.index(right), mask]
            record = error_flow(y[mask], a, b)
            delta = metrics(y[mask], b)['ba'] - metrics(y[mask], a)['ba']
            np.testing.assert_allclose(record['delta_ba'], delta, atol=1e-10)
            flow.append(dict(dataset=ds, contrast=name, draw=draw, **record))
            for c in np.unique(y[mask]):
                cm = y[mask] == c
                classes.append(dict(dataset=ds, contrast=name, draw=draw, class_id=int(c),
                                    n=int(cm.sum()), **error_flow(y[mask][cm], a[cm], b[cm])))
    # Check whether preserving G50 actually repairs examples lost by adapting G50.
    for draw, mask in enumerate(masks):
        truth = y[mask]
        a, b, d = (pred[draw, modes.index(m), mask] for m in ('A_G50', 'B_D50', 'D_G50_D18'))
        for name, group in {
            'single_adaptation_harmed': (a == truth) & (b != truth),
            'single_adaptation_rescued': (a != truth) & (b == truth),
            'both_R50_singles_wrong': (a != truth) & (b != truth),
        }.items():
            adaptation.append(dict(dataset=ds, draw=draw, group=name, n=int(group.sum()),
                                   dual_correct=int(((d == truth) & group).sum())))
    return flow, classes, adaptation


def crossover(receipt, root):
    """Change source prior and support geometry independently; never select a cell."""
    ds, modes, prior_modes = receipt['dataset'], receipt['modes'], receipt['prior_modes']
    folder = root / ds
    with np.load(folder / 'design.npz') as saved:
        data = {k: saved[k] for k in ('rows', 'support', 'support_labels', 'eval_masks')}
    with np.load(folder / 'support_state.npz') as saved:
        states = {k: saved[k] for k in saved.files}
    priors = np.load(folder / 'priors.npy', mmap_mode='r')
    original = np.load(folder / 'predictions.npy', mmap_mode='r')
    original_scores = np.load(folder / 'scores.npy', mmap_mode='r')
    truth = pd.read_csv(receipt['target_manifest'], usecols=['class_id']).class_id.to_numpy(int)[data['rows']]
    features = {a: np.load(path, mmap_mode='r') for a, path in receipt['features'].items()}
    results, checks = [], []
    for name, pair in CROSSOVERS.items():
        destination = root / f'{ds}_{name}_cross_predictions.npy'
        predictions = np.lib.format.open_memmap(destination, mode='w+', dtype=np.int8,
                                                shape=(5, 4, len(truth)))
        sp = [states[f'{m}_phi'] for m in pair]
        ps = [states[f'{m}_prior'] for m in pair]
        alphas = {(pi, ki): np.stack([
            coefficients(sp[ki][d] @ sp[ki][d].T, ps[pi][d], data['support_labels'][d], 8.)
            for d in range(5)]) for pi, ki in CELLS}
        max_error = 0.
        for lo in range(0, len(truth), 512):
            hi = min(lo + 512, len(truth))
            phi = [design_block(tuple(features[a] for a in PRIMARY_MODES[m]), data['rows'][lo:hi])
                   for m in pair]
            kernels = [[phi[ki] @ sp[ki][d].T for d in range(5)] for ki in range(2)]
            for ci, (pi, ki) in enumerate(CELLS):
                prior = priors[prior_modes.index(pair[pi]), lo:hi]
                for draw in range(5):
                    value = prior + kernels[ki][draw] @ alphas[pi, ki][draw]
                    predictions[draw, ci, lo:hi] = value.argmax(axis=1)
                    if pi == ki:
                        oi = modes.index(pair[pi])
                        reference = original_scores[draw, oi, lo:hi]
                        max_error = max(max_error, float(np.abs(value-reference).max()))
                        np.testing.assert_allclose(value, reference, atol=1e-9, rtol=1e-10)
                        np.testing.assert_array_equal(predictions[draw, ci, lo:hi], original[draw, oi, lo:hi])
        predictions.flush()
        for draw, mask in enumerate(data['eval_masks']):
            for ci, (pi, ki) in enumerate(CELLS):
                results.append(dict(dataset=ds, contrast=name, draw=draw,
                                    cell=f'P{pi}K{ki}', prior=pair[pi], kernel=pair[ki],
                                    **metrics(truth[mask], predictions[draw, ci, mask])))
        checks.append(dict(dataset=ds, contrast=name, diagonal_predictions_exact=True,
                           max_score_error=max_error, query_draw_exposures=int(data['eval_masks'].sum()),
                           predictions=str(destination)))
        print('CROSSOVER', ds, name, 'diagonals exact', flush=True)
        del predictions
    return results, checks


def source_combinations():
    labels = pd.read_csv(SOURCE_MANIFEST, usecols=['class_id']).class_id.to_numpy(int)
    train = np.load(OUT / 'source_split.npz')['inner_train']
    episodes = load_episodes(OUT / 'validation_episodes.npz')
    records = []
    for mode, archs in PRIMARY_MODES.items():
        if mode in ('A_G50', 'B_D50'):
            if mode == 'A_G50':
                saved = json.loads((OUT / 'r50' / 'GENERAL.json').read_text())['episodes']
            else:
                selected = json.loads((OUT / 'r50' / 'TRAINING_COMPLETE.json').read_text())['selected_epoch']
                curve = json.loads((OUT / 'r50' / 'CURVE.json').read_text())
                saved = next(r['validation_episodes'] for r in curve if r['epoch'] == selected)
            records.extend(dict(mode=mode, **r) for r in saved)
            print('REUSE SOURCE', mode, flush=True)
            continue
        fs = tuple(np.load(CACHE / 'features' / f'source_{a}.npy', mmap_mode='r') for a in archs)
        coef = fit_head(fs, labels, train)
        for i, episode in enumerate(episodes):
            sp, qp = (design_block(fs, episode[k]) for k in ('support', 'query'))
            alpha = coefficients(sp @ sp.T, sp @ coef, labels[episode['support']], 8.)
            pred = (qp @ coef + (qp @ sp.T) @ alpha).argmax(axis=1)
            records.append(dict(mode=mode, episode=i, port=episode['port'],
                                **metrics(labels[episode['query']], pred)))
        print('SOURCE', mode, np.mean([r['ba'] for r in records if r['mode'] == mode]), flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / 'source_validation.csv', index=False)
    frame.groupby('mode', sort=False)[['oa', 'ba']].mean().to_csv(OUT / 'source_summary.csv')


@threadpool_limits.wrap(limits=2, user_api='blas')
def source_r18_selection_check():
    """Post-hoc source-only probe; do not select or evaluate another target model."""
    labels = pd.read_csv(SOURCE_MANIFEST, usecols=['class_id']).class_id.to_numpy(int)
    train = np.load(OUT / 'source_split.npz')['inner_train']
    episodes = load_episodes(OUT / 'validation_episodes.npz')
    curve = json.loads((OUT / 'r18' / 'CURVE.json').read_text())
    last = next(r for r in curve if r['epoch'] == 8)
    fs = tuple(np.load(CACHE / 'features' / filename, mmap_mode='r') for filename in
               ('source_G50.npy', 'source_current_r18.npy'))
    coef = fit_head(fs, labels, train)
    details = []
    for i, episode in enumerate(episodes):
        sp, qp = (design_block(fs, episode[k]) for k in ('support', 'query'))
        alpha = coefficients(sp @ sp.T, sp @ coef, labels[episode['support']], 8.)
        pred = (qp @ coef + (qp @ sp.T) @ alpha).argmax(axis=1)
        details.append(dict(episode=i, port=episode['port'], **metrics(labels[episode['query']], pred)))
    source = pd.read_csv(OUT / 'source_summary.csv').set_index('mode')
    selected = json.loads((OUT / 'r18' / 'TRAINING_COMPLETE.json').read_text())
    baseline = json.loads((OUT / 'r18' / 'GENERAL.json').read_text())
    records = [dict(variant='general', epoch=0, single_ba=baseline['ba'],
                    combined_ba=float(source.loc['C_G50_G18', 'ba'])),
               dict(variant='selected', epoch=selected['selected_epoch'], single_ba=selected['selected_source_ba'],
                    combined_ba=float(source.loc['D_G50_D18', 'ba'])),
               dict(variant='last_source_only_diagnostic', epoch=8, single_ba=last['source_ba'],
                    combined_ba=float(np.mean([r['ba'] for r in details])))]
    pd.DataFrame(records).to_csv(OUT / 'selection_diagnosis.csv', index=False)
    dump(OUT / 'SELECTION_DIAGNOSIS.json', dict(records=records, last_combined_episodes=details,
         scope='Only source ports. Three retained states, not all epochs. Primary target models unchanged.',
         post_hoc_diagnostic=True, additional_target_predictions=0))
    print(pd.DataFrame(records).to_string(index=False), flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    summary = pd.read_csv(OUT / 'summary.csv')
    colors = ['#697984', '#168f8b', '#d6a247', '#c45068', '#5582bd']
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), layout='constrained')
    names = ['G50', 'D50', 'G50+G18', 'G50+D18', 'D50+D18']
    for ax, ds in zip(axes[:2], ('f0921', 'oss2')):
        data = summary[(summary.dataset == ds) & summary['primary']].set_index('mode')
        for i, (mode, color) in enumerate(zip(PRIMARY_MODES, colors)):
            ax.bar(i, data.loc[mode, 'ba'], color=color, width=.7)
            ax.text(i, data.loc[mode, 'ba']+.45, f"{data.loc[mode, 'ba']:.2f}", ha='center', fontsize=9)
        ax.set(xticks=np.arange(5), xticklabels=names, ylabel='Balanced accuracy (%)',
               title='F_0921' if ds == 'f0921' else 'OpenSARShip 2.0')
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylim(0, max(data.ba)*1.18)
    for arch, color in [('r18', colors[1]), ('r50', colors[4])]:
        baseline = json.loads((OUT / arch / 'GENERAL.json').read_text())['ba']
        curve = json.loads((OUT / arch / 'CURVE.json').read_text())
        selected = [r for r in curve if 'source_ba' in r]
        axes[2].plot([0]+[r['epoch'] for r in selected], [baseline]+[r['source_ba'] for r in selected],
                     marker='o', color=color, label=arch.upper())
    axes[2].set(xlabel='SSL epoch', ylabel='Source selection BA (%)', title='Source-held-port validation')
    axes[2].legend(frameon=False)
    fig.suptitle('Matched G/D representation pilot | one neural seed, five support draws', fontsize=12)
    fig.savefig(OUT / 'pilot_results.png', dpi=200)
    fig.savefig(OUT / 'pilot_results.pdf')
    plt.close(fig)
    flow = pd.read_csv(OUT / 'class_flow.csv')
    mapping = pd.read_csv(SOURCE_MANIFEST, usecols=['class_id', 'class_name']).drop_duplicates().set_index('class_id')
    columns = [(ds, name) for ds in ('f0921', 'oss2') for name in CROSSOVERS]
    grouped = flow.groupby(['dataset', 'contrast', 'class_id']).delta_ba.mean()
    ids = sorted(flow.class_id.unique())
    values = np.array([[grouped.loc[ds, name, c] for ds, name in columns] for c in ids])
    limit = max(1., float(np.abs(values).max()))
    fig, ax = plt.subplots(figsize=(11.2, 5.2), layout='constrained')
    im = ax.imshow(values, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    labels = [f'{ds}\n{short}' for ds in ('F_0921', 'OSS2')
              for short in ('Single SAR: B-A', 'R18 SAR: D-C', 'Keep general: D-E')]
    ax.set(xticks=np.arange(len(columns)), xticklabels=labels,
           yticks=np.arange(len(ids)), yticklabels=[mapping.loc[c, 'class_name'].replace('_', ' ') for c in ids],
           title='Class recall changes | fixed comparisons, mean of five support draws')
    for row in range(len(ids)):
        for col in range(len(columns)):
            ax.text(col, row, f'{values[row,col]:+.1f}', ha='center', va='center',
                    color='white' if abs(values[row,col]) > .55*limit else '#202020')
    fig.colorbar(im, ax=ax, label='Recall change (percentage points)', shrink=.85)
    fig.savefig(OUT / 'class_effects.png', dpi=200)
    fig.savefig(OUT / 'class_effects.pdf')
    plt.close(fig)


@threadpool_limits.wrap(limits=2, user_api='blas')
def main():
    root = CACHE / 'readout'
    if not (root / 'COMPLETE.json').exists():
        raise RuntimeError('Complete the fixed primary readout before diagnosis')
    for name in ('metrics.csv', 'per_class.csv', 'summary.csv'):
        shutil.copyfile(root / name, OUT / name)
    fixed_contrasts(root)
    frozen = json.loads((root / 'PREDICTIONS_FROZEN.json').read_text())
    flow, classes, adaptation, cross, checks = [], [], [], [], []
    for receipt in frozen['targets']:
        f, c, a = target_flow(receipt, root)
        flow.extend(f)
        classes.extend(c)
        adaptation.extend(a)
        r, check = crossover(receipt, root)
        cross.extend(r)
        checks.extend(check)
    for name, records in [('error_flow', flow), ('class_flow', classes),
                          ('adaptation_cases', adaptation), ('crossover', cross)]:
        pd.DataFrame(records).to_csv(OUT / f'{name}.csv', index=False)
    pd.DataFrame(cross).groupby(['dataset', 'contrast', 'cell'], sort=False)[['oa', 'ba']].mean().to_csv(
        OUT / 'crossover_summary.csv')
    source_combinations()
    plot()
    dump(OUT / 'REPORT_COMPLETE.json', dict(readout=str(root), crossover_checks=checks,
         source_validation_is_selection_data=True, intervention_cells_are_not_selected_models=True,
         neural_seeds=1, support_draws=5, historical_full_model_unchanged=True))
    print('REPORT COMPLETE', OUT, flush=True)


if __name__ == '__main__':
    main()
