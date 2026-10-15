"""Fixed prior/kernel crossover of the completed full-readout comparisons."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from full_ssl_run import OUT, CACHE, TARGETS, MODES, archive, write
from full_ssl_readout import target_context, take, kernels
from unified_head_core import coefficients, late
from general_domain_readout import metrics

PAIRS = {
    'ssl_single': (('G', 'single'), ('D', 'single')),
    'ssl_dual': (('G', 'dual'), ('D', 'dual')),
    'second_G': (('G', 'single'), ('G', 'dual')),
    'second_D': (('D', 'single'), ('D', 'dual')),
}
CELLS = ((0, 0), (1, 0), (0, 1), (1, 1))


def prior(parts, architecture):
    return (parts['prior'], parts[f'{architecture}_prior'])


def kernel(parts, architecture):
    return (parts['full'], parts[architecture])


@threadpool_limits.wrap(limits=2, user_api='blas')
def predict(ds):
    directory = CACHE / 'path_diagnosis' / ds
    directory.mkdir(parents=True, exist_ok=True)
    design = archive(OUT/f'{ds}_design.npz')
    rows = np.unique(design['support'])
    positions = np.searchsorted(rows, design['support'])
    contexts = {d: target_context(d, ds) for d in ('G', 'D')}
    refs = {d: take(c, rows) for d, c in contexts.items()}
    kk = {d: kernels(r, r) for d, r in refs.items()}
    alpha = {}
    for name, arms in PAIRS.items():
        for ci, (p, r) in enumerate(CELLS):
            pdomain, parch = arms[p]
            rdomain, rarch = arms[r]
            ps, ks = prior(refs[pdomain], parch), kernel(kk[rdomain], rarch)
            alpha[name, ci] = [[coefficients(k[np.ix_(ix, ix)], pr[ix], design['support_labels'][draw], 8.)
                                for pr, k in zip(ps, ks)] for draw, ix in enumerate(positions)]
    n = len(design['rows'])
    saved = {name: np.lib.format.open_memmap(directory/f'{name}.npy', mode='w+', dtype=np.int8,
                                            shape=(5, 4, n)) for name in PAIRS}
    original = {d: np.load(CACHE/d/ds/'predictions.npy', mmap_mode='r') for d in ('G', 'D')}
    for lo in range(0, n, 512):
        hi = min(n, lo+512)
        ix = np.arange(lo, hi)
        parts = {d: take(c, ix) for d, c in contexts.items()}
        query_k = {d: kernels(parts[d], refs[d]) for d in contexts}
        for name, arms in PAIRS.items():
            for ci, (p, r) in enumerate(CELLS):
                pdomain, parch = arms[p]
                rdomain, rarch = arms[r]
                ps, ks = prior(parts[pdomain], parch), kernel(query_k[rdomain], rarch)
                for draw, support in enumerate(positions):
                    base, expert = [pr+k[:, support]@a for pr, k, a in zip(ps, ks, alpha[name, ci][draw])]
                    pred = late(base, expert).argmax(1)
                    if p == r:
                        np.testing.assert_array_equal(pred, original[pdomain][draw, MODES.index(f'late_{parch}'), lo:hi])
                    saved[name][draw, ci, lo:hi] = pred
        if lo % 32768 == 0:
            print('PATHS', ds, hi, '/', n, flush=True)
    for value in saved.values():
        value.flush()
    write(directory/'COMPLETE.json', dict(all_diagonals_exact=True, pairs=list(PAIRS),
          cells=['P0K0', 'P1K0', 'P0K1', 'P1K1'], query_truth_used_for_prediction=False))


def main():
    if not (OUT/'READOUT_COMPLETE.json').exists():
        raise RuntimeError('Complete fixed full readout first')
    for ds in TARGETS:
        predict(ds)
    records = []
    for ds, paths in TARGETS.items():
        design = archive(OUT/f'{ds}_design.npz')
        truth = pd.read_csv(paths['manifest'], usecols=['class_id']).class_id.to_numpy()[design['rows']]
        for name in PAIRS:
            pred = np.load(CACHE/'path_diagnosis'/ds/f'{name}.npy', mmap_mode='r')
            for draw, mask in enumerate(design['eval_masks']):
                for ci, (p, k) in enumerate(CELLS):
                    records.append(dict(dataset=ds, contrast=name, draw=draw, cell=f'P{p}K{k}',
                                        **metrics(truth[mask], pred[draw, ci, mask])))
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'path_metrics.csv', index=False)
    summary = frame.groupby(['dataset', 'contrast', 'cell'])[['oa', 'ba']].mean().reset_index()
    summary.to_csv(OUT/'path_summary.csv', index=False)
    write(OUT/'PATH_DIAGNOSIS_COMPLETE.json', dict(pairs_per_dataset=4, all_diagonals_exact=True,
          interpretation='Functional source-prior vs support-kernel intervention; neither a physical causal proof nor a selected new model'))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
