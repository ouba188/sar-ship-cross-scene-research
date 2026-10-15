"""Posthoc fixed compact/full-base prior and metric crossover, no tuning."""

import pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from full_ssl_run import OUT, CACHE, MODES, TARGETS, archive, write, source_frame
from full_ssl_geometry import _parts
from general_domain_readout import design_block, fit_head, metrics
from unified_head_core import coefficients

CELLS = ((0, 0), (1, 0), (0, 1), (1, 1))
DEST = OUT/'compact_full_bridge'


def parts(raw, rows, state, head):
    compact = design_block(raw, rows)
    full = _parts(raw[rows], state, state['raw_base'])
    return (compact, full['phi']), (compact@head, full['prior'])


@threadpool_limits.wrap(limits=2, user_api='blas')
def predict(domain):
    directory = CACHE/domain
    with (directory/'geometry.pkl').open('rb') as f:
        state = pickle.load(f)
    source = np.load(directory/'source_raw.npy', mmap_mode='r')
    labels = source_frame().class_id.to_numpy(int)
    head = fit_head(source, labels, state['source_rows'])
    np.save(DEST/f'{domain}_compact_head.npy', head)
    for ds in TARGETS:
        design = archive(OUT/f'{ds}_design.npz')
        rows = np.unique(design['support'])
        positions = np.searchsorted(rows, design['support'])
        raw = np.load(directory/f'{ds}_raw.npy', mmap_mode='r')
        refs, rp = parts(raw, rows, state, head)
        kernels = [x@x.T for x in refs]
        alpha = {}
        for ci, (p, k) in enumerate(CELLS):
            alpha[ci] = [coefficients(kernels[k][np.ix_(ix, ix)], rp[p][ix], design['support_labels'][draw], 8.)
                         for draw, ix in enumerate(positions)]
        saved = np.lib.format.open_memmap(DEST/f'{domain}_{ds}_predictions.npy', mode='w+',
                                          dtype=np.int8, shape=(5, 4, len(raw)))
        old = np.load(directory/ds/'predictions.npy', mmap_mode='r')
        for lo in range(0, len(raw), 512):
            hi = min(lo+512, len(raw))
            query, qp = parts(raw, np.arange(lo, hi), state, head)
            kk = [q@r.T for q, r in zip(query, refs)]
            for ci, (p, k) in enumerate(CELLS):
                for draw, ix in enumerate(positions):
                    pred = (qp[p]+kk[k][:, ix]@alpha[ci][draw]).argmax(1)
                    if ci == 3:
                        np.testing.assert_array_equal(pred, old[draw, MODES.index('base_raw'), lo:hi])
                    saved[draw, ci, lo:hi] = pred
            if lo % 32768 == 0:
                print('BRIDGE', domain, ds, hi, '/', len(raw), flush=True)
        saved.flush()
        write(DEST/f'{domain}_{ds}_PREDICTIONS.json', dict(full_base_diagonal_exact=True,
              new_compact_fit_uses_consistent_source_and_target_features=True, query_truth_read=False))


def main():
    DEST.mkdir(exist_ok=True)
    write(DEST/'PROTOCOL.json', dict(trigger='Posthoc diagnosis of compact/full sign reversal; primary results remain unchanged',
        endpoints=['original 2048D row-L2 sqrt32 compact readout', 'original-view source PCA128 standardized full base'],
        cells=['P0K0', 'P1K0', 'P0K1', 'P1K1'], source_ridge=1, support_ridge=8,
        training='None; fresh compact source head, fixed encoders and full geometry',
        selection='No grid, no target-selected hyperparameters or replacement primary model',
        interpretation='Separates prior versus metric path, not PCA truncation versus centering versus whitening individually'))
    for domain in ('G', 'D'):
        predict(domain)
    records = []
    for ds, paths in TARGETS.items():
        design = archive(OUT/f'{ds}_design.npz')
        truth = pd.read_csv(paths['manifest'], usecols=['class_id']).class_id.to_numpy()[design['rows']]
        for domain in ('G', 'D'):
            pred = np.load(DEST/f'{domain}_{ds}_predictions.npy', mmap_mode='r')
            for draw, mask in enumerate(design['eval_masks']):
                for ci, (p, k) in enumerate(CELLS):
                    records.append(dict(dataset=ds, domain=domain, draw=draw, cell=f'P{p}K{k}',
                                        **metrics(truth[mask], pred[draw, ci, mask])))
    frame = pd.DataFrame(records)
    frame.to_csv(DEST/'metrics.csv', index=False)
    summary = frame.groupby(['dataset', 'domain', 'cell'])[['oa', 'ba']].mean().reset_index()
    summary.to_csv(DEST/'summary.csv', index=False)
    write(DEST/'COMPLETE.json', dict(full_base_diagonals_exact=True, cells=4, domains=2, datasets=2,
          neural_training=False, target_tuning=False, posthoc=True))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
