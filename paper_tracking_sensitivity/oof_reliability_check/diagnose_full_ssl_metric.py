"""Bounded metric intervention with the same full-base prior in every arm."""

import pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from full_ssl_run import OUT, CACHE, TARGETS, archive, write
from full_ssl_geometry import _parts, _project
from unified_head_core import coefficients
from general_domain_readout import metrics

DEST = OUT/'metric_diagnosis'
MODES = ('scaled_standardized', 'unit_standardized', 'unit_projected')


def parts(raw, rows, state):
    value = raw[rows]
    full = _parts(value, state, state['raw_base'])
    standardized = full['phi'][:, :-1]
    projected = _project(value, state['pca'])-state['center']
    unit_std = standardized/np.maximum(np.linalg.norm(standardized, axis=1, keepdims=True), 1e-12)
    unit_projected = projected/np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)
    return [np.c_[x*np.sqrt(32.), np.ones(len(rows))] for x in (standardized, unit_std, unit_projected)], full['prior']


@threadpool_limits.wrap(limits=2, user_api='blas')
def predict(domain, ds):
    directory = CACHE/domain
    with (directory/'geometry.pkl').open('rb') as f:
        state = pickle.load(f)
    raw = np.load(directory/f'{ds}_raw.npy', mmap_mode='r')
    design = archive(OUT/f'{ds}_design.npz')
    rows = np.unique(design['support'])
    positions = np.searchsorted(rows, design['support'])
    refs, prior = parts(raw, rows, state)
    alpha = []
    for ref in refs:
        kernel = ref@ref.T
        alpha.append([coefficients(kernel[np.ix_(ix, ix)], prior[ix], design['support_labels'][draw], 8.)
                      for draw, ix in enumerate(positions)])
    saved = np.lib.format.open_memmap(DEST/f'{domain}_{ds}_predictions.npy', mode='w+', dtype=np.int8,
                                      shape=(5, len(MODES), len(raw)))
    for lo in range(0, len(raw), 512):
        hi = min(lo+512, len(raw))
        values, prior = parts(raw, np.arange(lo, hi), state)
        for mi, (value, ref) in enumerate(zip(values, refs)):
            kernel = value@ref.T
            for draw, ix in enumerate(positions):
                scores = prior+kernel[:, ix]@alpha[mi][draw]
                if not np.isfinite(scores).all():
                    raise ValueError('Nonfinite intervention score')
                saved[draw, mi, lo:hi] = scores.argmax(1)
        if lo % 32768 == 0:
            print('METRIC', domain, ds, hi, '/', len(raw), flush=True)
    saved.flush()


def main():
    DEST.mkdir(exist_ok=True)
    write(DEST/'PROTOCOL.json', dict(posthoc=True, trigger='Compact-to-full bridge loses mainly through the metric on D50',
        fixed='Original-view full PCA128 source prior, same64, ridge8, same frozen features',
        interventions={
          MODES[0]: 'Multiply nonconstant full standardized kernel by32, retain bias1',
          MODES[1]: 'Row-L2 standardized128 coordinates, energy32, bias1',
          MODES[2]: 'Row-L2 centered unwhitened PCA128 coordinates, energy32, bias1'},
        scale='32 is the pre-existing compact protocol, not a searched target optimum',
        limits='Diagnostic base heads only, no replacement of prespecified full fusion; no neural training or hyperparameter grid'))
    for domain in ('G', 'D'):
        for ds in TARGETS:
            predict(domain, ds)
    records = []
    for ds, paths in TARGETS.items():
        design = archive(OUT/f'{ds}_design.npz')
        truth = pd.read_csv(paths['manifest'], usecols=['class_id']).class_id.to_numpy()[design['rows']]
        for domain in ('G', 'D'):
            pred = np.load(DEST/f'{domain}_{ds}_predictions.npy', mmap_mode='r')
            for draw, mask in enumerate(design['eval_masks']):
                for mi, mode in enumerate(MODES):
                    records.append(dict(dataset=ds, domain=domain, draw=draw, mode=mode,
                                        **metrics(truth[mask], pred[draw, mi, mask])))
    frame = pd.DataFrame(records)
    frame.to_csv(DEST/'metrics.csv', index=False)
    summary = frame.groupby(['dataset', 'domain', 'mode'])[['oa', 'ba']].mean().reset_index()
    summary.to_csv(DEST/'summary.csv', index=False)
    write(DEST/'COMPLETE.json', dict(neural_training=False, posthoc=True, searched_parameters=False, primary_outputs_unchanged=True))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
