"""Frozen complete-readout single/dual comparisons on the two external cohorts."""

import json
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from full_ssl_run import OUT, CACHE, OLD_CACHE, TARGETS, MODES, ETA, BETA, archive, write
from full_ssl_geometry import transform
from unified_head_core import coefficients, late
from general_domain_readout import metrics


DIRECT = {'base_raw': ('raw', 'raw_prior'), 'base_c4': ('c4', 'prior'),
          'base_condition': ('condition', 'prior'), 'base_full': ('full', 'prior'),
          'expert_single': ('single', 'single_prior'), 'expert_dual': ('dual', 'dual_prior')}


def target_context(domain, dataset):
    directory = CACHE / domain
    if not (OUT / domain / f'{dataset}_FEATURES.json').exists():
        raise RuntimeError('Complete fresh target feature extraction first')
    with (directory / 'geometry.pkl').open('rb') as f:
        state = pickle.load(f)
    conditions = archive(TARGETS[dataset]['design'])['conditions']
    return dict(state=state,
        raw=np.load(directory / f'{dataset}_raw.npy', mmap_mode='r'),
        extra=np.load(CACHE / f'{dataset}_D18.npy', mmap_mode='r'),
        c4=np.load(directory / f'{dataset}_c4.npy', mmap_mode='r'),
        embedding=np.load(directory / f'{dataset}_frequency.npy', mmap_mode='r'),
        conditions=conditions)


def take(context, rows):
    components = transform(context['raw'][rows], context['c4'][rows], context['extra'][rows], context['state'])
    components['embedding'] = np.asarray(context['embedding'][rows], dtype=float)
    components['conditions'] = context['conditions'][rows]
    return components


def kernels(query, reference):
    relation = np.exp(-np.sum((query['conditions'][:, None]-reference['conditions'][None])**2, axis=2)/.5)
    c4 = query['phi'] @ reference['phi'].T
    condition = c4 + ETA*(query['mean'] @ reference['mean'].T)*relation
    return dict(raw=query['raw_phi'] @ reference['raw_phi'].T, c4=c4, condition=condition,
                full=condition+BETA*(query['embedding'] @ reference['embedding'].T)*relation,
                single=query['single_expert'] @ reference['single_expert'].T,
                dual=query['dual_expert'] @ reference['dual_expert'].T)


def fit_support(reference, design, positions):
    k = kernels(reference, reference)
    result = {mode: [] for mode in (*DIRECT, 'unified_single', 'unified_dual')}
    for draw, ix in enumerate(positions):
        sy = design['support_labels'][draw]
        for mode, (kernel, prior) in DIRECT.items():
            result[mode].append(coefficients(k[kernel][np.ix_(ix, ix)], reference[prior][ix], sy, 8.))
        for architecture in ('single', 'dual'):
            combined = .5*(k['full']+k[architecture])
            prior = .5*(reference['prior']+reference[f'{architecture}_prior'])
            result[f'unified_{architecture}'].append(coefficients(combined[np.ix_(ix, ix)], prior[ix], sy, 8.))
    return {mode: np.stack(values) for mode, values in result.items()}


def scores(query, k, positions, alpha, draw):
    ix = positions[draw]
    result = {mode: query[prior] + k[kernel][:, ix] @ alpha[mode][draw]
              for mode, (kernel, prior) in DIRECT.items()}
    for architecture in ('single', 'dual'):
        result[f'late_{architecture}'] = late(result['base_full'], result[f'expert_{architecture}'])
        result[f'late_{architecture}_no_frequency'] = late(result['base_condition'], result[f'expert_{architecture}'])
        prior = .5*(query['prior']+query[f'{architecture}_prior'])
        result[f'unified_{architecture}'] = prior + .5*(k['full'][:, ix]+k[architecture][:, ix]) @ alpha[f'unified_{architecture}'][draw]
    return result


@threadpool_limits.wrap(limits=2, user_api='blas')
def predict_target(domain, dataset):
    folder = CACHE / domain / dataset
    if (folder / 'PREDICTIONS_SAVED.json').exists():
        return
    folder.mkdir(exist_ok=True)
    context = target_context(domain, dataset)
    design = archive(OUT / f'{dataset}_design.npz')
    np.testing.assert_array_equal(design['rows'], np.arange(len(context['raw'])))
    refrows = np.unique(design['support'])
    reference = take(context, refrows)
    positions = np.searchsorted(refrows, design['support'])
    alpha = fit_support(reference, design, positions)
    np.savez(folder / 'support_state.npz', refrows=refrows, positions=positions,
             **{f'ref_{k}': v for k, v in reference.items()}, **{f'alpha_{k}': v for k, v in alpha.items()})
    n = len(design['rows'])
    predictions = np.lib.format.open_memmap(folder / 'predictions.npy', mode='w+', dtype=np.int8,
                                            shape=(5, len(MODES), n))
    saved_scores = np.lib.format.open_memmap(folder / 'scores.npy', mode='w+', dtype=np.float64,
                                            shape=(5, len(MODES), n, 8))
    priors = np.lib.format.open_memmap(folder / 'priors.npy', mode='w+', dtype=np.float64, shape=(3, n, 8))
    for lo in range(0, n, 512):
        hi = min(lo+512, n)
        query = take(context, np.arange(lo, hi))
        k = kernels(query, reference)
        for pi, name in enumerate(('prior', 'single_prior', 'dual_prior')):
            priors[pi, lo:hi] = query[name]
        for draw in range(5):
            result = scores(query, k, positions, alpha, draw)
            for mi, mode in enumerate(MODES):
                value = result[mode]
                if not np.isfinite(value).all():
                    raise ValueError(f'Nonfinite {domain}/{dataset}/{mode} scores')
                saved_scores[draw, mi, lo:hi] = value
                predictions[draw, mi, lo:hi] = value.argmax(1)
        if lo % 16384 == 0:
            print('READOUT', domain, dataset, hi, '/', n, flush=True)
    for a in (predictions, saved_scores, priors):
        a.flush()
    write(folder / 'PREDICTIONS_SAVED.json', dict(domain=domain, dataset=dataset, modes=list(MODES),
          primary_modes=['late_single', 'late_dual'], labels_per_draw=64, draws=5, rows=n,
          query_labels_read=False, priors_order=['base', 'single', 'dual'],
          score_types='late_* are equal-softmax probabilities; remaining modes are ridge scores',
          features_from_new_source_only_SSL=True))


def report():
    if not (OUT / 'ALL_PREDICTIONS_FROZEN.json').exists():
        raise RuntimeError('Freeze every arm on both datasets before reading query labels')
    records, classes = [], []
    for ds, paths in TARGETS.items():
        design = archive(OUT / f'{ds}_design.npz')
        frame = pd.read_csv(paths['manifest'], usecols=['sample_id', 'class_id', 'class_name']).iloc[design['rows']]
        np.testing.assert_array_equal(frame.sample_id.to_numpy(str), design['ids'])
        truth = frame.class_id.to_numpy(int)
        np.testing.assert_array_equal(truth[design['support']], design['support_labels'])
        for domain in ('G', 'D'):
            prediction = np.load(CACHE / domain / ds / 'predictions.npy', mmap_mode='r')
            for draw, mask in enumerate(design['eval_masks']):
                for mi, mode in enumerate(MODES):
                    y, p = truth[mask], prediction[draw, mi, mask]
                    records.append(dict(dataset=ds, domain=domain, draw=draw, mode=mode,
                                        n=len(y), **metrics(y, p)))
                    for c in np.unique(y):
                        cm = y == c
                        classes.append(dict(dataset=ds, domain=domain, draw=draw, mode=mode,
                                            class_id=int(c), n=int(cm.sum()), recall=100*float(np.mean(p[cm] == c))))
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / 'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT / 'per_class.csv', index=False)
    summary = frame.groupby(['dataset', 'domain', 'mode'], sort=False)[['oa', 'ba']].mean().reset_index()
    summary.to_csv(OUT / 'summary.csv', index=False)
    write(OUT / 'READOUT_COMPLETE.json', dict(datasets=2, configurations=4, draws_per_configuration=5,
          metric_rows=len(frame), neural_seeds=1, source_only_new_neural_training=True,
          primary='late_single and late_dual for G/D', predictions_frozen_before_query_truth=True))
    print(summary.to_string(index=False), flush=True)


def main():
    for domain in ('G', 'D'):
        for ds in ('oss2', 'f0921'):
            predict_target(domain, ds)
    write(OUT / 'ALL_PREDICTIONS_FROZEN.json', dict(domains=['G', 'D'], datasets=['oss2', 'f0921'],
          target_query_labels_read=False, cache=str(CACHE)))
    report()


if __name__ == '__main__':
    main()
