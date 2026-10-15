"""Fixed prior/kernel crossover with exact score and probability accounting."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import gc
import time
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from backbone_ablation_core import folders, info_for, write, OUT as PREVIOUS
from dual_evidence_core import OUT, BATCH, ARMS, HEADS, MODES, load_task, base_cross, EvidenceContext
from unified_head_core import coefficients, late


def predict_task(folder):
    info, suffix = info_for(folder)
    dst = OUT/'targets'/suffix
    if (dst/'COMPLETE.json').exists():
        print('RESUME', suffix, flush=True)
        return
    started = time.perf_counter()
    data = load_task(folder)
    n = len(data['rows'])
    ctx = EvidenceContext(folder, data)
    with np.load(PREVIOUS/'targets'/suffix/'predictions.npz') as z:
        old_modes, old_predictions = list(z['modes']), z['predictions']
        for key in ('rows', 'ids', 'support', 'eval_masks'):
            np.testing.assert_array_equal(data[key], z[key])
    union = np.unique(data['support'])
    ref = ctx.features(union)
    kref = ctx.cross(ref, ref)
    alphas, states = {}, []
    maximum_coefficient_error = 0.
    for draw, support in enumerate(data['support']):
        col = np.searchsorted(union, support)
        labels = data['support_labels'][draw]
        kb = base_cross(data, support, support, info['eta'])
        ab = coefficients(kb, data['prior'][support], labels, 8.)
        np.testing.assert_allclose(ab, ctx.saved[f'{draw}_base'], atol=1e-9, rtol=1e-9)
        alphas[f'{draw}_base'] = ab
        state = dict(col=col, support=support, base=ab)
        for arm in ARMS:
            prior = ref['priors']['r50' if arm[0] == 'R' else 'dual'][col]
            ke = kref[arm[1]][np.ix_(col, col)]
            ae = coefficients(ke, prior, labels, 8.)
            aj = coefficients(.5*kb+.5*ke, .5*data['prior'][support]+.5*prior, labels, 8.)
            state[arm] = (ae, aj)
            alphas[f'{draw}_{arm}_expert'], alphas[f'{draw}_{arm}_unified'] = ae, aj
            if arm in ('RR', 'DD'):
                old = 'r50' if arm == 'RR' else 'dual_refit'
                for head, actual in [('expert', ae), ('unified', aj)]:
                    maximum_coefficient_error = max(maximum_coefficient_error,
                        float(np.max(abs(actual-ctx.saved[f'{draw}_{old}_{head}']))))
                    np.testing.assert_allclose(actual, ctx.saved[f'{draw}_{old}_{head}'],
                                               atol=3e-7, rtol=1e-6)
        states.append(state)
    # Store scores, not the much larger N x support-union kernels.
    dst.mkdir(parents=True, exist_ok=True)
    scores = np.lib.format.open_memmap(dst/'scores.npy', mode='w+', dtype=np.float64,
                                      shape=(5, len(MODES), n, 8))
    parts = np.lib.format.open_memmap(dst/'expert_delta_parts.npy', mode='w+', dtype=np.float64,
                                     shape=(5, 3, n, 8))
    maximum_error, maximum_probability_error = 0., 0.
    predictions = np.empty((5, len(MODES), n), dtype=np.int8)
    for lo in range(0, n, BATCH):
        ix = np.arange(lo, min(lo+BATCH, n))
        features = ctx.features(ix)
        kernels = ctx.cross(features, ref)
        prior = {'R': features['priors']['r50'], 'D': features['priors']['dual']}
        for draw, state in enumerate(states):
            kb = base_cross(data, ix, state['support'], info['eta'])
            b = data['prior'][ix]+kb@state['base']
            results = {'base': b}
            kr, kd = (kernels[key][:, state['col']] for key in ('R', 'D'))
            for arm in ARMS:
                ke = kr if arm[1] == 'R' else kd
                pe = prior[arm[0]]
                ae, aj = state[arm]
                e = pe+ke@ae
                j = (.5*data['prior'][ix]+.5*pe)+(.5*kb+.5*ke)@aj
                results.update({arm+'_expert': e, arm+'_late': late(b, e), arm+'_unified': j})
            ar, ad = state['RR'][0], state['DD'][0]
            delta_parts = np.stack([prior['D']-prior['R'], kr@(ad-ar), (kd-kr)@ad])
            error = np.max(np.abs(delta_parts.sum(0)-(results['DD_expert']-results['RR_expert'])))
            maximum_error = max(maximum_error, float(error))
            np.testing.assert_allclose(delta_parts.sum(0), results['DD_expert']-results['RR_expert'],
                                       atol=1e-10, rtol=1e-10)
            prob_delta = .5*(softmax(results['DD_expert'], axis=1)-softmax(results['RR_expert'], axis=1))
            probability_error = np.max(np.abs(prob_delta-(results['DD_late']-results['RR_late'])))
            maximum_probability_error = max(maximum_probability_error, float(probability_error))
            np.testing.assert_allclose(prob_delta, results['DD_late']-results['RR_late'], atol=1e-14)
            parts[draw, :, lo:lo+len(ix)] = delta_parts
            for mi, mode in enumerate(MODES):
                value = results[mode]
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite score: '+mode)
                scores[draw, mi, lo:lo+len(ix)] = value
                predictions[draw, mi, ix] = value.argmax(1)
        if lo % (BATCH*24) == 0:
            print(f'CHUNK {suffix} {lo+len(ix)}/{n}', flush=True)
    for new, old in [('base', 'base')] + [(a+'_'+h, b+'_'+h)
        for a, b in [('RR', 'r50'), ('DD', 'dual_refit')] for h in HEADS]:
        np.testing.assert_array_equal(predictions[:, MODES.index(new)],
                                     old_predictions[:, old_modes.index(old)])
    scores.flush()
    parts.flush()
    np.savez_compressed(dst/'predictions.npz', predictions=predictions, modes=np.array(MODES),
        rows=data['rows'], ids=data['ids'], support=data['support'], eval_masks=data['eval_masks'])
    np.savez_compressed(dst/'coefficients.npz', **alphas)
    write(dst/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], draws=5,
        n=n, seconds=time.perf_counter()-started, diagonal_predictions_exact=True,
        source_coefficients='Reused exact matched source-refit coefficients from backbone_ablation_20261010',
        source_rows_exact=True, coefficient_replay_atol=3e-7,
        maximum_diagonal_coefficient_error=maximum_coefficient_error,
        precision_note='Chunk-size changes in inherited float32 projections may change coefficient last bits; all diagonal class predictions must be identical',
        maximum_expert_score_decomposition_error=maximum_error,
        maximum_final_probability_delta_error=maximum_probability_error,
        score_units='expert/base/unified: logits; late: equal mixture of softmax probabilities',
        target_query_labels_used=False, cpu_threads=2))
    print(f'COMPLETE {suffix}: {time.perf_counter()-started:.1f}s, replay exact', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    tasks = folders()
    write(OUT/'PROTOCOL.json', dict(tasks=[info_for(f)[1] for f in tasks], arms=ARMS, modes=MODES,
        support='Original 5 x 64 target labels; full base is fixed in all arms',
        definitions='RR/DR/RD/DD first letter prior, second letter expert kernel; R=matched source-refit R50, D=matched source-refit dual',
        equation='eD-eR=(pD-pR)+KR@(aD-aR)+(KD-KR)@aD',
        equation_scope='Exact computational accounting under dual coefficients, not an independent causal allocation; softmax is nonlinear',
        knobs='No tuning, permutation tests, neural training or GPU',
        confidence_strata='R50-only final probability top1-top2 gap: <0.05 uncertain; [0.05,0.20) moderate; >=0.20 strong',
        cases='Posthoc draw0; first task/sample-ID available for each dataset and type: dual rescue, dual harm, both wrong. Not representative prevalence estimates.',
        evaluation='All predictions frozen before reading query labels. Same prior/kernel interventions applied to expert, late fusion and unified.',
        limitations=['Inherited representations and development-used datasets',
            'F0921 historical SSL pretraining target-corpus exposure',
            'No claim of distinct physical mechanisms or impossibility of a capable single backbone']))
    with threadpool_limits(limits=2):
        for folder in tasks:
            if args.tasks and info_for(folder)[0]['task'] not in args.tasks:
                continue
            predict_task(folder)
            gc.collect()


if __name__ == '__main__':
    main()
