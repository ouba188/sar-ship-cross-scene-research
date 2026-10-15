"""Trace the extra-target-label reader's corrections without training a new model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment import OUT as PARENT, TASKS, SEEDS, residual, softmax, write
from report import metric
import numpy as np
import pandas as pd
from math import factorial
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
NAMES = ('bias', 'r50_evidence', 'r18_evidence')


def decompose(x, source, wu, bu, wt, bt):
    mu = x[source].mean(0)
    xc = x-mu
    dw = wt-wu
    b = bt-bu+mu@dw
    change = xc@dw
    old_scores = xc@wu
    remix = np.linalg.lstsq(old_scores[source], change[source], rcond=1e-6)[0]
    shared = old_scores@remix
    remainder = change-shared
    u, s, v = np.linalg.svd(dw, full_matrices=False)
    return dict(center=mu, xc=xc, dw=dw, bias=np.broadcast_to(b, change.shape),
                weights=change, remix=shared, remainder=remainder, singular=s,
                lowrank={k:xc@((u[:, :k]*s[:k])@v[:k]) for k in (1, 2, 4)})


def shapley(values):
    out = []
    for j in range(3):
        contribution = np.zeros_like(values[0])
        for subset in range(8):
            if subset & (1 << j):
                continue
            n = subset.bit_count()
            weight = factorial(n)*factorial(2-n)/factorial(3)
            contribution += weight*(values[subset | (1 << j)]-values[subset])
        out.append(contribution)
    return np.stack(out, axis=1)


def support_decomposition(x, support, dw, db):
    h = np.c_[x, np.ones(len(x))]
    _, singular, vectors = np.linalg.svd(h[support], full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    basis = vectors[:rank].T
    theta = np.vstack([dw, db])
    inside = basis@(basis.T@theta)
    outside = theta-inside
    return dict(inside=h@inside, outside=h@outside, rank=rank,
                null_on_support=float(np.max(np.abs(h[support]@outside))),
                parameter_outside_ratio=float(np.sum(outside**2)/np.sum(theta**2)))


def run():
    frame = pd.read_csv(PARENT/'manifest.csv', dtype={'mmsi': str})
    rows_metrics, records, pairs, structural, coefficients, branch = [], [], [], [], [], []
    endpoints, spans = [], []
    for task in TASKS:
        for seed in SEEDS:
            folder = PARENT/task/str(seed)
            split = dict(np.load(folder/'split.npz'))
            z = dict(np.load(folder/'features.npz'))
            rows = z.pop('rows')
            ix = pd.Index(rows)
            si, qi, source = (ix.get_indexer(split[k]) for k in ('support', 'query', 'source'))
            labels_support = frame.iloc[split['support']].class_id.to_numpy()
            labels = frame.iloc[split['query']].class_id.to_numpy()
            xmode = dict(r50=z['r50'].astype(float), dual=np.c_[z['r50'], z['r18']].astype(float)/np.sqrt(2.))
            hu = dict(np.load(folder/'frozen/S_U/heads.npz'))
            ht = dict(np.load(folder/'frozen/S_T/heads.npz'))
            prior, decomposition, atoms, support_parts = {}, {}, {}, {}
            for mode, x in xmode.items():
                w, b = hu[mode+'_coef'].astype(float), hu[mode+'_bias'].astype(float)
                wt, bt = ht[mode+'_coef'].astype(float), ht[mode+'_bias'].astype(float)
                prior[mode] = x@w+b
                d = decompose(x, source, w, b, wt, bt)
                decomposition[mode] = d
                sp = support_decomposition(x, si, wt-w, bt-b)
                support_parts[mode] = sp
                np.testing.assert_allclose(sp['inside']+sp['outside'], x@(wt-w)+(bt-b), atol=1e-12)
                if sp['null_on_support'] > 1e-10:
                    raise ValueError('Support-null component changes support scores')
                total_query = sp['inside'][qi]+sp['outside'][qi]
                spans.append(dict(task=task, seed=seed, mode=mode, rank=sp['rank'],
                    feature_dim=x.shape[1]+1, support_null_error=sp['null_on_support'],
                    parameter_outside_ratio=sp['parameter_outside_ratio'],
                    query_outside_energy_ratio=float(np.sum(sp['outside'][qi]**2)/np.sum(total_query**2))))
                if mode == 'r50':
                    atoms[mode] = [d['bias'], d['weights'], np.zeros_like(d['weights'])]
                else:
                    atoms[mode] = [d['bias'], d['xc'][:, :128]@d['dw'][:128],
                                   d['xc'][:, 128:]@d['dw'][128:]]
                for population, pos in [('source', source), ('query', qi)]:
                    total = float(np.sum(d['weights'][pos]**2))
                    structural.append(dict(task=task, seed=seed, mode=mode, population=population,
                        weight_change_energy=total,
                        remainder_energy_ratio=float(np.sum(d['remainder'][pos]**2)/max(total, 1e-30)),
                        bias_rms=float(np.sqrt(np.mean(d['bias'][pos]**2))),
                        weights_rms=float(np.sqrt(np.mean(d['weights'][pos]**2)))))
                for rank in (1, 2, 4):
                    coefficients.append(dict(task=task, seed=seed, mode=mode, rank=rank,
                        delta_weight_energy_ratio=float(np.sum(d['singular'][:rank]**2)/np.sum(d['singular']**2))))

            def evaluate(deltas, which=None):
                adapted = {}
                for mode, x in xmode.items():
                    p = prior[mode]+deltas[mode]
                    adapted[mode] = residual(x, p, si, labels_support)
                if which:
                    return adapted[which]
                return .5*softmax(adapted['r50'])+.5*softmax(adapted['dual'])

            zero = {mode:np.zeros_like(p) for mode,p in prior.items()}
            full = {mode:d['weights']+d['bias'] for mode,d in decomposition.items()}
            interventions = dict(reference=zero, bias_only={m:d['bias'] for m,d in decomposition.items()},
                weights_only={m:d['weights'] for m,d in decomposition.items()},
                old_score_remix_bias={m:d['remix']+d['bias'] for m,d in decomposition.items()},
                new_evidence_bias={m:d['remainder']+d['bias'] for m,d in decomposition.items()},
                support_span_only={m:d['inside'] for m,d in support_parts.items()},
                support_null_only={m:d['outside'] for m,d in support_parts.items()},
                full=full,
                r50_head_only=dict(r50=full['r50'], dual=zero['dual']),
                dual_head_only=dict(r50=zero['r50'], dual=full['dual']))
            for rank in (1, 2, 4):
                interventions['rank'+str(rank)+'_bias'] = {m:d['lowrank'][rank]+d['bias'] for m,d in decomposition.items()}
            saved, all_score = {}, {}
            for name, delta in interventions.items():
                score = evaluate(delta)
                all_score[name] = score
                saved[name] = score[qi]
            dest = OUT/task/str(seed)
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest/'interventions.npz', rows=split['query'], **saved)
            for name, arm in [('reference', 'S_U'), ('full', 'S_T')]:
                original = np.load(folder/'frozen'/arm/'predictions.npz')['fusion']
                error = float(np.max(np.abs(original-saved[name])))
                np.testing.assert_allclose(original, saved[name], atol=1e-6, rtol=0)
                np.testing.assert_array_equal(original.argmax(1), saved[name].argmax(1))
                endpoints.append(dict(task=task, seed=seed, arm=arm, max_abs_error=error))

            a, b = saved['reference'].argmax(1), saved['full'].argmax(1)
            group = np.select([(a != labels) & (b == labels), (a == labels) & (b != labels),
                               (a == labels) & (b == labels)], ['rescued', 'harmed', 'always_correct'], default='always_wrong')
            class_counts = np.bincount(labels, minlength=8)
            for name, score in saved.items():
                mm, _ = metric(labels, score)
                prediction = score.argmax(1)
                rescue, harm = (a != labels) & (prediction == labels), (a == labels) & (prediction != labels)
                rows_metrics.append(dict(task=task, seed=seed, intervention=name, **mm,
                    rescues=int(rescue.sum()), harms=int(harm.sum())))
            rival_scores = saved['reference'].copy()
            rival_scores[np.arange(len(labels)), labels] = -np.inf
            rival = rival_scores.argmax(1)
            margins, subsets = [], []
            for mask in range(8):
                delta = {m:sum((atom for j,atom in enumerate(atoms[m]) if mask & (1 << j)), zero[m]) for m in atoms}
                score = evaluate(delta)[qi]
                subsets.append(score)
                margins.append(score[np.arange(len(labels)), labels]-score[np.arange(len(labels)), rival])
            values = np.stack(margins)
            attribution = shapley(values)
            np.testing.assert_allclose(attribution.sum(1), values[7]-values[0], atol=1e-12)
            np.testing.assert_allclose(subsets[7], saved['full'], atol=1e-12)
            np.savez_compressed(dest/'attribution.npz', rows=split['query'], competitor=rival,
                                reference_margin=values[0], final_margin=values[7], attribution=attribution)
            source_sorted = np.sort(all_score['reference'][source], axis=1)
            cuts = np.quantile(source_sorted[:, -1]-source_sorted[:, -2], [.25, .75])
            ordered = np.sort(saved['reference'], axis=1)
            confidence_gap = ordered[:, -1]-ordered[:, -2]
            confidence_bin = np.where(confidence_gap <= cuts[0], 'low_gap',
                                      np.where(confidence_gap > cuts[1], 'high_gap', 'middle_gap'))
            head_delta = {}
            for mode in xmode:
                before = softmax(evaluate(zero, mode)[qi]); after = softmax(evaluate(full, mode)[qi])
                change = .5*(after-before)
                head_delta[mode] = change[np.arange(len(labels)), labels]-change[np.arange(len(labels)), rival]
                mm_a, _ = metric(labels, before); mm_b, _ = metric(labels, after)
                branch.append(dict(task=task, seed=seed, head=mode, delta_oa=mm_b['oa']-mm_a['oa'], delta_ba=mm_b['ba']-mm_a['ba']))
            np.testing.assert_allclose(head_delta['r50']+head_delta['dual'], values[7]-values[0], atol=1e-12)
            for i, row in enumerate(split['query']):
                meta = frame.iloc[row]
                records.append(dict(task=task, seed=seed, row=int(row), mmsi=meta.mmsi, product=meta['product'],
                    class_id=int(labels[i]), before=int(a[i]), after=int(b[i]), group=group[i],
                    competitor=int(rival[i]), gap=float(confidence_gap[i]), gap_bin=confidence_bin[i],
                    before_margin=float(values[0, i]), after_margin=float(values[7, i]),
                    r50_head_effect=float(head_delta['r50'][i]), dual_head_effect=float(head_delta['dual'][i]),
                    **{name:float(attribution[i,j]) for j,name in enumerate(NAMES)}))
            for c in range(8):
                for d in range(8):
                    if c == d:
                        continue
                    rescued = (labels == c) & (a == d) & (b == c)
                    harmed = (labels == c) & (a == c) & (b == d)
                    rr = frame.iloc[split['query'][rescued]]; hh = frame.iloc[split['query'][harmed]]
                    pairs.append(dict(task=task, seed=seed, true_class=c, competitor=d,
                        n_class=int(class_counts[c]), rescues=int(rescued.sum()), harms=int(harmed.sum()),
                        rescue_ships=int(rr.mmsi.nunique()), harm_ships=int(hh.mmsi.nunique()),
                        rescue_scenes=int(rr['product'].nunique()), harm_scenes=int(hh['product'].nunique()),
                        ba_contribution=100*float(rescued.sum()-harmed.sum())/class_counts[c]/8))
            print('TRACED', task, seed, flush=True)
    for name, result in [('metrics', rows_metrics), ('observations', records), ('class_pairs', pairs),
                         ('structure', structural), ('lowrank_energy', coefficients), ('branch', branch),
                         ('support_span', spans)]:
        pd.DataFrame(result).to_csv(OUT/(name+'.csv'), index=False)
    m = pd.DataFrame(rows_metrics)
    m.groupby(['task', 'intervention'], sort=False)[['oa', 'ba', 'rescues', 'harms']].mean().reset_index().to_csv(OUT/'mean_metrics.csv', index=False)
    write(OUT/'COMPLETE.json', dict(tasks=TASKS, seeds=SEEDS, query_observation_exposures=len(records),
        new_neural_fits=0, new_label_supervised_fits=0, source_score_projections=16,
        endpoints=endpoints, attribution_additivity_passed=True,
        query_labels_for_diagnostics_only=True, additional_target_labels_in_teacher_only=True))
    print(m.groupby(['task', 'intervention'])[['oa', 'ba']].mean().to_string(), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
