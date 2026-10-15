"""Bounded diagnosis of scene correction: reference offset and decision alignment."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scene_shared import BASE, OUT, LEVELS, MODES, view, cosine, corrected_scores


def bootstrap_class_macro(frame, value, seed=20261010):
    cell = frame.groupby(['product', 'class_id'])[value].mean().unstack()
    values = cell.to_numpy()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(2000):
        sample = values[rng.integers(len(values), size=len(values))]
        estimates.append(float(np.nanmean(np.nanmean(sample, axis=0))))
    return float(np.nanmean(np.nanmean(values, axis=0))), *np.percentile(estimates, [2.5, 97.5])


def bootstrap_BA(g, numerator, seed=20261010):
    a = g.groupby(['product', 'class_id'])[[numerator, 'n']].sum()
    sums = a[numerator].unstack(fill_value=0).to_numpy(float)
    ns = a.n.unstack(fill_value=0).to_numpy(float)
    rng = np.random.default_rng(seed)
    point = 100*np.mean(sums.sum(0)/ns.sum(0))
    estimates = []
    for _ in range(2000):
        index = rng.integers(len(sums), size=len(sums))
        count = ns[index].sum(0)
        estimates.append(100*np.mean(sums[index].sum(0)[count > 0]/count[count > 0]))
    return point, *np.percentile(estimates, [2.5, 97.5])


def diagnose():
    with np.load(BASE/'FROZEN_PREDICTIONS.npz') as z:
        scores = z['scores']
    roots, alignments, decision_rows = [], [], []
    for fold in range(3):
        matching = pd.read_csv(OUT/f'fold_{fold}_matches.csv')
        geo = pd.read_csv(OUT/f'fold_{fold}_geometry.csv')
        with np.load(OUT/f'fold_{fold}_residual_vectors.npz') as z:
            same, other, query = z['same'], z['control'], z['query']
        with np.load(BASE/f'fold_{fold}_models.npz') as z:
            model = {k: z[k] for k in z.files}
        index = geo.set_index(['product', 'class_id', 'level', 'mode'])
        for row in matching.itertuples(index=False):
            oi = np.fromstring(str(row.query_obs_ids), sep=',', dtype=int)
            cls = row.class_id
            for mi, mode in enumerate(MODES):
                w = model[mode+'_head_coef']
                wc = w-w.mean(0)
                ids = [int(index.loc[row.product, cls, level, mode].vector_id) for level in LEVELS]
                g, local = ids
                reference_offset = same[g]-same[local]
                np.testing.assert_allclose(reference_offset, other[g]-other[local], atol=1e-6, rtol=1e-5)
                score = scores[oi, mi]
                before = score.argmax(1)
                after = corrected_scores(score, view(reference_offset, mode), w, .5).argmax(1)
                roots.append(dict(fold=fold, product=row.product, port=row.port, class_id=cls, mode=mode,
                    n=len(oi), correct_base=int((before == cls).sum()), correct_after=int((after == cls).sum()),
                    rescues=int(((before != cls) & (after == cls)).sum()),
                    harms=int(((before == cls) & (after != cls)).sum())))
                for level, vi in zip(LEVELS, ids):
                    sd, od, qd = [view(v[vi], mode)@wc.T for v in (same, other, query)]
                    alignments.append(dict(fold=fold, product=row.product, class_id=cls, mode=mode,
                        level=level, same_decision_cosine=cosine(sd, qd),
                        control_decision_cosine=cosine(od, qd),
                        decision_cosine_advantage=cosine(sd, qd)-cosine(od, qd)))
                    rival_scores = score.copy(); rival_scores[:, cls] = -np.inf
                    rival = rival_scores.argmax(1)
                    ds = view(same[vi], mode)@w.T
                    dc = view(other[vi], mode)@w.T
                    for state, mask in [('base_correct', before == cls), ('base_wrong', before != cls)]:
                        if not mask.any():
                            continue
                        decision_rows.append(dict(fold=fold, product=row.product, class_id=cls, mode=mode,
                            level=level, state=state, n=int(mask.sum()),
                            same_damage_sum=float((ds[rival[mask]]-ds[cls]).sum()),
                            other_damage_sum=float((dc[rival[mask]]-dc[cls]).sum())))
    roots, alignments, decisions = pd.DataFrame(roots), pd.DataFrame(alignments), pd.DataFrame(decision_rows)
    roots.to_csv(OUT/'reference_component_outcomes.csv', index=False)
    alignments.to_csv(OUT/'decision_alignment_cells.csv', index=False)
    decisions.to_csv(OUT/'margin_damage_cells.csv', index=False)
    root_summary, root_classes = [], []
    for mode, g in roots.groupby('mode'):
        c = g.groupby('class_id')[['n', 'correct_base', 'correct_after', 'rescues', 'harms']].sum()
        delta = 100*(c.correct_after-c.correct_base)/c.n
        root_summary.append(dict(mode=mode, delta_BA=delta.mean(), rescue_BA=(100*c.rescues/c.n).mean(),
                                 harm_BA=(100*c.harms/c.n).mean()))
        root_classes.extend(dict(mode=mode, class_id=cls, delta_BA=val) for cls, val in delta.items())
    pd.DataFrame(root_summary).to_csv(OUT/'reference_component_summary.csv', index=False)
    pd.DataFrame(root_classes).to_csv(OUT/'reference_component_by_class.csv', index=False)
    intervals = []
    geometry = pd.concat([pd.read_csv(OUT/f'fold_{f}_geometry.csv') for f in range(3)])
    for level in LEVELS:
        for mode in MODES:
            for metric, frame in [('cosine_advantage', geometry), ('decision_cosine_advantage', alignments)]:
                part = frame.loc[frame.level.eq(level) & frame['mode'].eq(mode)]
                mean, lo, hi = bootstrap_class_macro(part, metric)
                intervals.append(dict(level=level, mode=mode, metric=metric, mean=mean, low=lo, high=hi))
    pd.DataFrame(intervals).to_csv(OUT/'alignment_intervals.csv', index=False)
    outcomes = pd.concat([pd.read_csv(OUT/f'fold_{f}_outcomes.csv') for f in range(3)])
    outcomes['delta_count'] = outcomes.correct_after-outcomes.correct_base
    intervals = []
    for key, g in outcomes[outcomes.strength.eq(.5)].groupby(['level', 'mode']):
        s, c = g[g.arm.eq('same_scene')], g[g.arm.eq('other_scene')]
        shared = s.merge(c, on=['fold', 'product', 'class_id', 'mode', 'level'], suffixes=('_s', '_c'), validate='one_to_one')
        shared['n'] = shared.n_s
        shared['difference_count'] = shared.correct_after_s-shared.correct_after_c
        for arm, frame, numerator in [('same_vs_base', s, 'delta_count'),
                                       ('same_vs_control', shared, 'difference_count')]:
            mean, lo, hi = bootstrap_BA(frame, numerator)
            intervals.append(dict(level=key[0], mode=key[1], comparison=arm, mean=mean, low=lo, high=hi))
    pd.DataFrame(intervals).to_csv(OUT/'classification_intervals.csv', index=False)
    dm = decisions.groupby(['level', 'mode', 'state', 'class_id'])[['n', 'same_damage_sum', 'other_damage_sum']].sum()
    rates = dm[['same_damage_sum', 'other_damage_sum']].div(dm.n, axis=0)
    rates.groupby(['level', 'mode', 'state']).mean().to_csv(OUT/'margin_damage_summary.csv')
    print('REFERENCE COMPONENT ONLY', flush=True)
    print(pd.DataFrame(root_summary).to_string(index=False), flush=True)
    print('DECISION SPACE ALIGNMENT', flush=True)
    print(alignments.groupby(['level', 'mode', 'class_id', 'product'])[
        ['same_decision_cosine', 'control_decision_cosine', 'decision_cosine_advantage']].mean().groupby(
        ['level', 'mode', 'class_id']).mean().groupby(['level', 'mode']).mean().to_string(), flush=True)
    print('CLASSIFICATION INTERVALS', flush=True)
    print(pd.DataFrame(intervals).round(4).to_string(index=False), flush=True)


def figures():
    geo = pd.read_csv(OUT/'geometry_summary.csv')
    align = pd.read_csv(OUT/'alignment_intervals.csv')
    result = pd.read_csv(OUT/'classification_summary.csv')
    reference = pd.read_csv(OUT/'reference_component_summary.csv').set_index('mode')
    local = geo[geo.level.eq(LEVELS[1])].set_index('mode').loc[list(MODES)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
    x = np.arange(3)
    colors = ['#19776b', '#bf527a', '#665545']
    axes[0].bar(x-.17, local.same_cosine, width=.34, color=colors[0], label='Same scene, other ship classes')
    axes[0].bar(x+.17, local.control_cosine, width=.34, color=colors[1], label='Other scene, same port')
    axes[0].set(title='Shared feature shift after port/class centering', ylabel='Cosine similarity', xticks=x, xticklabels=MODES)
    axes[0].legend(fontsize=8, loc='upper left')
    axes[0].set_ylim(0, .4)
    for j, metric in enumerate(['cosine_advantage', 'decision_cosine_advantage']):
        a = align[align.level.eq(LEVELS[1]) & align.metric.eq(metric)].set_index('mode').loc[list(MODES)]
        axes[1].errorbar(x+(j-.5)*.15, a['mean'], yerr=[a['mean']-a.low, a.high-a['mean']],
                         fmt='o', capsize=4, color=colors[j], label=['Feature space', 'Class-score space'][j])
    axes[1].axhline(0, color='gray', linewidth=.7)
    axes[1].set(title='Same-scene advantage over matched control', ylabel='Cosine advantage; scene bootstrap 95% interval', xticks=x, xticklabels=MODES)
    axes[1].legend(fontsize=8)
    for j, (level, arm) in enumerate([(LEVELS[1], 'same_scene'), (LEVELS[1], 'other_scene'), (LEVELS[0], 'same_scene')]):
        r = result[result.level.eq(level) & result.arm.eq(arm) & result.strength.eq(.5)].set_index('mode').loc[list(MODES)]
        axes[2].bar(x+(j-1)*.24, r.delta_BA, width=.24, color=colors[j],
                    label=['Within-port reference / same scene', 'Within-port reference / other scene', 'Global reference / same scene'][j])
    axes[2].axhline(0, color='gray', linewidth=.7)
    axes[2].set(title='Subtracting half the estimated shift', ylabel='BA change (percentage points)', xticks=x, xticklabels=MODES)
    axes[2].legend(fontsize=7, loc='lower left')
    fig.suptitle('25,657 queries / 343 scenes / 20 ports | Frozen, label-conditioned diagnostic, NOT a deployable model', fontsize=11)
    fig.savefig(OUT/'scene_mechanism.png', dpi=160)
    fig.savefig(OUT/'scene_mechanism.pdf')
    plt.close(fig)


if __name__ == '__main__':
    diagnose()
    figures()
