"""Summarize fixed module contrasts and trace final fusion decisions."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fusion_module_ablation import (OUT, SOURCE, ROOT, DATASETS, ARMS, READOUTS,
    folders, metadata, load_npz, parts, kernel_for, write)

DATASET_NAMES = dict(self24ports='24 ports (23 tasks)', f0921='F_0921', oss2='OpenSARShip 2.0')
CHINESE_NAMES = dict(self24ports='24港，23任务', f0921='F_0921', oss2='OpenSARShip 2.0')


def truth_for(info, p):
    path, idcol = ((SOURCE/'manifest.csv', 'instance_id') if info['dataset'] == 'self24ports'
        else (ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', 'sample_id'))
    frame = pd.read_csv(path, usecols=[idcol, 'class_id'], keep_default_na=False).iloc[p['rows']]
    np.testing.assert_array_equal(frame[idcol].to_numpy(str), p['ids'])
    return frame.class_id.to_numpy()


def trace_cases():
    from unified_head_expert import ExpertContext
    candidates, transition_rows = {}, []
    for folder in folders():
        info, suffix = metadata(folder)
        saved = load_npz(OUT/'targets'/suffix/'policies.npz')
        y = truth_for(info, saved)
        a, b = ARMS.index('mean_condition'), ARMS.index('full')
        for draw, ev in enumerate(saved['eval_masks']):
            bp0, bp1 = saved['predictions'][draw, 0, a], saved['predictions'][draw, 0, b]
            fp0, fp1 = saved['predictions'][draw, 1, a], saved['predictions'][draw, 1, b]
            old, new, f0, f1 = bp0 == y, bp1 == y, fp0 == y, fp1 == y
            for status, mask in [('base_rescue', ~old & new), ('base_harm', old & ~new),
                                  ('base_both_correct', old & new), ('base_both_wrong', ~old & ~new)]:
                transition_rows.append(dict(dataset=info['dataset'], task=info['task'], draw=draw,
                    base_status=status, n=int(np.sum(ev & mask)),
                    fusion_rescues=int(np.sum(ev & mask & ~f0 & f1)),
                    fusion_harms=int(np.sum(ev & mask & f0 & ~f1))))
            types = dict(final_rescue=ev & ~f0 & f1, final_harm=ev & f0 & ~f1,
                         base_rescue_blocked=ev & ~old & new & ~f0 & ~f1)
            for kind, mask in types.items():
                key = (info['dataset'], kind)
                if key not in candidates and mask.any():
                    indices = np.flatnonzero(mask)
                    q = int(indices[np.argmin(saved['ids'][indices])])
                    candidates[key] = dict(folder=folder, task=info['task'], draw=draw, q=q,
                        true=int(y[q]), rival=int(fp0[q] if kind == 'final_rescue' else fp1[q]), kind=kind)
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(OUT/'frequency_error_flow.csv', index=False)
    transitions.groupby(['dataset', 'base_status'])[['n', 'fusion_rescues', 'fusion_harms']].sum().to_csv(
        OUT/'frequency_error_flow_totals.csv')
    results = []
    for folder in sorted(set(c['folder'] for c in candidates.values())):
        info, suffix = metadata(folder)
        data = load_npz(folder/'target.npz')
        z = load_npz(folder/'selective_embedding.npz')['embedding'].astype(float)
        saved = load_npz(OUT/'targets'/suffix/'policies.npz')
        context = ExpertContext(folder)
        for key, case in candidates.items():
            if case['folder'] != folder:
                continue
            draw, q, a, b = case['draw'], case['q'], case['true'], case['rival']
            support = data['support'][draw]
            # Kernel rows and coefficients are reused; only selected query scores are reconstructed.
            indices = np.r_[support, q]
            reduced = {k: data[k][indices] for k in ('phi', 'mean', 'conditions')}
            p = parts(reduced, z[indices], np.arange(64))
            oldk = kernel_for(p, np.arange(64), info['eta'], 'mean_condition')[-1]
            fk = .25*p['frequency'][-1]*p['condition'][-1]
            ac0 = saved['coefficients'][draw, 0, ARMS.index('mean_condition')]
            ac1 = saved['coefficients'][draw, 0, ARMS.index('full')]
            b0 = data['prior'][q]+oldk@ac0
            b1 = data['prior'][q]+(oldk+fk)@ac1
            pe, ke = context.target_components(data, support)
            from unified_head_core import coefficients
            ae = coefficients(ke[support], pe[support], data['support_labels'][draw], 8.)
            e = pe[q]+ke[q]@ae
            pb0, pb1, ep = softmax(b0), softmax(b1), softmax(e)
            f0, f1 = .5*(pb0+ep), .5*(pb1+ep)
            np.testing.assert_array_equal([b0.argmax(), b1.argmax(), f0.argmax(), f1.argmax()],
                [saved['predictions'][draw, ri, ARMS.index(arm), q]
                 for ri, arm in [(0, 'mean_condition'), (0, 'full'), (1, 'mean_condition'), (1, 'full')]])
            refit = oldk*((ac1[:, a]-ac1[:, b])-(ac0[:, a]-ac0[:, b]))
            direct = fk*(ac1[:, a]-ac1[:, b])
            np.testing.assert_allclose((b1[a]-b1[b])-(b0[a]-b0[b]), refit.sum()+direct.sum(), atol=1e-10)
            row = dict(dataset=info['dataset'], task=info['task'], draw=draw, kind=case['kind'],
                query_id=str(data['ids'][q]), true_class=a, rival_class=b,
                base_old=int(b0.argmax()), base_new=int(b1.argmax()), expert=int(e.argmax()),
                fusion_old=int(f0.argmax()), fusion_new=int(f1.argmax()),
                base_margin_old=float(b0[a]-b0[b]), refit_delta=float(refit.sum()),
                frequency_direct=float(direct.sum()), base_margin_new=float(b1[a]-b1[b]),
                base_probability_half_old=float(.5*(pb0[a]-pb0[b])),
                base_probability_half_new=float(.5*(pb1[a]-pb1[b])),
                expert_probability_half=float(.5*(ep[a]-ep[b])),
                fusion_margin_old=float(f0[a]-f0[b]), fusion_margin_new=float(f1[a]-f1[b]))
            results.append(row)
            name = f"{info['dataset']}_{case['kind']}"
            case_dir = OUT/'cases'
            case_dir.mkdir(exist_ok=True)
            pd.DataFrame(dict(support_id=data['ids'][support], label=data['support_labels'][draw],
                refit_contribution=refit, frequency_contribution=direct,
                total_contribution=refit+direct)).to_csv(case_dir/f'{name}_support.csv', index=False)
            fig, ax = plt.subplots(figsize=(6.2, 3.5))
            ax.bar(['Base old / 2', 'Base new / 2', 'Expert / 2', 'Fusion old', 'Fusion new'],
                [row['base_probability_half_old'], row['base_probability_half_new'],
                 row['expert_probability_half'], row['fusion_margin_old'], row['fusion_margin_new']],
                color=['#7298b5', '#286488', '#ad5b45', '#777777', '#3b8566'])
            ax.axhline(0, color='black', lw=.7)
            ax.set_ylabel(f'Probability margin: true {a} - rival {b}')
            ax.set_title(f"{DATASET_NAMES[info['dataset']]}: {case['kind']}\n{row['query_id']}", fontsize=10)
            ax.tick_params(axis='x', labelsize=8)
            fig.tight_layout()
            fig.savefig(case_dir/f'{name}.png', dpi=180)
            plt.close(fig)
    pd.DataFrame(results).to_csv(OUT/'case_margins.csv', index=False)
    write(OUT/'cases'/'PROVENANCE.json', dict(selection='First sorted task, earliest draw, lexicographically first eligible query per dataset/event',
        cases=len(results), purpose='Illustrative rescues, harms and blocked base rescues; not representative physical evidence',
        exact_prediction_and_margin_replay=True, support_contributions='Signed raw base margin; not additive fusion probabilities or accuracy'))


def plot_effects():
    effects = pd.read_csv(OUT/'effects.csv')
    contrasts = ('frequency_added', 'full_pair_condition', 'full_stack')
    labels = ('Frequency added', 'Pair condition\nvs uniform shrinkage', 'All added relations')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, dataset in zip(axes, CHINESE_NAMES):
        for j, (readout, color) in enumerate(zip(READOUTS, ('#397fa6', '#bd6547', '#438368'))):
            e = effects[(effects.dataset == dataset) & (effects.readout == readout)].set_index('contrast')
            values = e.loc[list(contrasts), 'delta_ba'].to_numpy()
            ax.bar(np.arange(3)+(j-1)*.24, values, .23, label=readout, color=color)
        ax.axhline(0, color='black', lw=.7)
        ax.set_xticks(np.arange(3), labels, fontsize=8)
        ax.set_title(DATASET_NAMES[dataset], fontsize=11)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Paired BA change (percentage points)')
    axes[-1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT/'module_effects.png', dpi=200)
    fig.savefig(OUT/'module_effects.pdf')
    plt.close(fig)


if __name__ == '__main__':
    if not (OUT/'COMPLETE.json').exists():
        raise RuntimeError('Finish fixed ablations before case selection')
    plot_effects()
    trace_cases()
    print('Saved module figure, nine exact fusion case traces and signed support contributions.', flush=True)
