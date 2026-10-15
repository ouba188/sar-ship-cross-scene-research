"""Read out fixed wavelet-statistic additions without target model selection."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_dfwa_wavelet_probe import OUT, ROOT, SOURCE, DATASETS, MODES


def metrics(y, p):
    correct = y == p
    return 100*correct.mean(), 100*np.mean([correct[y == c].mean() for c in np.unique(y)])


def main():
    source = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id'])
    folders = [('self24ports', p.name, p) for p in sorted((OUT/'targets/self24ports').iterdir())]
    folders += [(name, name, OUT/'targets'/name) for name in ['oss2', 'f0921']]
    records, classes = [], []
    for dataset, task, path in folders:
        frame = source if dataset == 'self24ports' else pd.read_csv(ROOT/DATASETS[dataset][0]/'target_manifest.csv', usecols=['class_id'])
        with np.load(path/'policies.npz') as f:
            p = {k: f[k] for k in f.files}
        with np.load(OUT/'half_mean_control'/path.relative_to(OUT)/'policies.npz') as f:
            half = {k: f[k] for k in f.files}
        np.testing.assert_array_equal(p['query_rows'], half['query_rows'])
        np.testing.assert_array_equal(p['eval_masks'], half['eval_masks'])
        y = frame.class_id.to_numpy()[p['query_rows']]
        for draw, ev in enumerate(p['eval_masks']):
            for stage, key in [('base', 'base_predictions'), ('full', 'predictions')]:
                base, mean = p[key][draw, 0], p[key][draw, 1]
                boa, bba = metrics(y[ev], base[ev])
                moa, mba = metrics(y[ev], mean[ev])
                hoa, hba = metrics(y[ev], half[key][draw, 1, ev])
                for mode, pred in zip(MODES, p[key][draw]):
                    oa, ba = metrics(y[ev], pred[ev])
                    records.append(dict(dataset=dataset, task=task, draw=draw, stage=stage, mode=mode,
                        oa=oa, ba=ba, delta_c4_oa=oa-boa, delta_c4_ba=ba-bba,
                        delta_mean_oa=oa-moa, delta_mean_ba=ba-mba,
                        half_mean_oa=hoa, half_mean_ba=hba,
                        delta_half_oa=oa-hoa, delta_half_ba=ba-hba,
                        rescues_vs_mean=int((ev & (pred == y) & (mean != y)).sum()),
                        harms_vs_mean=int((ev & (pred != y) & (mean == y)).sum()),
                        changed_vs_mean=int((ev & (pred != mean)).sum())))
                    for cl in np.unique(y[ev]):
                        mask = ev & (y == cl)
                        classes.append(dict(dataset=dataset, task=task, draw=draw, stage=stage, mode=mode,
                            class_id=int(cl), n=int(mask.sum()), support_count=int((y[p['support_rows'][draw]] == cl).sum()),
                            recall=100*(pred[mask] == cl).mean(),
                            delta_mean_recall=100*((pred[mask] == cl).mean()-(mean[mask] == cl).mean())))
    data = pd.DataFrame(records)
    data.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_metrics.csv', index=False)
    measures = ['oa', 'ba', 'delta_c4_oa', 'delta_c4_ba', 'delta_mean_oa', 'delta_mean_ba',
                'half_mean_oa', 'half_mean_ba', 'delta_half_oa', 'delta_half_ba']
    task = data.groupby(['dataset', 'task', 'stage', 'mode'])[measures].mean().reset_index()
    task.to_csv(OUT/'task_means.csv', index=False)
    summary = task.groupby(['dataset', 'stage', 'mode'])[measures].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    data.groupby(['dataset', 'draw', 'stage', 'mode'])[measures].mean().to_csv(OUT/'draw_metrics.csv')
    data.groupby(['dataset', 'stage', 'mode'])[['rescues_vs_mean', 'harms_vs_mean', 'changed_vs_mean']].sum().to_csv(
        OUT/'event_summary.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True)
    labels = ['+ low-frequency', '+ high-frequency', '+ both']
    for ax, dataset in zip(axes, ['self24ports', 'f0921', 'oss2']):
        part = summary[summary.dataset.eq(dataset) & summary.stage.eq('full')].set_index('mode').loc[list(MODES[2:])]
        y = np.arange(3)
        ax.barh(y-.17, part.delta_mean_oa, .32, color='#237A78', label='OA')
        ax.barh(y+.17, part.delta_mean_ba, .32, color='#B34463', label='BA')
        ax.set_yticks(y, labels)
        ax.set_title(dataset)
        ax.set_xlabel('Change from mean-only control (pp)')
        ax.axvline(0, color='#555555', linewidth=.8)
        ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    axes[-1].legend()
    fig.suptitle('DFWA-inspired fixed wavelet probe, not a trained DFWA-Net')
    fig.tight_layout()
    fig.savefig(OUT/'wavelet_increment.png', dpi=160)
    plt.close(fig)
    (OUT/'READOUT_COMPLETE.json').write_text(json.dumps(dict(
        tasks=len(folders), draws=5, modes=list(MODES), eta=.25, selected_after_target_readout=False,
        independent_confirmation=False, current_model_promoted=False), indent=2), encoding='utf-8')
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
