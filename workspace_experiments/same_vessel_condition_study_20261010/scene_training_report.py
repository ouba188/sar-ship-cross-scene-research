"""Report paired outcomes and bounded failure localization, without new fitting."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scene_training import BASE, OUT, MODES, ARMS


def run():
    r = pd.read_csv(OUT/'all_results.csv')
    baseline = r[r.arm.eq('baseline')][['task', 'mode', 'BA', 'OA']].rename(columns={'BA': 'base_BA', 'OA': 'base_OA'})
    r = r.merge(baseline, on=['task', 'mode'])
    r['delta_BA'], r['delta_OA'] = r.BA-r.base_BA, r.OA-r.base_OA
    r.to_csv(OUT/'paired_results.csv', index=False)
    mean = r.groupby(['split', 'arm', 'mode'])[['BA', 'OA', 'delta_BA', 'delta_OA']].mean()
    mean.to_csv(OUT/'mean_results.csv')
    obs = pd.read_csv(BASE/'observations.csv')
    records, confusions, failures = [], [], []
    for task in r.loc[r.split.eq('held_port'), 'task'].unique():
        with np.load(OUT/f'{task}_predictions.npz') as z:
            pred, ids = z['predictions'], z['test_obs_ids']
        meta = obs.iloc[ids].reset_index(drop=True)
        y = meta.class_id.to_numpy()
        for mi, mode in enumerate(MODES):
            base_pred = pred[mi*len(ARMS)]
            for arm in ['scene_consistency', 'semantic_protected_consistency', 'isotropic_regularized']:
                after = pred[mi*len(ARMS)+ARMS.index(arm)]
                for cls, group in meta.groupby('class_id'):
                    ix = group.index.to_numpy()
                    net = (after[ix] == cls).astype(int)-(base_pred[ix] == cls).astype(int)
                    records.append(dict(task=task, mode=mode, arm=arm, class_id=cls,
                        n=len(ix), ships=group.mmsi.nunique(), net_correct=int(net.sum()),
                        recall_change=100*net.mean(), BA_contribution=100*net.sum()/(8*len(ix))))
                if mode == 'dual' and arm == 'scene_consistency':
                    changed = pd.DataFrame(dict(truth=y, before=base_pred, after=after,
                        product=meta['product'], net=(after == y).astype(int)-(base_pred == y).astype(int)))
                    for key, g in changed.loc[changed.net.ne(0)].groupby(['truth', 'before', 'after']):
                        confusions.append(dict(task=task, truth=key[0], before=key[1], after=key[2],
                                               records=len(g), scenes=g['product'].nunique(), net=int(g.net.sum())))
                    weights = np.array([100/(8*np.sum(y == c)) for c in y])
                    changed['BA_contribution'] = changed.net*weights
                    changed.groupby('product')[['net', 'BA_contribution']].sum().to_csv(OUT/f'{task}_scene_attribution.csv')
        with np.load(OUT/f'{task}_models.npz') as z:
            for mode in MODES:
                scene, sem = z[mode+'_scene_cov'], z[mode+'_semantic_projector']
                perpendicular = np.eye(len(scene))-sem
                protected = perpendicular@scene@perpendicular
                base, half, new = [z[mode+'_'+a+'_coef'] for a in ('baseline', 'half_refit', 'semantic_protected_consistency')]
                energy = lambda w, s: float(np.trace(w.T@s@w))
                failures.append(dict(task=task, mode=mode,
                    half_refit_effective_weight_relative_change=float(np.linalg.norm(half-base)/np.linalg.norm(base)),
                    protected_penalty_energy_ratio=energy(new, protected)/energy(base, protected),
                    original_scene_energy_ratio=energy(new, scene)/energy(base, scene)))
    pd.DataFrame(records).to_csv(OUT/'class_gain_attribution.csv', index=False)
    pd.DataFrame(confusions).to_csv(OUT/'dual_scene_constraint_confusions.csv', index=False)
    pd.DataFrame(failures).to_csv(OUT/'failure_localization.csv', index=False)

    selected = ['half_test_only', 'half_refit', 'remove_test_only', 'remove_refit']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, split in zip(axes, ['identity', 'held_port']):
        values = mean.loc[split].reset_index().pivot(index='mode', columns='arm', values='delta_BA').loc[list(MODES), selected].to_numpy()
        im = ax.imshow(values, cmap='RdBu', vmin=-4, vmax=4, aspect='auto')
        ax.set_xticks(range(4), ['Half: test only', 'Half: refit', 'Remove: test only', 'Remove: refit'], rotation=25, ha='right')
        ax.set_yticks(range(3), MODES)
        ax.set_title('Identity-held folds' if split == 'identity' else 'Three held-port pilots')
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f'{values[i,j]:+.3f}', ha='center', va='center', color='white' if abs(values[i,j]) > 2.4 else 'black')
        fig.colorbar(im, ax=ax, label='BA change (pp)', shrink=.7)
    fig.suptitle('Source-derived scene directions: feature intervention versus coordinated readout fitting\nHalf shrink is invertible; recovery does not establish semantic disentanglement', fontsize=11)
    fig.savefig(OUT/'training_alignment.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    tasks = ['heldport_Santos', 'heldport_Port_Klang', 'heldport_Los_Angeles']
    methods = ['isotropic_regularized', 'scene_consistency', 'semantic_protected_consistency']
    colors = ['#777777', '#19776b', '#bf527a']
    labels = ['Ordinary regularization', 'Scene-score consistency', 'Centroid-protected consistency']
    for ax, field in zip(axes, ['delta_BA', 'delta_OA']):
        for j, (method, color, label) in enumerate(zip(methods, colors, labels)):
            values = r[r['mode'].eq('dual') & r.arm.eq(method)].set_index('task').loc[tasks, field].to_numpy()
            xx = np.arange(3)+(j-1)*.24
            ax.bar(xx, values, width=.23, color=color, label=label)
            for pos, val in zip(xx, values):
                ax.annotate(f'{val:+.2f}', (pos, val), xytext=(0, 4 if val >= 0 else -12),
                            textcoords='offset points', ha='center', fontsize=8)
        ax.axhline(0, color='black', linewidth=.6)
        ax.set_xticks(range(3), ['Santos', 'Port Klang', 'Los Angeles'])
        ax.set_ylabel(('BA' if field == 'delta_BA' else 'OA')+' change (pp)')
        ax.margins(y=.22)
    axes[0].legend(fontsize=8)
    fig.suptitle('Non-subtractive training: unchanged raw features at inference\nThree metadata-selected ports; all fits source-only; frozen-encoder development pilot', fontsize=11)
    fig.savefig(OUT/'non_subtractive_results.png', dpi=170)
    plt.close(fig)
    print(r[r.split.eq('held_port') & r['mode'].eq('dual') & r.arm.isin(methods)][
        ['task', 'arm', 'base_BA', 'BA', 'delta_BA', 'base_OA', 'OA', 'delta_OA']].round(4).to_string(index=False))


if __name__ == '__main__':
    run()
