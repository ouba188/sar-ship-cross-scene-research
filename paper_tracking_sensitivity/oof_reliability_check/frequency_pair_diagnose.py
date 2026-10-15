"""Post-freeze coverage and matched decision-change diagnosis."""

from frequency_pair_run import OUT, EXP, MODES
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from prepare_learned_frequency import ROOT, SOURCE, DATASETS, load_npz


def main():
    if len(list((OUT/'targets').glob('*/*/COMPLETE.json'))) != 25:
        raise ValueError('Freeze all predictions first')
    intervention, ranking, transitions = [], [], []
    for folder in [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]:
        info = json.loads((folder/'PREPARED.json').read_text())
        meta = dict(dataset=info['dataset'], task=info['task'])
        p = load_npz(OUT/'targets'/meta['dataset']/meta['task']/'policies.npz')
        original = load_npz(folder/'target.npz')
        manifest = SOURCE/'manifest.csv' if meta['dataset'] == 'self24ports' else ROOT/DATASETS[meta['dataset']][0]/'target_manifest.csv'
        idcol = 'instance_id' if meta['dataset'] == 'self24ports' else 'sample_id'
        aligned = pd.read_csv(manifest, usecols=[idcol, 'class_id'], keep_default_na=False).iloc[p['rows']]
        np.testing.assert_array_equal(aligned[idcol].to_numpy(str), original['ids'])
        y_all = aligned.class_id.to_numpy()
        for draw, ev in enumerate(p['eval_masks']):
            y = y_all[ev]
            main_pred = p['base_predictions'][draw, MODES.index('learned'), ev]
            band_pred, rho = p['candidates'][draw, ev], p['coverage'][draw, ev]
            active = rho > 0
            _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
            ba_weight = 100/(len(counts)*counts[inverse])
            baseline = p['base_predictions'][draw, MODES.index('smooth_parent'), ev]
            oldchoice = p['choices'][draw, MODES.index('smooth_parent'), ev]
            for mode in ('class_local', 'pair_local'):
                pred = p['base_predictions'][draw, MODES.index(mode), ev]
                choice = p['choices'][draw, MODES.index(mode), ev]
                np.testing.assert_array_equal(choice[~active.any(1)], oldchoice[~active.any(1)])
                delta = (pred == y).astype(float)-(baseline == y)
                for group, mask in dict(extra_switch=(oldchoice == 0) & (choice > 0),
                    stop_switch=(oldchoice > 0) & (choice == 0),
                    different_band=(oldchoice > 0) & (choice > 0) & (oldchoice != choice),
                    same_choice=(oldchoice == choice)).items():
                    transitions.append(dict(**meta, draw=draw, mode=mode, group=group, n=int(mask.sum()),
                        rescues=int(np.sum(mask & (delta > 0))), harms=int(np.sum(mask & (delta < 0))),
                        delta_oa=100*float(delta[mask].sum())/len(y),
                        delta_ba=float(np.sum(delta[mask]*ba_weight[mask]))))
            for bi in range(4):
                rescue = (main_pred != y) & (band_pred[:, bi] == y)
                harm = (main_pred == y) & (band_pred[:, bi] != y)
                conflict = main_pred != band_pred[:, bi]
                for event, mask in dict(conflict=conflict, rescue=rescue, harm=harm).items():
                    n, changed = int(mask.sum()), int(np.sum(mask & active[:, bi]))
                    intervention.append(dict(**meta, draw=draw, band=bi, event=event,
                        n=n, active_n=changed, fraction=changed/n if n else np.nan,
                        mean_rho=float(rho[mask, bi].mean()) if n else np.nan,
                        mean_active_rho=float(rho[mask & active[:, bi], bi].mean()) if changed else np.nan))
                events = rescue | harm
                for subset, mask in dict(active=events & active[:, bi], inactive=events & ~active[:, bi]).items():
                    for ui, name in enumerate(('smooth', 'class_local', 'pair_local')):
                        value = p['utilities'][draw, ui, ev, bi]
                        defined = bool((rescue & mask).any() and (harm & mask).any())
                        ranking.append(dict(**meta, draw=draw, band=bi, subset=subset, utility=name,
                            auc=float(roc_auc_score(rescue[mask], value[mask])) if defined else np.nan,
                            n=int(mask.sum()), defined=int(defined)))
        print(f'DIAGNOSE {meta}', flush=True)
    for name, rows, group, values, numbers in (
        ('intervention', intervention, ['event'], ['fraction', 'mean_rho', 'mean_active_rho'], ['n', 'active_n']),
        ('ranking_subsets', ranking, ['subset', 'utility'], ['auc'], ['n', 'defined']),
        ('transitions', transitions, ['mode', 'group'], ['delta_oa', 'delta_ba'], ['n', 'rescues', 'harms'])):
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT/(name+'.csv'), index=False)
        means = frame.groupby(['dataset', 'task', *group])[values].mean().groupby(['dataset', *group]).mean()
        sums = frame.groupby(['dataset', *group])[numbers].sum()
        means.join(sums).to_csv(OUT/(name+'_summary.csv'))
        print(name+'\n'+means.join(sums).to_string(), flush=True)


if __name__ == '__main__':
    main()
