"""Post-freeze accounting: evidence changes, recoverable opportunities and routing."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from prepare_learned_frequency import ROOT, SOURCE, DATASETS, load_npz
from frequency_standalone_experiment import OUT, FAMILIES
from frequency_band_core import BANDS


def main():
    if not (OUT/'REPORT_COMPLETE.json').exists():
        raise ValueError('Freeze and align all target outputs through the report first')
    source_y = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    changes, selectors, gradients, qualities, transitions = [], [], [], [], []
    for marker in sorted((OUT/'targets').glob('*/*/COMPLETE.json')):
        info = json.loads(marker.read_text())
        meta = dict(dataset=info['dataset'], task=info['task'])
        p = load_npz(marker.parent/'policies.npz')
        all_y = source_y if info['dataset'] == 'self24ports' else pd.read_csv(
            ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        y = all_y[p['rows']]
        modes = list(p['modes'])
        for draw, ev in enumerate(p['eval_masks']):
            truth = y[ev]
            ref = p['base_predictions'][draw, modes.index('learned'), ev]
            correct = ref == truth
            bands = {f: np.stack([p['base_predictions'][draw, modes.index(f+'_'+b), ev] for b in BANDS], 1) for f in FAMILIES}
            possible = {f: ~correct & (b == truth[:, None]).any(1) for f, b in bands.items()}
            for family in ('standalone', 'pair'):
                added = possible[family] & ~possible['independent']
                lost = possible['independent'] & ~possible[family]
                changes.append(dict(**meta, draw=draw, family=family, added=int(added.sum()), lost=int(lost.sum()),
                    delta_oracle_ba=float(np.mean([100*(added[truth == c].mean()-lost[truth == c].mean()) for c in np.unique(truth)]))))
                for bi, band in enumerate(BANDS):
                    before, after = bands['independent'][:, bi], bands[family][:, bi]
                    useful_change = (before == truth) != (after == truth)
                    codes = truth[useful_change]*64+before[useful_change]*8+after[useful_change]
                    code, count = np.unique(codes, return_counts=True)
                    for c, n in zip(code, count):
                        actual, prior, new = int(c//64), int(c//8 % 8), int(c % 8)
                        transitions.append(dict(**meta, draw=draw, family=family, band=band,
                            true_class=actual, before_class=prior, after_class=new,
                            event='rescue' if new == actual else 'harm', count=int(n)))
            for fi, family in enumerate(FAMILIES):
                pred = p['base_predictions'][draw, modes.index(family+'_support_local'), ev]
                choice = p['choices'][draw, fi, ev]
                rescue, harm = (~correct) & (pred == truth), correct & (pred != truth)
                selectors.append(dict(**meta, draw=draw, family=family, opportunities=int(possible[family].sum()),
                    captured=int(rescue.sum()), harms=int(harm.sum()),
                    missed_keep=int(np.sum(possible[family] & (choice == 0))),
                    missed_wrong=int(np.sum(possible[family] & (choice != 0) & ~rescue))))
                for bi, band in enumerate(BANDS):
                    r = ~correct & (bands[family][:, bi] == truth)
                    h = correct & (bands[family][:, bi] != truth)
                    action, value = r | h, p['benefit'][draw, fi, ev, bi]
                    qualities.append(dict(**meta, draw=draw, family=family, band=band,
                        auc=float(roc_auc_score(r[action], value[action])) if r.any() and h.any() else np.nan,
                        nonzero=float(np.mean(value[action] != 0)) if action.any() else np.nan))
        for family in ('standalone', 'pair'):
            log = json.loads((OUT/'tasks'/meta['dataset']/meta['task']/f'{family}_training.json').read_text())
            g = log['auxiliary_first_gradient']
            gradients.append(dict(**meta, family=family, **g,
                auxiliary_to_base_norm=g['weighted_aux_norm']/g['base_norm'],
                auxiliary_first=log['epochs'][0]['auxiliary'], auxiliary_last=log['epochs'][-1]['auxiliary'],
                min_branch_gradient=min(log['first_branch_gradients']),
                min_branch_parameter_change=min(log['branch_parameter_l2_change'])))
    for name, rows in [('opportunities', changes), ('selectors', selectors), ('gradients', gradients), ('utility', qualities)]:
        df = pd.DataFrame(rows)
        df.to_csv(OUT/(name+'_diagnostics.csv'), index=False)
        if name in ('opportunities', 'selectors'):
            fields = ['added', 'lost'] if name == 'opportunities' else ['opportunities', 'captured', 'harms', 'missed_keep', 'missed_wrong']
            df.groupby(['dataset', 'family'])[fields].sum().to_csv(OUT/(name+'_summary.csv'))
        else:
            fields = ['auc', 'nonzero'] if name == 'utility' else ['auxiliary_to_base_norm', 'cosine', 'auxiliary_first', 'auxiliary_last']
            df.groupby(['dataset', 'task', 'family'])[fields].mean().groupby(['dataset', 'family']).mean().to_csv(OUT/(name+'_summary.csv'))
    pd.DataFrame(transitions).to_csv(OUT/'branch_class_transitions.csv', index=False)
    pd.DataFrame(transitions).groupby(['dataset', 'family', 'band', 'true_class', 'before_class', 'after_class', 'event'])['count'].sum().to_csv(OUT/'branch_class_transition_summary.csv')


if __name__ == '__main__':
    main()
