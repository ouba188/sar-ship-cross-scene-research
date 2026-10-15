"""Post-freeze selector capture and exact old-admission routing decomposition."""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from frequency_preserve_experiment import OUT, EXP, FAMILIES
from prepare_learned_frequency import ROOT, SOURCE, DATASETS, load_npz
from frequency_band_core import BANDS
from frequency_reliability_run import statistics


def main():
    if not (OUT/'retention_summary.csv').exists():
        raise ValueError('Complete frozen target reporting first')
    captures, routing, utilities = [], [], []
    for folder in [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]:
        info = json.loads((folder/'PREPARED.json').read_text())
        meta = dict(dataset=info['dataset'], task=info['task'])
        p = load_npz(OUT/'targets'/meta['dataset']/meta['task']/'policies.npz')
        d = load_npz(folder/'target.npz')
        manifest = SOURCE/'manifest.csv' if meta['dataset'] == 'self24ports' else ROOT/DATASETS[meta['dataset']][0]/'target_manifest.csv'
        labels = pd.read_csv(manifest, usecols=['class_id']).class_id.to_numpy()[p['rows']]
        modes = list(p['modes'])
        for draw, ev in enumerate(p['eval_masks']):
            y = labels[ev]
            main_pred = p['base_predictions'][draw, modes.index('learned'), ev]
            main_correct = main_pred == y
            bands = {f: p['base_predictions'][draw, [modes.index(f+'_'+b) for b in BANDS]][:, ev].T
                for f in ('standalone', 'preserved')}
            opportunity = {f: ~main_correct & (v == y[:, None]).any(1) for f, v in bands.items()}
            added = opportunity['preserved'] & ~opportunity['standalone']
            for f in ('standalone', 'preserved'):
                fi = FAMILIES.index(f)
                base = p['base_predictions'][draw, modes.index(f+'_support_local'), ev]
                choice = p['choices'][draw, fi, ev]
                correct = base == y
                captures.append(dict(**meta, draw=draw, family=f, opportunities=int(opportunity[f].sum()),
                    captured=int(np.sum(~main_correct & correct)), harms=int(np.sum(main_correct & ~correct)),
                    added_opportunities=int(added.sum()), captured_added=int(np.sum(added & correct)),
                    added_keep_main=int(np.sum(added & (choice == 0))),
                    added_wrong_band=int(np.sum(added & (choice != 0) & ~correct))))
                for bi, band in enumerate(BANDS):
                    rescue = ~main_correct & (bands[f][:, bi] == y)
                    harm = main_correct & (bands[f][:, bi] != y)
                    events = rescue | harm
                    value = p['benefit'][draw, fi, ev, bi]
                    utilities.append(dict(**meta, draw=draw, family=f, band=band,
                        auc=float(roc_auc_score(rescue[events], value[events])) if rescue.any() and harm.any() else np.nan,
                        nonzero=float(np.mean(value[events] != 0)) if events.any() else np.nan))
            for suffix in ('equal', 'support_local'):
                before, after = (modes.index(f+'_'+suffix) for f in ('standalone', 'preserved'))
                old_full = p['predictions'][draw, before, ev]
                new_base = p['base_predictions'][draw, after, ev]
                expert = d['expert'][draw, ev].argmax(1)
                fixed = np.where(p['selected'][draw, before, ev], expert, new_base)
                new_full = p['predictions'][draw, after, ev]
                old_stats, fixed_stats, new_stats = [statistics(y, pred)[:2] for pred in (old_full, fixed, new_full)]
                row = dict(**meta, draw=draw, suffix=suffix)
                for i, name in enumerate(('oa', 'ba')):
                    row['fixed_standalone_route_delta_'+name] = fixed_stats[i]-old_stats[i]
                    row['changed_route_delta_'+name] = new_stats[i]-fixed_stats[i]
                    row['full_delta_'+name] = new_stats[i]-old_stats[i]
                routing.append(row)
    for name, rows in (('capture', captures), ('route', routing), ('utility', utilities)):
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT/f'{name}_probe.csv', index=False)
        if name == 'capture':
            fields = ['opportunities', 'captured', 'harms', 'added_opportunities', 'captured_added', 'added_keep_main', 'added_wrong_band']
            summary = frame.groupby(['dataset', 'family'])[fields].sum()
        else:
            group = ['dataset', 'suffix'] if name == 'route' else ['dataset', 'family']
            fields = [c for c in frame if c.endswith(('_oa', '_ba'))] if name == 'route' else ['auc', 'nonzero']
            summary = frame.groupby([*group, 'task'])[fields].mean().groupby(group).mean()
        summary.to_csv(OUT/f'{name}_probe_summary.csv')
        print(name+'\n'+summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
