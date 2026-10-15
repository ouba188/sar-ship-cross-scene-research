"""Post-freeze branch diversity and support-evidence failure accounting."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from prepare_learned_frequency import ROOT, OUT as EXP, SOURCE, DATASETS, load_npz
from frequency_specialist_run import OUT, FAMILIES, BANDS


def main():
    markers = sorted((OUT/'targets').glob('*/*/COMPLETE.json'))
    if len(markers) != 25:
        raise ValueError('All25 target predictions must be frozen')
    source_y = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    shifts, events, quality, training, embeddings = [], [], [], [], []
    for marker in markers:
        info = json.loads(marker.read_text())
        meta = dict(dataset=info['dataset'], task=info['task'])
        p = load_npz(marker.parent/'policies.npz')
        y = source_y[p['rows']] if info['dataset'] == 'self24ports' else pd.read_csv(
            ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        modes = list(p['modes'])
        for draw, ev in enumerate(p['eval_masks']):
            truth = y[ev]
            reference = p['base_predictions'][draw, modes.index('learned'), ev]
            correct_ref = reference == truth
            band_preds = {f: np.stack([p['base_predictions'][draw, modes.index(f+'_'+b), ev] for b in BANDS], 1) for f in FAMILIES}
            possible = {f: (~correct_ref) & (pred == truth[:, None]).any(1) for f, pred in band_preds.items()}
            for family, prior in (('continued', 'old'), ('independent', 'continued'), ('specialist', 'independent')):
                added, lost = possible[family] & ~possible[prior], possible[prior] & ~possible[family]
                per_class = [100*(added[truth == c].mean()-lost[truth == c].mean()) for c in np.unique(truth)]
                shifts.append(dict(**meta, draw=draw, family=family, reference=prior,
                    added_opportunities=int(added.sum()), lost_opportunities=int(lost.sum()),
                    delta_oracle_ba=float(np.mean(per_class)),
                    any_band_changes=int(np.sum(np.any(band_preds[family] != band_preds[prior], axis=1)))))
            for fi, family in enumerate(FAMILIES):
                pred = p['base_predictions'][draw, modes.index(family+'_support_local'), ev]
                choices = p['choices'][draw, fi, ev]
                rescue, harm = (~correct_ref) & (pred == truth), correct_ref & (pred != truth)
                events.append(dict(**meta, draw=draw, family=family, opportunities=int(possible[family].sum()),
                    captured=int(rescue.sum()), harmed=int(harm.sum()), switches=int(np.sum(choices != 0)),
                    missed_keep_main=int(np.sum(possible[family] & (choices == 0))),
                    missed_wrong_band=int(np.sum(possible[family] & (choices != 0) & ~rescue))))
                for bi, band in enumerate(BANDS):
                    r = (~correct_ref) & (band_preds[family][:, bi] == truth)
                    h = correct_ref & (band_preds[family][:, bi] != truth)
                    values = p['benefit'][draw, fi, ev, bi]
                    action = r | h
                    quality.append(dict(**meta, draw=draw, family=family, band=band,
                        rescue_events=int(r.sum()), harm_events=int(h.sum()),
                        auc=float(roc_auc_score(r[action], values[action])) if r.any() and h.any() else np.nan,
                        nonzero_event_fraction=float(np.mean(values[action] != 0)) if action.any() else np.nan,
                        mean_on_rescue=float(values[r].mean()) if r.any() else np.nan,
                        mean_on_harm=float(values[h].mean()) if h.any() else np.nan))
        task = OUT/'tasks'/meta['dataset']/meta['task']
        for family in FAMILIES[1:]:
            train = json.loads((task/f'{family}_training.json').read_text())
            for b, band in enumerate(BANDS):
                training.append(dict(**meta, family=family, band=band,
                    first_gradient=train['first_branch_gradients'][b],
                    parameter_change=train['branch_parameter_l2_change'][b],
                    allocation_mean=train['allocation_mean'][b],
                    allocation_min=train['allocation_min'], allocation_max=train['allocation_max'],
                    initial_ce=train['epochs'][0]['ce'], final_ce=train['epochs'][-1]['ce'],
                    final_protection=train['epochs'][-1]['protection']))
        independent = load_npz(task/'independent_embedding.npz')['embedding'].reshape(-1, 4, 64).astype(float)*2
        specialist = load_npz(task/'specialist_embedding.npz')['embedding'].reshape(-1, 4, 64).astype(float)*2
        cosine = np.sum(independent*specialist, 2)/(np.linalg.norm(independent, axis=2)*np.linalg.norm(specialist, axis=2))
        for b, band in enumerate(BANDS):
            embeddings.append(dict(**meta, band=band, specialist_independent_cosine=float(cosine[:, b].mean())))
    for name, data in [('opportunity_changes', shifts), ('support_selector_events', events),
                       ('support_utility', quality), ('training_diagnostics', training), ('embedding_diagnostics', embeddings)]:
        pd.DataFrame(data).to_csv(OUT/(name+'.csv'), index=False)
    change = pd.DataFrame(shifts)
    change.groupby(['dataset', 'family', 'reference'])[['added_opportunities', 'lost_opportunities', 'any_band_changes']].sum().to_csv(OUT/'opportunity_event_summary.csv')
    change.groupby(['dataset', 'task', 'family', 'reference']).delta_oracle_ba.mean().groupby(
        ['dataset', 'family', 'reference']).mean().to_csv(OUT/'opportunity_ba_summary.csv')
    pd.DataFrame(events).groupby(['dataset', 'family'])[['opportunities', 'captured', 'harmed', 'switches',
        'missed_keep_main', 'missed_wrong_band']].sum().to_csv(OUT/'support_selector_summary.csv')
    pd.DataFrame(quality).groupby(['dataset', 'task', 'family', 'band'])[['auc', 'nonzero_event_fraction',
        'mean_on_rescue', 'mean_on_harm']].mean().groupby(['dataset', 'family', 'band']).mean().to_csv(OUT/'support_utility_summary.csv')
    print(change.groupby(['dataset', 'family'])[['added_opportunities', 'lost_opportunities']].sum().to_string(), flush=True)


if __name__ == '__main__':
    main()
