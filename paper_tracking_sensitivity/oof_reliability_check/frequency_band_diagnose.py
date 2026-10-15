"""Post-freeze benefit calibration: distinguish available rescues from selection."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prepare_learned_frequency import ROOT, OUT as EXP, SOURCE, DATASETS, load_npz
from frequency_band_core import BANDS, predict_benefit
from frequency_band_run import OUT, DYNAMIC, fit_choice, predict_choice


def event_record(labels, reference, alternative, benefit):
    utility = (alternative == labels).astype(int)-(reference == labels)
    event = utility != 0
    rescue, harm = utility > 0, utility < 0
    return dict(rescues=int(rescue.sum()), harms=int(harm.sum()),
        event_auc=float(roc_auc_score(rescue[event], benefit[event])) if rescue.any() and harm.any() else np.nan,
        predicted_on_rescue=float(benefit[rescue].mean()) if rescue.any() else np.nan,
        predicted_on_harm=float(benefit[harm].mean()) if harm.any() else np.nan,
        rescue_positive=int(np.sum(rescue & (benefit > 0))),
        harm_positive=int(np.sum(harm & (benefit > 0))))


def main():
    markers = sorted((OUT/'targets').glob('*/*/COMPLETE.json'))
    if len(markers) != 25:
        raise ValueError('All25 target predictions must be frozen before diagnosis')
    source_y = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    rows, choice_rows, count_rows = [], [], []
    for marker in markers:
        meta = json.loads(marker.read_text())
        dataset, task = meta['dataset'], meta['task']
        p = load_npz(marker.parent/'policies.npz')
        y = source_y[p['rows']] if dataset == 'self24ports' else pd.read_csv(
            ROOT/DATASETS[dataset][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        src = OUT/'source'/dataset/task
        s = load_npz(src/'examples.npz')
        selection = json.loads((src/'SELECTION.json').read_text())
        with (src/'models.pkl').open('rb') as handle:
            models = pickle.load(handle)
        for j, mode in enumerate(DYNAMIC):
            width = selection['widths'][mode]
            for b, band in enumerate(BANDS):
                reference, alternative = s['scores'][:, 0].argmax(1), s['scores'][:, b+2].argmax(1)
                values = predict_benefit(models[mode], s['features'][:, b, :width])
                rows.append(dict(dataset=dataset, task=task, scope='source_fit', draw=-1, mode=mode, band=band,
                    **event_record(s['labels'], reference, alternative, values)))
        if dataset in ('f0921', 'oss2'):
            prior_oof = load_npz(src/'oof.npz')
            oof_values = np.zeros((len(s['labels']), 2, 4))
            for held in np.unique(s['episode']):
                valid = s['episode'] == held
                for j, mode in enumerate(DYNAMIC):
                    width = selection['widths'][mode]
                    model = fit_choice(s, ~valid, width)
                    score, _, values = predict_choice(model, s['scores'][valid], s['features'][valid], width)
                    np.testing.assert_array_equal(score.argmax(1), prior_oof['predictions'][valid, j])
                    oof_values[valid, j] = values
            for j, mode in enumerate(DYNAMIC):
                for b, band in enumerate(BANDS):
                    rows.append(dict(dataset=dataset, task=task, scope='source_oof', draw=-1, mode=mode, band=band,
                        **event_record(s['labels'], s['scores'][:, 0].argmax(1),
                            s['scores'][:, b+2].argmax(1), oof_values[:, j, b])))
            np.savez_compressed(src/'oof_benefit_diagnostic.npz', benefit=oof_values,
                labels=s['labels'], episode=s['episode'])
        mode_names = list(p['modes'])
        for draw, ev in enumerate(p['eval_masks']):
            reference = p['base_predictions'][draw, 0, ev]
            alternatives = np.stack([p['base_predictions'][draw, mode_names.index('fixed_'+b), ev] for b in BANDS], 1)
            opportunities = (reference != y[ev]) & (alternatives == y[ev, None]).any(1)
            for j, mode in enumerate(DYNAMIC):
                for b, band in enumerate(BANDS):
                    rows.append(dict(dataset=dataset, task=task, scope='target', draw=draw, mode=mode, band=band,
                        **event_record(y[ev], reference, alternatives[:, b], p['benefit'][draw, j, ev, b])))
                chosen = p['base_predictions'][draw, mode_names.index(mode), ev]
                indices = p['choices'][draw, j, ev]
                rescue = (reference != y[ev]) & (chosen == y[ev])
                harm = (reference == y[ev]) & (chosen != y[ev])
                kept = opportunities & (indices == 0)
                wrong_band = opportunities & (indices != 0) & ~rescue
                choice_rows.append(dict(dataset=dataset, task=task, draw=draw, mode=mode,
                    opportunities=int(opportunities.sum()), captured=int(rescue.sum()), harms=int(harm.sum()),
                    missed_keep_main=int(kept.sum()), missed_choose_wrong_band=int(wrong_band.sum()),
                    switches=int(np.sum(indices != 0))))
                for b in (0, 2, 3, 4, 5):
                    mask = indices == b
                    count_rows.append(dict(dataset=dataset, task=task, draw=draw, mode=mode,
                        band='keep_main' if b == 0 else BANDS[b-2], n=int(mask.sum()),
                        rescued=int(np.sum(mask & rescue)), harmed=int(np.sum(mask & harm))))
    details = pd.DataFrame(rows)
    details.to_csv(OUT/'benefit_diagnostics.csv', index=False)
    fields = ['event_auc', 'predicted_on_rescue', 'predicted_on_harm']
    details.groupby(['dataset', 'task', 'scope', 'mode', 'band'])[fields].mean().groupby(
        ['dataset', 'scope', 'mode', 'band']).mean().to_csv(OUT/'benefit_summary.csv')
    choice = pd.DataFrame(choice_rows)
    choice.to_csv(OUT/'selection_diagnosis.csv', index=False)
    sums = choice.groupby(['dataset', 'mode'])[['opportunities', 'captured', 'harms',
        'missed_keep_main', 'missed_choose_wrong_band', 'switches']].sum()
    sums['capture_percent'] = 100*sums.captured/sums.opportunities
    sums.to_csv(OUT/'selection_diagnosis_summary.csv')
    pd.DataFrame(count_rows).groupby(['dataset', 'mode', 'band'])[['n', 'rescued', 'harmed']].sum().to_csv(OUT/'chosen_band_events.csv')
    print(sums.to_string(), flush=True)


if __name__ == '__main__':
    main()
