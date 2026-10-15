"""Fixed-head-preference intervention; no new fitted policy or source selection."""

from frequency_calibration_run import OUT, EXP, MODES
import json
import numpy as np
import pandas as pd
from prepare_learned_frequency import ROOT, SOURCE, DATASETS, load_npz
from frequency_reliability_run import statistics


def main():
    if not (OUT/'REPORT_COMPLETE.json').exists():
        raise ValueError('Read out frozen trial first')
    rows, offsets = [], []
    for folder in [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]:
        info = json.loads((folder/'PREPARED.json').read_text())
        meta = dict(dataset=info['dataset'], task=info['task'])
        out = OUT/'targets'/meta['dataset']/meta['task']
        p, data = load_npz(out/'policies.npz'), load_npz(folder/'target.npz')
        fits = json.loads((out/'CALIBRATORS.json').read_text())
        rule = json.loads((OUT/'source'/meta['dataset']/meta['task']/'SELECTION.json').read_text())['rule']
        manifest = SOURCE/'manifest.csv' if meta['dataset'] == 'self24ports' else ROOT/DATASETS[meta['dataset']][0]/'target_manifest.csv'
        labels = pd.read_csv(manifest, usecols=['class_id']).class_id.to_numpy()[p['rows']]
        ci = MODES.index('calibrated')
        for draw, ev in enumerate(p['eval_masks']):
            fit = fits[draw]
            bias = np.zeros(4) if fit['constant'] else np.array(fit['coefficients'][2:])
            reference, candidates = p['base_predictions'][draw, 0, ev], p['candidates'][draw, ev]
            value = np.broadcast_to(bias, candidates.shape).copy()
            value[candidates == reference[:, None]] = -np.inf
            choice = np.c_[np.zeros(len(reference)), value].argmax(1)
            static = np.c_[reference, candidates][np.arange(len(reference)), choice]
            dynamic = p['base_predictions'][draw, ci, ev]
            y = labels[ev]
            scenarios = {'base': (dynamic, static)}
            expert, admission = data['expert'][draw, ev].argmax(1), p['selected'][draw, ci, ev]
            scenarios['frozen_calibrated_route'] = (p['predictions'][draw, ci, ev], np.where(admission, expert, static))
            scenarios['source_selected_fixed_route'] = (scenarios['frozen_calibrated_route'] if rule == 'calibrated'
                else (p['predictions'][draw, 0, ev], p['predictions'][draw, 0, ev]))
            for stage, (a, b) in scenarios.items():
                sa, sb = statistics(y, a), statistics(y, b)
                rows.append(dict(**meta, draw=draw, stage=stage, source_rule=rule,
                    dynamic_oa=sa[0], dynamic_ba=sa[1], static_oa=sb[0], static_ba=sb[1],
                    dynamic_minus_static_oa=sa[0]-sb[0], dynamic_minus_static_ba=sa[1]-sb[1],
                    changes=int(np.sum(a != b)), dynamic_rescues=int(np.sum((a == y) & (b != y))),
                    dynamic_harms=int(np.sum((a != y) & (b == y)))))
            offsets.append(dict(**meta, draw=draw, iterations=fit['iterations'],
                **{f'offset_{i}': float(v) for i, v in enumerate(bias)},
                dominant_offset=int(bias.argmax()),
                offset_only_switches=int(np.sum(choice != 0)), dynamic_switches=int(np.sum(p['choices'][draw, ci, ev] != 0))))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'fixed_preference_diagnosis.csv', index=False)
    fields = [c for c in frame if c.endswith(('_oa', '_ba'))]
    means = frame.groupby(['dataset', 'task', 'stage'])[fields].mean().groupby(['dataset', 'stage']).mean()
    counts = frame.groupby(['dataset', 'stage'])[['changes', 'dynamic_rescues', 'dynamic_harms']].sum()
    means.join(counts).to_csv(OUT/'fixed_preference_summary.csv')
    pd.DataFrame(offsets).to_csv(OUT/'calibrator_offsets.csv', index=False)
    print(means.join(counts).to_string(), flush=True)


if __name__ == '__main__':
    main()
