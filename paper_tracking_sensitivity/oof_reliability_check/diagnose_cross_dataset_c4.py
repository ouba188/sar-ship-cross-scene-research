"""Separate support adaptation from source-head refit; fixed diagnostic only."""

import json
import pickle
import numpy as np
import pandas as pd

from cross_dataset_c4 import ROOT, OUT, DATASETS, CLASSES, base_coordinates, base_phi, route
from residual_expert import fit_correction
from budgeted_residual_handoff import score_policy


def main():
    records = []
    class_records = []
    for name, (target_dir, source_dir, joint_dir) in DATASETS.items():
        out = OUT/name/'support_refit_diagnostic'
        out.mkdir(exist_ok=True)
        frame = pd.read_csv(ROOT/target_dir/'target_manifest.csv', keep_default_na=False)
        ids, y = frame.target_manifest_row.to_numpy(int), frame.class_id.to_numpy(int)
        provenance = json.loads((OUT/name/'FROZEN.json').read_text())
        model_path = provenance.get('model_path', str(ROOT/source_dir/'source_model.pkl'))
        base_path = provenance.get('base_policy_path', str(ROOT/target_dir/'policies.npz'))
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with np.load(base_path) as saved:
            support, masks = saved['support_rows'], saved['eval_masks']
            ref_base = saved['base_scores'].argmax(2)
            expert = saved['expert_scores']
        if joint_dir:
            with np.load(ROOT/joint_dir/'policies.npz') as saved:
                expert = saved['expert_scores']
        if not (out/'FROZEN.json').exists():
            x = base_coordinates(np.load(OUT/name/'pooled_c4.npy', mmap_mode='r'), model)
            prior = model['base'].decision_function(x).astype(np.float32).astype(float)
            phi = base_phi(x, model)
            bases, fulls, selections = [], [], []
            for draw, s in enumerate(support):
                coef = fit_correction(phi[s], y[s], prior[s])
                score = prior+phi@coef
                pred, sel = route(score, expert[draw], model['gate'], y[s], masks[draw], ids)
                bases.append(score.argmax(1))
                fulls.append(pred)
                selections.append(sel)
            np.savez_compressed(out/'policies.npz', base_predictions=np.stack(bases), predictions=np.stack(fulls),
                                selected=np.stack(selections), support_rows=support, eval_masks=masks)
            (out/'FROZEN.json').write_text(json.dumps(dict(
                arm='c4_support_refit', source_head='original unchanged',
                support_and_query='C4 features; refit support64 ridge8 only',
                diagnostic_added_after_initial_readout=True,
                model_selection=False, reason='Separate support refit from source-head refit',
                no_new_features_or_hyperparameters=True), indent=2), encoding='utf-8')
        with np.load(out/'policies.npz') as saved:
            for d, ev in enumerate(masks):
                for branch, key in [('base', 'base_predictions'), ('full', 'predictions')]:
                    m = score_policy(saved[key][d], ref_base[d], y, ev)
                    records.append(dict(dataset=name, draw=d, arm='c4_support_refit', branch=branch,
                                        oa=100*m['acc'], ba=100*m['ba']))
                    for c, recall in enumerate(m['class_recall']):
                        if recall is not None:
                            class_records.append(dict(dataset=name, draw=d, branch=branch,
                                class_id=c, class_name=CLASSES[c], recall=100*recall,
                                n=int(np.sum(ev & (y == c)))))
        print('DIAGNOSED '+name, flush=True)
    df = pd.DataFrame(records)
    df.to_csv(OUT/'support_refit_diagnostic_metrics.csv', index=False)
    pd.DataFrame(class_records).to_csv(OUT/'support_refit_diagnostic_classes.csv', index=False)
    existing = pd.read_csv(OUT/'metrics.csv')
    wanted = existing[existing.arm.isin(['reference', 'c4_query_only', 'c4_matched'])]
    summary = pd.concat([wanted[df.columns], df]).groupby(['dataset', 'branch', 'arm'], sort=False)[['oa', 'ba']].mean()
    summary.to_csv(OUT/'support_refit_diagnostic_summary.csv')
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
