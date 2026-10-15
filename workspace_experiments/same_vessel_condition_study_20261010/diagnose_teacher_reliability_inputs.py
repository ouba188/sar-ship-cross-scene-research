"""Separate removing confidence from refitting knowledge coefficients."""

import json
import numpy as np
import pandas as pd
from check_teacher_weight_strength import ROOT, CACHE, SEEDS, load_bundle, apply_quality, evaluate
from check_motion_teacher_feasibility import assess

OUT = ROOT/'input_probe'


def main():
    rows, remaining, events, coefficients = [], [], [], []
    for role in ('fit', 'held'):
        for seed in SEEDS:
            b = load_bundle(role, seed); cells = b['cells']; x = b['values']['x']
            q = b['values']['direction'].copy()
            for k in (range(3) if role == 'fit' else [-1]):
                name = f'fold{k}' if k >= 0 else 'full'
                use = cells.fold.to_numpy() == k if k >= 0 else np.ones(len(cells), bool)
                with np.load(CACHE/f'{name}_direction_{seed}.npz') as f:
                    beta = f['beta']
                q[use] -= x[use, 0]*beta[0]
                coefficients.append(dict(role=role, seed=seed, fold=k, confidence=float(beta[0]),
                    speed=float(beta[1]), speed_square=float(beta[2]),
                    direction_cos=float(beta[3]), direction_sin=float(beta[4])))
            removed, _ = apply_quality(b, q, 'minmax')
            with np.load(OUT/f'{role}_{seed}_predictions.npz') as f:
                np.testing.assert_array_equal(f['obs_ids'], b['predictions']['obs_ids'])
                knowledge = f['knowledge_minmax']
            predictions = dict(equal=b['predictions']['equal'], old_direction=b['predictions']['direction_soft'],
                               old_confidence_removed=removed, refit_knowledge=knowledge)
            for name, pred in predictions.items():
                rows.append(evaluate(b, pred, name, role, seed))
                if role == 'held':
                    mask = b['predictions']['multiview'] & (b['meta'].mmsi.to_numpy() != 563050500)
                    remaining.append(dict(seed=seed, arm=name, n=int(mask.sum()),
                        **assess(pred[mask], b['meta'].loc[mask].class_id.to_numpy(), b['meta'].loc[mask])))
            if role == 'held':
                y = b['meta'].class_id.to_numpy()
                old = predictions['old_direction'].argmax(1)
                for name in ('old_confidence_removed', 'refit_knowledge'):
                    now = predictions[name].argmax(1)
                    for i in np.flatnonzero(b['predictions']['multiview'] & (old != now)):
                        events.append(dict(seed=seed, arm=name, obs_id=int(b['predictions']['obs_ids'][i]),
                            mmsi=int(b['meta'].iloc[i].mmsi), class_id=int(y[i]), before=int(old[i]), after=int(now[i]),
                            rescue=bool(old[i] != y[i] and now[i] == y[i]),
                            harm=bool(old[i] == y[i] and now[i] != y[i])))
    pd.DataFrame(rows).to_csv(OUT/'confidence_removal.csv', index=False)
    pd.DataFrame(remaining).to_csv(OUT/'leave_previous_rescue_ship_out.csv', index=False)
    pd.DataFrame(events).to_csv(OUT/'confidence_removal_events.csv', index=False)
    pd.DataFrame(coefficients).to_csv(OUT/'old_direction_coefficients.csv', index=False)
    (OUT/'DIAGNOSIS_COMPLETE.json').write_text(json.dumps(dict(new_fits=0,
        intervention='Remove confidence at frozen old coefficients; compare to separately refitted knowledge-only reader.')),
        encoding='utf-8')
    print(pd.DataFrame(rows)[['role', 'seed', 'arm', 'OA', 'BA', 'rescue', 'harm']].to_string(index=False))
    print(pd.DataFrame(remaining)[['seed', 'arm', 'OA', 'BA']].to_string(index=False))


if __name__ == '__main__':
    main()
