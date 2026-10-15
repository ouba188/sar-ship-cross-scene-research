"""Finite label-informed feasible candidates under fixed weight-ratio limits."""

import json
import numpy as np
import pandas as pd
from check_teacher_weight_strength import ROOT, SEEDS, load_bundle, evaluate
from check_translation_teacher_feasibility import margins

OUT = ROOT/'bounded_headroom'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'PROTOCOL.json').write_text(json.dumps(dict(new_fits=0,
        candidates='Equal; upweight one known donor to max ratio or downweight one while others get max ratio',
        ratio2=2, damped_ratio='1+actual direction quality span/(1+span), different per focal',
        boundary='True-label diagnostic of finite feasible candidates, not exhaustive optimum, achievable learned gain, or student gain. Unknown mass unchanged.')),
        encoding='utf-8')
    rows, per_ship = [], []
    for role in ('fit', 'held'):
        for seed in SEEDS:
            b = load_bundle(role, seed); cells = b['cells']; cs = b['values']['scores']
            old = b['predictions']['equal']; quality = b['values']['direction']
            pred = {k: old.copy() for k in ('ratio2_oracle', 'damped_ratio_oracle')}
            for _, group in cells.groupby('mmsi', sort=True):
                indexes = group.index.to_numpy()
                for focal in indexes:
                    donor = indexes[indexes != focal]
                    known = cells.loc[donor, 'known'].to_numpy(bool)
                    k = int(known.sum())
                    if k < 2:
                        continue
                    ids = b['members'][focal]
                    span = np.ptp(quality[donor[known]])
                    y = int(cells.loc[focal, 'class_id'])
                    for name, ratio in (('ratio2_oracle', 2.), ('damped_ratio_oracle', 1+span/(1+span))):
                        options = [old[ids[0]]]
                        for high_one in (True, False):
                            for j in range(k):
                                local = np.ones(k) if high_one else np.full(k, ratio)
                                local[j] = ratio if high_one else 1.
                                assert local.max()/local.min() <= ratio+1e-12
                                weight = np.ones(len(donor)); weight[known] = local/local.mean()
                                options.append(weight@cs[donor]/len(donor))
                        options = np.asarray(options)
                        pred[name][ids] = options[np.argmax(margins(options, np.full(len(options), y)))]
            for name, pp in pred.items():
                result = evaluate(b, pp, name, role, seed)
                assert result['harm'] == 0
                rows.append(result)
                mm = b['meta']; y = mm.class_id.to_numpy()
                rescue = b['predictions']['multiview'] & (old.argmax(1) != y) & (pp.argmax(1) == y)
                for (ship, cls), g in mm.loc[rescue].groupby(['mmsi', 'class_id']):
                    per_ship.append(dict(role=role, seed=seed, arm=name, mmsi=int(ship), class_id=int(cls), rescues=len(g)))
            rows.append(evaluate(b, old, 'equal', role, seed))
            np.savez_compressed(OUT/f'{role}_{seed}.npz', obs_ids=b['predictions']['obs_ids'], **pred)
    pd.DataFrame(rows).to_csv(OUT/'results.csv', index=False)
    pd.DataFrame(per_ship).to_csv(OUT/'rescue_ships.csv', index=False)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(completed=True, new_fits=0, no_oracle_harms=True)), encoding='utf-8')
    print(pd.DataFrame(rows)[['role', 'seed', 'arm', 'OA', 'BA', 'rescue', 'rescue_ships']].to_string(index=False))


if __name__ == '__main__':
    main()
