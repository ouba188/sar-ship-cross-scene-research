"""Post-hoc diagnostic: add the old A-trained signed output to calibrated B."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_increment import ref


def main():
    current = pd.read_csv(HERE/'predictions.csv')
    rows, per_case = [], []
    for part in ['new', 'old']:
        path = HERE.parent/('source_utility_alignment' if part == 'new' else 'old_partition_replication')/'predictions.csv'
        prior = pd.read_csv(path)
        prior = prior[prior.cell.eq('jointCE__J')] if part == 'new' else prior[prior.arm.eq('signed_only') & prior.selector.eq('J')]
        for seed in [17,29,43]:
            g = current[current.partition.eq(part) & current.arm.eq('signed_only') & current.seed.eq(seed)].set_index('instance_id').sort_index()
            old = prior[prior.seed.eq(seed)].set_index('instance_id').loc[g.index]
            np.testing.assert_array_equal(g.class_id, old.class_id)
            np.testing.assert_array_equal(g.fold, old.fold)
            b = g[['b0','b1','b2']].to_numpy()
            added = old[['delta0','delta1','delta2']].to_numpy()
            score = b+added
            y = g.class_id.to_numpy()
            m0 = ref.assessment(y,b)
            m = ref.assessment(y,score)
            net = (score.argmax(1) == y).astype(int)-(b.argmax(1) == y).astype(int)
            rows.append(dict(partition=part,seed=seed,net=int(net.sum()),**m,
                             delta_OA=m['OA']-m0['OA'],delta_BA=m['BA']-m0['BA']))
            h = g[['fold','class_id','port','pass_group','mmsi']].copy()
            h['partition'],h['seed'] = part,seed
            h['naive_prediction'],h['retrained_prediction'],h['base_prediction'] = score.argmax(1),g.prediction,b.argmax(1)
            h['naive_net'] = net
            per_case.append(h.reset_index())
    pd.DataFrame(rows).to_csv(HERE/'naive_relocation_metrics.csv',index=False)
    pd.concat(per_case,ignore_index=True).to_csv(HERE/'naive_relocation_cases.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
