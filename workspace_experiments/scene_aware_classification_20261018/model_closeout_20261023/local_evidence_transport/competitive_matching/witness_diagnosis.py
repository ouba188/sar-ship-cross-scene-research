"""Label-assisted diagnosis of candidate-specific witness proliferation.

Labels are used only to count possible explanations, never for a deployed score.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
UP = HERE.parent.parent / 'regional_utility_pilot/upstream'


def main():
    z = np.load(HERE / 'scores.npz')
    table = pd.read_csv(UP / 'table.csv').iloc[z['idx']].reset_index(drop=True)
    y = table.class_id.to_numpy()
    outputs = []
    for role in ('utility_train', 'selection', 'evaluation'):
        take = np.flatnonzero(table.role.eq(role))
        for p, pol in enumerate(('VV', 'VH')):
            for r, region in enumerate(('body', 'side', 'end')):
                fully = z['available'][take, :, p, r].all(-1)
                ids = take[fully]
                f = z['fixed'][ids, :, p, r]
                c = z['coupled'][ids, :, p, r]
                truth = y[ids]
                true_f = f[np.arange(len(ids)), truth]
                true_c = c[np.arange(len(ids)), truth]
                count_f = (f > 1e-5).sum(-1)
                count_c = (c > 1e-5).sum(-1)
                newly_feasible = (true_f <= 1e-5) & (true_c > 1e-5)
                outputs.append(dict(role=role, pol=pol, region=region, n=len(ids),
                                    fixed_multiple_positive=int((count_f > 1).sum()),
                                    coupled_multiple_positive=int((count_c > 1).sum()),
                                    fixed_mean_positive_candidates=float(count_f.mean()),
                                    coupled_mean_positive_candidates=float(count_c.mean()),
                                    true_fixed_positive=int((true_f > 1e-5).sum()),
                                    true_coupled_positive=int((true_c > 1e-5).sum()),
                                    true_new_positive=int(newly_feasible.sum()),
                                    true_new_positive_but_loses=int((newly_feasible & (c.argmax(-1) != truth)).sum()),
                                    true_fixed_wins_but_coupled_loses=int(((f.argmax(-1) == truth) & (c.argmax(-1) != truth)).sum()),
                                    fixed_top1=int((f.argmax(-1) == truth).sum()),
                                    coupled_top1=int((c.argmax(-1) == truth).sum())))
    pd.DataFrame(outputs).to_csv(HERE / 'witness_diagnosis.csv', index=False)
    print(pd.DataFrame(outputs).query("role == 'evaluation'").to_string(index=False))


if __name__ == '__main__':
    main()
