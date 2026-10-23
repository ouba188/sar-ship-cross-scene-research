"""Fixed-head source-mean replacement: test query-dependent coordination."""
from pathlib import Path
import numpy as np
import pandas as pd
from run_router import STRATEGIES, blend, metrics

HERE = Path(__file__).resolve().parent


def main():
    p = pd.read_csv(HERE/'router_results_constantfix/predictions.csv')
    table = pd.read_csv(HERE/'upstream/table.csv')
    rows = []
    for seed in (17,29,43):
        with np.load(HERE/'upstream'/f'strategies_seed{seed}.npz') as z:
            scores = z['logits'].astype(float)
        for arm in ('ce_only','utility'):
            train = p[(p.seed==seed)&p.arm.eq(arm)&p.role.eq('utility_train')]
            w = train[[f'w_{k}_{c}' for c in range(8) for k in STRATEGIES]].to_numpy().reshape(-1,8,3)
            replacements = {'global_train_mean':w.mean((0,1)), 'class_train_mean':w.mean(0)}
            for role in ('selection','evaluation'):
                take = table.role.eq(role)
                y = table.loc[take,'class_id'].to_numpy(int)
                actual = p[(p.seed==seed)&p.arm.eq(arm)&p.role.eq(role)].set_index('row_index').loc[table.loc[take,'row_index']]
                a = actual.prediction.to_numpy()==y
                for name,mean in replacements.items():
                    new = blend(scores[take],np.broadcast_to(mean,(int(take.sum()),8,3)))
                    b = new.argmax(1)==y
                    rows.append(dict(seed=seed,arm=arm,role=role,replacement=name,
                        **metrics(y,new),actual_OA=float(a.mean()*100),
                        actual_rescue=int(np.sum(a&~b)),actual_harm=int(np.sum(~a&b)),
                        actual_delta_BA=float(np.mean([(a[y==c].mean()-b[y==c].mean()) for c in np.unique(y)])*100),
                        weights=mean.tolist()))
    f=pd.DataFrame(rows)
    f.to_csv(HERE/'analysis_constantfix/fixed_weight_replacement.csv',index=False)
    print(f[f.role.eq('evaluation')].drop(columns='weights').to_string(index=False))


if __name__=='__main__':main()
