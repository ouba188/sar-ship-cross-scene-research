"""Posthoc fixed-terminal B diagnostic; no new training or target selection."""
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import run_consistent as run


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    out=run.HERE/'analysis'/'terminal_B'; out.mkdir(exist_ok=True)
    metrics=[]
    for arm in run.ARMS:
        data=run.load(arm,True)
        for seed in range(3):
            saved=torch.load(run.b_path(arm,seed).with_name(f'B_seed{seed}_last.pt'),map_location='cuda',weights_only=False)
            assert saved['step']==200
            model=run.geo.GeometryReader('B').cuda().eval()
            model.load_state_dict(saved['state'])
            table,pred=run.geo.evaluate(model,data,'target',True)
            table['arm'],table['seed']=arm,seed
            metrics.append(table)
            pred.to_csv(out/f'{arm}_seed{seed}_predictions.csv',index=False)
    metrics=pd.concat(metrics,ignore_index=True)
    metrics.to_csv(out/'metrics.csv',index=False)
    met=['oa','ba','macro_nll','macro_brier']
    seed=metrics.groupby(['arm','seed','origin','acquisition'])[met].mean().groupby(['arm','seed','origin']).mean().reset_index()
    seed.to_csv(out/'seed_means.csv',index=False)
    p=seed[seed.arm.eq('fixed')].merge(seed[seed.arm.eq('wrong')],on=['seed','origin'],suffixes=('','_wrong'))
    for col in met:
        p['delta_'+col]=p[col]-p[col+'_wrong']
    p.to_csv(out/'paired_deltas.csv',index=False)
    print(p[['seed','origin']+['delta_'+c for c in met]].to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
