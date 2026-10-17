"""Posthoc fixed-weight attribution: retain new eligibility, mask restored radar."""
import numpy as np
import pandas as pd
import torch
import run_effect as run
from bounded_core import OptionalGeometryReader


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    assert (run.HERE/'REPLAY_CHECK.json').exists()
    data=run.load('Singapore',True)
    folder=run.HERE/'diagnostics'
    records=[]; predictions=[]; checks=[]
    for seed in range(3):
        run.prepare(data,seed)
        run.set_input(data,'strict','new')
        model=OptionalGeometryReader('geometry').to(run.base.DEVICE)
        reference=run.model_from(run.HERE,'Singapore','strict',seed)
        model.load_state_dict(reference.state_dict()); model.eval()
        full,pp=run.fit.evaluate(model,data,'target',True)
        actual=pd.read_csv(run.HERE/'evaluation/Singapore'/f'strict_new_seed{seed}_predictions.csv')
        keys=['task','global_row']; cols=[f'p{i}' for i in range(8)]
        m=pp.merge(actual[keys+cols],on=keys,suffixes=('','_saved'),validate='one_to_one')
        error=float(np.abs(m[cols].to_numpy()-m[[c+'_saved' for c in cols]].to_numpy()).max())
        assert error<1e-10
        restored=(data['geometries']['old'][:,4]==0)&(data['geometries']['new'][:,4]>0)
        masked=data['geometries']['new'].clone()
        masked[restored,1]=0; masked[restored,4]=0
        data['tensors']['geometry']=masked
        # Eligibility is deliberately held at the restored full-geometry gate.
        metrics,pred=run.fit.evaluate(model,data,'target',True)
        for frame in (metrics,pred):
            frame['arm']='strict_coverage_without_new_radar'; frame['seed']=seed
        records.append(metrics); predictions.append(pred)
        checks.append(dict(seed=seed,full_input_replay_error=error,
                           gate_unchanged=bool(torch.equal(data['tensors']['valid'],run.eligibility(data['geometries']['new'],'strict')))))
    pd.concat(records).to_csv(folder/'masked_radar_metrics.csv',index=False)
    pd.concat(predictions).to_csv(folder/'masked_radar_predictions.csv',index=False)
    run.fit.dump(folder/'masked_radar_check.json',dict(posthoc_attribution=True,new_training=False,
                   model_selection=False,scope='Singapore fixed target tasks, all three seeds',checks=checks))
    frame=pd.concat(records)
    print(frame.loc[frame.scope=='full'].groupby(['origin','seed'])[['oa','ba','macro_nll','macro_brier']].mean().to_string())


if __name__=='__main__':
    main()
