"""Measure the selected gate's objective change; do not fit another model."""
import numpy as np
import pandas as pd
import torch
import run as exp


def main():
    f,splits=exp.old.relation.load_data();rows=[]
    for fold,(tr,val,test) in enumerate(splits):
        assignment=np.load(exp.PREV/f'oof_f{fold}.npz')['inner']
        for inner in range(3):
            _,cal,_=exp.source_pools(f,tr,assignment,fold,inner)
            for seed in range(3):
                z=np.load(exp.ROOT/f'inputs_f{fold}_k{inner}_s{seed}.npz')
                state=torch.load(exp.ROOT/f'head_f{fold}_k{inner}_calibration_score_s{seed}.pt',map_location='cpu',weights_only=False)
                gate=exp.held.Gate().eval();gate.load_state_dict(state['state'])
                for part,ii in (('calibration',cal),('validation',val),('test_posthoc',test)):
                    a,d=z['base'][ii],z['residual'][ii];y=f.iloc[ii].class_id.to_numpy()
                    with torch.no_grad():w=gate(torch.tensor(a),torch.tensor(d)).numpy()
                    before=a+d;after=a+w[:,None]*d;target=2*np.eye(8)[y]-1
                    sw=exp.old.prior.weights(f,ii)
                    lb=float(np.average(np.mean((before-target)**2,1),weights=sw))
                    la=float(np.average(np.mean((after-target)**2,1),weights=sw))
                    rows.append(dict(fold=fold,inner=inner,seed=seed,partition=part,n=len(ii),step=state['step'],
                        active=bool(np.abs(d).max()>1e-8),before_mse=lb,after_mse=la,mse_gain=lb-la,
                        net_correct=int(((after.argmax(1)==y).astype(int)-(before.argmax(1)==y)).sum())))
    table=pd.DataFrame(rows);table.to_csv(exp.ROOT/'loss_diagnostic.csv',index=False)
    print(table[table.step.gt(0)].to_string(index=False))


if __name__=='__main__':main()
