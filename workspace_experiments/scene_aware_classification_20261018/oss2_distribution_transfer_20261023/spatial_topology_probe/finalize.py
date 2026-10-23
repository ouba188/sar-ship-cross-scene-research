"""Saved-result replay and raw-case completion; no network or optimizer."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from prepare_coverage import graph,footprint

HERE=Path(__file__).resolve().parent


def main():
    checks=[]
    with np.load(HERE/'onepoint_replay.npz') as z:
        for key in z.files:
            if not key.endswith('_r'):continue
            prefix=key[:-2];r=z[key];weights=z[prefix+'_weight'];expected=z[prefix+'_R']
            for pol in range(2):
                grid=weights[pol].reshape(14,14);nodes,edges=graph(grid)
                w=grid.ravel()[nodes];x=r[pol][:,nodes];prob=w/w.sum()
                variance=(x*x*prob).sum(1)
                alpha=w[edges[:,0]]*w[edges[:,1]]
                local=np.sum(alpha*(x[:,edges[:,0]]-x[:,edges[:,1]])**2,axis=1)/(2*alpha.sum())
                ii,jj=np.triu_indices(len(nodes),1);allweight=w[ii]*w[jj]
                complete=np.sum(allweight*(x[:,ii]-x[:,jj])**2,axis=1)/(2*allweight.sum())
                actual=np.stack([complete/variance,local/variance],-1)
                error=float(abs(actual-expected[pol]).max());assert error<1e-10
                checks.append(dict(point=prefix,pol=pol,ratio_replay_max_abs=error))
    pd.DataFrame(checks).to_csv(HERE/'onepoint_checks.csv',index=False)
    pred=pd.read_csv(HERE/'predictions.csv');curves=pd.read_csv(HERE/'selection_curves.csv')
    rows=[]
    for (seed,arm),g in curves.groupby(['seed','arm']):
        initial=g[g.epoch.eq(0)].iloc[0];terminal=g[g.epoch.eq(200)].iloc[0]
        rows.append(dict(seed=seed,arm=arm,initial_val_J=initial.J,terminal_val_J=terminal.J,
            terminal_train_jointCE_pre_update=terminal.train_loss,
            terminal_val_OA=terminal.OA,terminal_val_BA=terminal.BA,
            delta_val_NLL=terminal.NLL-initial.NLL,delta_val_balanced_NLL=terminal.balanced_NLL-initial.balanced_NLL))
    pd.DataFrame(rows).to_csv(HERE/'terminal_summary.csv',index=False)
    agreement=[]
    for seed in [17,29,43]:
        local=pred[pred.seed.eq(seed)&pred.arm.eq('local')&pred.role.eq('validation')].sort_values('instance_id')
        for control in ['marginal','complete']:
            other=pred[pred.seed.eq(seed)&pred.arm.eq(control)&pred.role.eq('validation')].sort_values('instance_id')
            agreement.append(dict(seed=seed,control=control,top1_different=int((local.prediction.to_numpy()!=other.prediction.to_numpy()).sum())))
    pd.DataFrame(agreement).to_csv(HERE/'selected_agreement.csv',index=False)
    # Local seed17 selected epoch0, so use explicitly labelled complete-arm cases
    # for the two outcome types absent from the primary local readout.
    with np.load(HERE/'topology_features.npz') as z:
        ratios,kish=z['ratios'],z['kish']
    cases=[]
    for outcome in ['rescued','harmed']:
        g=pred[pred.seed.eq(17)&pred.arm.eq('complete')&pred.role.eq('validation')&pred.outcome.eq(outcome)].sort_values('instance_id')
        if not len(g):cases.append(dict(outcome=outcome,available=False));continue
        row=g.iloc[0];i=int(row.source_row)
        fig,axes=plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
        for p,pol in enumerate(['vv','vh']):
            with Image.open(row[f'chip_path_{pol}']) as im:raw=np.array(im)
            wt=footprint(*raw.shape);nodes,edges=graph(wt);pos=np.stack([nodes%14,nodes//14],-1)
            axes[p,0].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[p,0].set_title(f'{pol.upper()} native {raw.shape[1]}x{raw.shape[0]}')
            for col,kind in [(1,'local'),(2,'complete')]:
                ax=axes[p,col]
                if kind=='local':ee=edges
                else:ee=np.stack(np.triu_indices(len(nodes),1),axis=1)
                ax.add_collection(LineCollection(pos[ee],colors='#335577',linewidths=.5,alpha=.6 if col==1 else .14))
                ax.scatter(pos[:,0],pos[:,1],s=6+15*wt.ravel()[nodes],color='black')
                ax.set_xlim(-.5,13.5);ax.set_ylim(13.5,-.5);ax.set_aspect('equal')
                value=ratios[i,p,int(row.class_id),1 if kind=='local' else 0]
                ax.set_title(f'{kind}: {len(ee)} edges; Kish={kish[i,p]:.2f}\nTrue-class C4 R={value:.3f}',fontsize=10)
            for ax in axes[p]:ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f'{row.instance_id} COMPLETE {outcome}; y={row.class_id}, B={row.base_prediction}, complete={row.prediction}\nLocal seed17 is epoch0: no local rescue/harm; graph shown is geometry, not a response map',fontsize=11)
        path=HERE/f'complete_{outcome}.png';fig.savefig(path,dpi=130);plt.close(fig)
        rec=row.to_dict();rec.update(available=True,panel=str(path));cases.append(rec)
    pd.DataFrame(cases).to_csv(HERE/'complete_cases.csv',index=False)
    print(json.dumps(dict(max_onepoint_ratio_error=max(r['ratio_replay_max_abs'] for r in checks),agreement=agreement)))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(cases)[['instance_id','class_id','base_prediction','prediction','outcome']].to_string(index=False))


if __name__=='__main__':main()
