"""Saved-head donor intervention and native query/donor panels; no training."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from osgeo import gdal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_scale import HERE,SEEDS,restore,measures


def main():
    torch.set_num_threads(2)
    inputs=pd.read_csv(HERE/'source_inputs.csv',dtype={'mmsi':str})
    previous=pd.read_csv(HERE/'predictions.csv')
    with np.load(HERE/'source_inputs.npz') as z:
        cache={k:z[k] for k in z.files}
    va=cache['validation'];donor=cache['donor'];anchors=cache['log_anchor']
    rows=[];metrics=[];checks=[]
    for seed in SEEDS:
        base=cache[f'B{seed}']
        common=np.column_stack([base-base.mean(1,keepdims=True),cache['png_log_rms'],cache['log_pixels']])
        raw=np.column_stack([common,anchors]);swap=np.column_stack([common,anchors[donor]])
        ck,model=restore(HERE/'heads'/f'own_scale_{seed}.pt')
        with torch.inference_mode():
            original=base+model(torch.tensor((raw-ck['mu'])/ck['scale'],dtype=torch.float32)).numpy().astype(float)
            changed=base+model(torch.tensor((swap-ck['mu'])/ck['scale'],dtype=torch.float32)).numpy().astype(float)
        expected=previous[previous.seed.eq(seed)&previous.arm.eq('own_scale')].sort_values('source_row')
        np.testing.assert_array_equal(expected.source_row,np.arange(len(inputs)))
        err=float(abs(original-expected[[f's{c}' for c in range(8)]].to_numpy()).max());assert err<1e-12
        assert not (original[donor==np.arange(len(donor))]!=changed[donor==np.arange(len(donor))]).any()
        if ck['epoch']==0:np.testing.assert_array_equal(original,changed)
        checks.append(dict(seed=seed,epoch=ck['epoch'],original_score_replay_max_abs=err,
                           zero_head=ck['epoch']==0,max_swap_change=float(abs(original[va]-changed[va]).max())))
        g=inputs.iloc[va][['sample_id','source_row','class_id','class_name','pass_group','matched','donor_id','donor_row','donor_class','match_distance']].copy()
        g['seed']=seed;g['epoch']=ck['epoch'];y=g.class_id.to_numpy()
        g['base_prediction']=base[va].argmax(1);g['original_prediction']=original[va].argmax(1);g['swapped_prediction']=changed[va].argmax(1)
        g['net_swap_minus_original']=(g.swapped_prediction==y).astype(int)-(g.original_prediction==y).astype(int)
        g['swap_outcome']=np.select([g.net_swap_minus_original.eq(1),g.net_swap_minus_original.eq(-1),g.swapped_prediction.eq(y)],['rescued_by_swap','harmed_by_swap','CC'],default='WW')
        for c in range(8):
            g[f'b{c}']=base[va,c];g[f'original{c}']=original[va,c];g[f'swapped{c}']=changed[va,c]
        for p,pol in enumerate(['vv','vh']):
            g[f'{pol}_own_log_anchor']=anchors[va,p];g[f'{pol}_donor_log_anchor']=anchors[donor[va],p]
        rows.append(g)
        for subset,mask in [('all',np.ones(len(va),bool)),('matched',inputs.iloc[va].matched.to_numpy())]:
            sub=va[mask];truth=inputs.iloc[sub].class_id.to_numpy()
            orig=measures(truth,original[sub]);sw=measures(truth,changed[sub]);bas=measures(truth,base[sub])
            delta=(changed[sub].argmax(1)==truth).astype(int)-(original[sub].argmax(1)==truth).astype(int)
            metrics.append(dict(seed=seed,subset=subset,n=len(sub),epoch=ck['epoch'],
                flips=int((changed[sub].argmax(1)!=original[sub].argmax(1)).sum()),
                rescued_by_swap=int((delta==1).sum()),harmed_by_swap=int((delta==-1).sum()),net_swap_minus_original=int(delta.sum()),
                **{f'original_{k}':orig[k] for k in ['OA','BA','NLL','balanced_NLL','Brier']},
                **{f'swapped_{k}':sw[k] for k in ['OA','BA','NLL','balanced_NLL','Brier']},
                **{f'base_{k}':bas[k] for k in ['OA','BA','NLL']},
                **{f'delta_{k}':sw[k]-orig[k] for k in ['OA','BA','NLL','balanced_NLL','Brier']}))
    intervention=pd.concat(rows,ignore_index=True)
    intervention.to_csv(HERE/'fixed_head_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(HERE/'fixed_head_metrics.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'fixed_head_checks.csv',index=False)
    intervention[intervention.original_prediction.ne(intervention.swapped_prediction)].to_csv(HERE/'fixed_head_flips.csv',index=False)
    curves=pd.read_csv(HERE/'selection_curves.csv');terminal=[]
    for (seed,arm),g in curves.groupby(['seed','arm']):
        initial=g[g.epoch.eq(0)].iloc[0];end=g[g.epoch.eq(200)].iloc[0]
        terminal.append(dict(seed=seed,arm=arm,initial_train_jointCE=initial.train_loss,
            terminal_train_jointCE_pre_update=end.train_loss,initial_val_J=initial.J,terminal_val_J=end.J,
            checks_J_above_zero=int((g.J>initial.J+1e-14).sum()),terminal_val_OA=end.OA,terminal_val_BA=end.BA,
            terminal_val_NLL=end.NLL,terminal_val_balanced_NLL=end.balanced_NLL))
    pd.DataFrame(terminal).to_csv(HERE/'terminal_summary.csv',index=False)
    # Original predeclared own-vs-B cases, plus the two explicitly requested cases.
    selection=[]
    own=previous[previous.seed.eq(17)&previous.arm.eq('own_scale')&previous.role.eq('validation')]
    for outcome in ['rescued','harmed','CC','WW']:
        g=own[own.outcome.eq(outcome)].sort_values('instance_id')
        if len(g):selection.append((outcome,g.iloc[0].instance_id))
        else:selection.append((outcome,None))
    selection.extend([('offshore_5340','pub2_opensarship2_005340'),('offshore_5344','pub2_opensarship2_005344')])
    for outcome in ['rescued_by_swap','harmed_by_swap']:
        changed=intervention[intervention.seed.eq(17)&intervention.swap_outcome.eq(outcome)].sort_values('sample_id')
        selection.append((outcome,None if not len(changed) else changed.iloc[0].sample_id))
    limits=json.loads((HERE.parent/'display_limits.json').read_text(encoding='utf-8'))
    gdal.UseExceptions();gdal.PushErrorHandler('CPLQuietErrorHandler')
    cases=[]
    for label,sid in selection:
        if sid is None:cases.append(dict(case=label,available=False));continue
        q=inputs[inputs.sample_id.eq(sid)].iloc[0];d=inputs.iloc[int(q.donor_row)]
        chosen=intervention[intervention.seed.eq(17)&intervention.sample_id.eq(sid)].iloc[0]
        arm_preds={arm:int(previous[previous.seed.eq(17)&previous.arm.eq(arm)&previous.instance_id.eq(sid)].iloc[0].prediction) for arm in ['png_summary','own_scale','matched_scale']}
        dbase=int(previous[previous.seed.eq(17)&previous.arm.eq('own_scale')&previous.instance_id.eq(d.sample_id)].iloc[0].base_prediction)
        fig,axes=plt.subplots(2,4,figsize=(14,7.8),constrained_layout=True)
        for p,pol in enumerate(['vv','vh']):
            for col,obj in [(0,q),(1,d)]:
                with Image.open(obj[f'chip_path_{pol}']) as im:png=np.array(im)
                axes[p,col].imshow(png,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[p,col].set_title(f'{pol.upper()} {"query" if col==0 else "donor"} PNG {str(obj.sample_id)[-6:]}\n{png.shape[1]}x{png.shape[0]}',fontsize=10)
                dataset=gdal.Open(obj[f'{pol}_calibrated_path']);f=dataset.GetRasterBand(int(obj[f'{pol}_band'])).ReadAsArray().astype(float);dataset=None
                log=np.full(f.shape,np.nan);log[f>0]=10*np.log10(f[f>0])
                image=axes[p,col+2].imshow(log,cmap='gray',vmin=limits[pol.upper()][0],vmax=limits[pol.upper()][1],interpolation='nearest')
                axes[p,col+2].set_title(f'{"Query" if col==0 else "Donor"} 10log10(Patch_Cal)\nshared {pol.upper()} limits; units unverified',fontsize=10)
                fig.colorbar(image,ax=axes[p,col+2],fraction=.04)
                axes[p,col+2].set_xlabel(f'anchor255={obj[f"{pol}_amplitude_anchor_255"]:.4g}; max={obj[f"{pol}_float_q1"]:.4g}',fontsize=9)
            for ax in axes[p]:ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f"{sid[-6:]} {label}: y={q.class_id}, B={chosen.base_prediction}, summary={arm_preds['png_summary']}, own={chosen.original_prediction}, trained-matched={arm_preds['matched_scale']}\nFixed own head + donor anchors: {chosen.swapped_prediction}; donor {str(d.sample_id)[-6:]} y={d.class_id}, B={dbase}; matched={bool(q.matched)}",fontsize=11)
        path=HERE/f'case_{label}.png';fig.savefig(path,dpi=120);plt.close(fig)
        rec=chosen.to_dict();rec.update(case=label,available=True,panel=str(path),query_mmsi=q.mmsi,donor_mmsi=d.mmsi,
            query_path_vv=q.chip_path_vv,query_path_vh=q.chip_path_vh,donor_path_vv=d.chip_path_vv,donor_path_vh=d.chip_path_vh,
            summary_prediction=arm_preds['png_summary'],trained_matched_prediction=arm_preds['matched_scale'])
        cases.append(rec)
    pd.DataFrame(cases).to_csv(HERE/'review_cases.csv',index=False)
    print(pd.DataFrame(metrics).to_string(index=False))
    print(pd.DataFrame(terminal).to_string(index=False))
    print(pd.DataFrame(cases)[['case','sample_id','class_id','donor_id','donor_class','base_prediction','original_prediction','swapped_prediction','matched']].to_string(index=False))


if __name__=='__main__':main()
