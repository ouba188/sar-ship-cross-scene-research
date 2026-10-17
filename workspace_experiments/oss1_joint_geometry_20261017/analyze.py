"""Paired full-system effects, scope, and case-level attribution tables."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,MANIFEST,folds


def main():
    names=[f[0] for f in folds()]
    metrics=pd.concat([pd.read_csv(HERE/f/'metrics.csv') for f in names],ignore_index=True)
    fits=pd.concat([pd.read_csv(HERE/f/'fits.csv') for f in names],ignore_index=True)
    pred=pd.concat([pd.read_csv(HERE/f/'predictions.csv') for f in names],ignore_index=True)
    m=pd.read_csv(MANIFEST,dtype={'mmsi_or_ship_id':str})
    det=pd.read_csv(HERE/'detections.csv').set_index('global_row')
    geo=pd.read_csv(HERE/'geometry.csv').set_index('global_row')
    metrics.to_csv(HERE/'ALL_METRICS.csv',index=False)
    fits.to_csv(HERE/'ALL_FITS.csv',index=False)
    cols=['oa','ba','delta_oa','delta_ba','baseline_oa','baseline_ba','brier','nll','delta_brier','rescues','harms']
    fold=metrics.groupby(['policy','origin','arm','fold'])[cols].mean().reset_index()
    table=fold.groupby(['policy','origin','arm'])[cols].mean().reset_index()
    table.to_csv(HERE/'MAIN_TABLE.csv',index=False)
    fold.to_csv(HERE/'FOLD_TABLE.csv',index=False)
    seed=metrics.groupby(['policy','origin','arm','seed'])[cols].mean().reset_index()
    seed.to_csv(HERE/'SEED_TABLE.csv',index=False)
    keys=['policy','origin','fold','seed','draw','global_row']
    paired=[]
    for control in ['image_ring','ship','ship_radar_encoding','body_only']:
        a=pred[pred.arm.eq('joint')]
        b=pred[pred.arm.eq(control)]
        z=a.merge(b,on=keys,suffixes=('_joint','_control'),validate='one_to_one')
        np.testing.assert_array_equal(z.truth_joint,z.truth_control)
        z['correct_joint']=z.prediction_joint.eq(z.truth_joint).astype(float)
        z['correct_control']=z.prediction_control.eq(z.truth_control).astype(float)
        z['net_correct']=z.correct_joint-z.correct_control
        z['control']=control
        paired.append(z[keys+['control','truth_joint','prediction_joint','prediction_control','net_correct']])
    paired=pd.concat(paired,ignore_index=True)
    paired.to_csv(HERE/'PAIRED_CASES.csv',index=False)
    class_effect=paired.groupby(['policy','origin','fold','control','truth_joint']).net_correct.mean().mul(100).rename('delta_recall_pp').reset_index()
    class_effect['query_ships']=class_effect.apply(lambda r: paired[
        (paired.policy==r.policy)&(paired.origin==r.origin)&(paired.fold==r.fold)&
        (paired.control==r.control)&(paired.truth_joint==r.truth_joint)].global_row.nunique(),axis=1)
    class_effect.to_csv(HERE/'CLASS_EFFECTS.csv',index=False)
    comparisons=[]
    for (policy,origin,control),z in paired.groupby(['policy','origin','control']):
        per=z.groupby(['fold','seed','draw']).apply(lambda g:pd.Series(dict(
            delta_oa=100*g.net_correct.mean(),delta_ba=100*g.groupby('truth_joint').net_correct.mean().mean())),include_groups=False).reset_index()
        by_seed=per.groupby('seed')[['delta_oa','delta_ba']].mean()
        by_fold=per.groupby('fold')[['delta_oa','delta_ba']].mean()
        comparisons.append(dict(policy=policy,origin=origin,control=control,
            delta_oa=by_fold.delta_oa.mean(),delta_ba=by_fold.delta_ba.mean(),
            seed_sd_ba=by_seed.delta_ba.std(),positive_seed_means=int((by_seed.delta_ba>0).sum()),
            positive_folds=int((by_fold.delta_ba>0).sum()),
            rescued_unique_rows=z.loc[z.net_correct>0,'global_row'].nunique(),
            harmed_unique_rows=z.loc[z.net_correct<0,'global_row'].nunique()))
    pd.DataFrame(comparisons).to_csv(HERE/'JOINT_VS_CONTROLS.csv',index=False)
    rows=sorted(pred.global_row.unique())
    inventory=m.iloc[rows].copy();inventory['global_row']=rows
    inventory['axis_valid']=det.loc[rows,'axis_valid'].to_numpy()
    inventory['relative_axis_deg']=np.rad2deg(det.loc[rows,'ship_axis'].to_numpy())%180
    inventory['relative_axis_deg']=np.minimum(inventory.relative_axis_deg,180-inventory.relative_axis_deg)
    inventory['incidence_deg']=geo.loc[rows,'incidence_deg'].to_numpy()
    inventory.to_csv(HERE/'QUERY_INVENTORY.csv',index=False)
    scope=dict(query_rows=len(rows),query_ships=inventory.mmsi_or_ship_id.nunique(),
               query_products=inventory.scene_id.nunique(),valid_query_axes=int(inventory.axis_valid.sum()),
               local_fits=len(fits),source_backbone_seeds_per_fold=1,local_reader_seeds=3,
               same_query_support_draws=5,query_classes=inventory.groupby('class_name').size().to_dict(),
               geometry_gate='same detected elongated-axis eligibility in all arms; no AIS heading',
               discarded_source_updates=int((fits.alpha==0).sum()))
    (HERE/'SUMMARY.json').write_text(json.dumps(scope,indent=2),encoding='utf-8')
    arms=['image_ring','ship','ship_radar_encoding','joint','body_only']
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,origin in zip(axes,['cross','same']):
        x=fold[(fold.policy=='source_selected')&(fold.origin==origin)]
        for i,arm in enumerate(arms):
            values=x[x.arm.eq(arm)].delta_ba.to_numpy()
            ax.scatter(np.arange(len(values))+(i-2)*.1,values,s=25,label=arm)
        ax.axhline(0,color='black',lw=.8)
        ax.set_xticks(range(4),names,rotation=20,ha='right',fontsize=8)
        ax.set_title(origin+' acquisition supports');ax.set_ylabel('Full BA change vs frozen baseline (pp)')
    axes[1].legend(fontsize=8,loc='best')
    fig.savefig(HERE/'fold_effects.png',dpi=160);plt.close(fig)
    print('MAIN',table[table.policy.eq('source_selected')].to_string(index=False),flush=True)
    print('JOINT_CONTROLS',pd.DataFrame(comparisons).query("policy=='source_selected'").to_string(index=False),flush=True)
    print('SCOPE',json.dumps(scope),flush=True)


if __name__=='__main__':
    main()
