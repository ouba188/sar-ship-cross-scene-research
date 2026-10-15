"""Does satellite platform add a replicable paired response beyond angle/position?"""

import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='2'

import json
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from conditioned_vessel_drift import split_observations, condition_differences, fit_response, response_metrics
from same_vessel_temporal import collect_observations, make_pairs
from run_same_vessel_temporal_probe import latent
from run_conditioned_vessel_drift import SIDECAR
from task_stable_sensitivity import SOURCE, PARENT, ROOT
from sar_joint_completion import write_json


OUT=ROOT/'platform_condition_probe_20261009'
MODELS={'angle_position':list(range(4)), 'plus_platform':list(range(5)), 'platform_angle_interaction':list(range(7))}


def platform_pair_features(obs,conditions,pairs):
    base=condition_differences(obs,conditions[:,:3],pairs)
    a=pairs.obs_a.to_numpy(dtype=int); b=pairs.obs_b.to_numpy(dtype=int)
    t=(conditions[:,0]-38.)/5.; platform=conditions[:,3]
    features=np.column_stack((platform,platform*t,platform*t*t))
    return np.column_stack((base,features[b]-features[a]))


def run_port(folder,frame):
    out=OUT/'ports'/folder.name; out.mkdir(parents=True,exist_ok=True)
    sr=np.load(folder/'source_rows.npy'); tr=np.load(folder/'target_rows.npy')
    with (folder/'geometry.pkl').open('rb') as f: g=pickle.load(f)
    bank=np.load(folder/'source_bank.npy',mmap_mode='r'); z=latent(bank,g); width=128+z.shape[1]
    meta=frame.loc[sr,['incidence_deg','latitude','longitude','platform_c']].to_numpy(float)
    obs,values=collect_observations(frame,sr,np.column_stack((bank[:,:128],z,meta)))
    good=np.isfinite(values[:,width:]).all(axis=1)&np.isin(values[:,-1],[0.,1.])
    obs=obs.loc[good].reset_index(drop=True); values=values[good]
    if np.intersect1d(obs.mmsi,frame.loc[tr,'mmsi']).size: raise ValueError('Target vessel in source')
    split=split_observations(obs); models={}; records=[]; coverage=[]; by_class=[]
    train_ships=set(obs.loc[split['fit'],'mmsi']); train_products=set(obs.loc[split['fit'],'product'])
    for name,mask in split.items():
        o=obs.loc[mask].reset_index(drop=True); v=values[mask]
        pairs,dy,w=make_pairs(o,v[:,:width]); x=platform_pair_features(o,v[:,width:],pairs)
        if name in ('new_ship','new_ship_month') and train_ships & set(o.mmsi): raise ValueError('Vessel overlap')
        if name in ('new_month','new_ship_month') and train_products & set(o['product']): raise ValueError('Product overlap')
        pairs['weight']=w; pairs['platform_switch']=np.abs(x[:,4])>.5
        pairs['product_a']=o.iloc[pairs.obs_a]['product'].to_numpy(); pairs['product_b']=o.iloc[pairs.obs_b]['product'].to_numpy()
        pairs.to_csv(out/(name+'_pairs.csv'),index=False)
        coverage.append(dict(target_port=folder.name,split=name,pairs=len(pairs),ships=pairs.mmsi.nunique(),
            platform_switch_pairs=int(pairs.platform_switch.sum()),weighted_switch_fraction=float(w @ pairs.platform_switch.to_numpy()),
            switch_ships=pairs[pairs.platform_switch].mmsi.nunique(),switch_ports=pairs[pairs.platform_switch].port.nunique(),
            switch_classes=pairs[pairs.platform_switch].class_id.nunique(),condition_rank=int(np.linalg.matrix_rank(x))))
        for space,cols in [('retained128',slice(0,128)),('stable_drift',slice(128,width))]:
            y=dy[:,cols]
            for model,features in MODELS.items():
                key=space+'_'+model; xx=x[:,features]
                if name=='fit': models[key]=fit_response(xx,y,w)
                pred=xx @ models[key]
                records.append(dict(target_port=folder.name,split=name,space=space,model=model,**response_metrics(y,pred,w)))
                for c in sorted(pairs.class_id.unique()):
                    use=pairs.class_id.eq(c).to_numpy(); ww=w[use]/w[use].sum()
                    by_class.append(dict(target_port=folder.name,split=name,space=space,model=model,class_id=int(c),
                        pairs=int(use.sum()),**response_metrics(y[use],pred[use],ww)))
    np.savez_compressed(out/'response_models.npz',**models)
    write_json(out/'COMPLETE.json',dict(source_observations=len(obs),target_query_labels_used=0))
    print(json.dumps(dict(port=folder.name,coverage=coverage[-1],held=[r for r in records if r['split']=='new_ship_month' and r['space']=='stable_drift'])),flush=True)
    return records,coverage,by_class


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'COMPLETE.json').exists(): raise ValueError('Preserve completed experiment')
    write_json(OUT/'PROTOCOL.json',dict(date='2026-10-09',factor='Satellite S1A/S1C from exact manifest product prefix',
        reference='same complete cases and angle+position source-only paired regression',models=MODELS,
        ridge=.01,identity_seed=20261009,held_months=[3,6,9,12],target_query_labels_used=0,
        boundary='Existing source-fitted geometry includes held source; conditional association not causal identification',main_model_unchanged=True))
    frame=pd.read_csv(SOURCE/'manifest.csv',usecols=['instance_id','product','port','class_id','mmsi'],dtype={'mmsi':str},keep_default_na=False)
    side=pd.read_csv(SIDECAR,usecols=['instance_id','product','status','incidence_deg','latitude','longitude'])
    frame=frame.merge(side,on=['instance_id','product'],how='left',sort=False,validate='one_to_one')
    frame.loc[frame.status.ne('ok'),['incidence_deg','latitude','longitude']]=np.nan
    frame['platform_c']=frame['product'].str.split('_').str[0].map({'S1A':0.,'S1C':1.})
    frame.groupby(['port','class_id','platform_c'],dropna=False).size().rename('rows').to_csv(OUT/'platform_inventory.csv')
    raw_records=[]; coverage=[]; byclass=[]
    with threadpool_limits(limits=2):
        for folder in sorted((PARENT/'ports').iterdir()):
            if not (folder/'policies.npz').exists(): continue
            r,c,k=run_port(folder,frame); raw_records.extend(r); coverage.extend(c); byclass.extend(k)
    scores=pd.DataFrame(raw_records); scores.to_csv(OUT/'response_metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'pair_coverage.csv',index=False)
    pd.DataFrame(byclass).to_csv(OUT/'class_metrics.csv',index=False)
    scores.groupby(['space','split','model']).mean(numeric_only=True).to_csv(OUT/'response_means.csv')
    p=scores.pivot(index=['target_port','space','split'],columns='model',values='skill').reset_index()
    p['platform_increment']=p.plus_platform-p.angle_position
    p['interaction_increment']=p.platform_angle_interaction-p.angle_position
    p.to_csv(OUT/'incremental_skill.csv',index=False)
    # This is the same complete-case baseline, so a discrepancy signals an implementation issue.
    old=pd.read_csv(ROOT/'conditioned_vessel_drift_20261009/response_metrics.csv')
    old=old[old.model=='incidence_position'][['target_port','space','split','skill']]
    check=p.merge(old,on=['target_port','space','split'],validate='one_to_one')
    error=float(np.max(np.abs(check.angle_position-check.skill)))
    if error>1e-10: raise ValueError(f'Original condition baseline did not replay: {error}')
    write_json(OUT/'COMPLETE.json',dict(ports=23,records=len(scores),target_query_labels_used=0,max_baseline_replay_error=error))
    print(p.groupby(['space','split'])[['platform_increment','interaction_increment']].mean().to_string())


if __name__=='__main__': main()
