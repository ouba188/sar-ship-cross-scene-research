"""Chronological final-observation prediction with held-vessel transfer."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
METRICS=['length_m','width_m','r3_length_m','r3_width_m']
GEOM=['incidence','sin2phi','cos2phi']
META=['day','platform_b','platform_c','ascending']
ARMS={'transfer_metadata':META,'transfer_incidence':META+['incidence'],
      'transfer_direction':META+['sin2phi','cos2phi'],'transfer_geometry':META+GEOM}


def fit_relation(frame,metric,columns,alpha,fixed_scale=None):
    x=frame[columns].to_numpy(float);y=np.log(frame[metric].to_numpy(float))
    groups=frame.mmsi.to_numpy()
    xc=np.empty_like(x);yc=np.empty_like(y);weights=np.empty(len(y))
    for ship in np.unique(groups):
        use=groups==ship
        xc[use]=x[use]-x[use].mean(0);yc[use]=y[use]-y[use].mean();weights[use]=1/use.sum()
    scale=np.sqrt(np.average(xc*xc,axis=0,weights=weights))
    active=scale>1e-8
    if fixed_scale is not None:
        scale=np.asarray(fixed_scale,float)
    safe=np.where(active,scale,1)
    z=xc/safe;z[:,~active]=0
    coef=np.linalg.solve(z.T@(weights[:,None]*z)+alpha*np.eye(len(columns)),z.T@(weights*yc))
    return coef/safe*active


def predict(history,query,metric,columns,beta,last=False):
    ref=history.iloc[[-1]] if last else history
    return float(np.log(ref[metric]).mean()+(query[columns].to_numpy(float)-ref[columns].mean().to_numpy())@beta)


def load():
    f=pd.read_csv(ROOT/'multitemporal_20261021/measurements.csv',dtype={'mmsi':str})
    c=pd.read_csv(ROOT/'multitemporal_20261021/cohort.csv',dtype={'mmsi':str})
    f=f.merge(c[['obs_id','overpass','prediction','correct']],on='obs_id',validate='many_to_one')
    f['timestamp']=pd.to_datetime(f.time)
    f['day']=(f.timestamp-pd.Timestamp('2025-01-01')).dt.total_seconds()/86400
    f['platform_b']=f.platform.eq('S1B').astype(float)
    f['platform_c']=f.platform.eq('S1C').astype(float)
    f['ascending']=f.pass_name.str.lower().eq('ascending').astype(float)
    ships=np.array(sorted(f.mmsi.unique()))
    folds={}
    for fold,(_,test) in enumerate(KFold(5,shuffle=True,random_state=20261021).split(ships)):
        folds.update({s:fold for s in ships[test]})
    f['split_fold']=f.mmsi.map(folds)
    return f.sort_values(['timestamp','obs_id']),folds


def run_variant(frame,variant):
    results=[];manifests=[];cache={}
    for pol,full in frame.groupby('pol'):
        last_ids=full.sort_values(['timestamp','obs_id']).groupby('mmsi').tail(1).obs_id.tolist()
        for metric in METRICS:
            valid=full[full[metric].gt(0)&np.isfinite(full[metric])].copy()
            if metric.startswith('r3_'): valid=valid[valid.r3_valid.eq(True)]
            for oid in last_ids:
                qframe=valid[valid.obs_id.eq(oid)]
                if qframe.empty: continue
                q=qframe.iloc[0];hist=valid[valid.mmsi.eq(q.mmsi)&valid.timestamp.lt(q.timestamp)].sort_values(['timestamp','obs_id'])
                if len(hist)<3: continue
                key=(pol,metric,int(q.split_fold),q.time,q.overpass)
                if key not in cache:
                    src=valid[valid.split_fold.ne(q.split_fold)&valid.timestamp.lt(q.timestamp)&valid.overpass.ne(q.overpass)].copy()
                    src=src.groupby('mmsi').filter(lambda x:len(x)>=2)
                    assert q.mmsi not in set(src.mmsi)
                    assert (src.timestamp<q.timestamp).all() and not src.overpass.eq(q.overpass).any()
                    if src.mmsi.nunique()<5: continue
                    fitted={name:fit_relation(src,metric,columns,10) for name,columns in ARMS.items()}
                    fit_id=f'{variant}_{len(cache)}'
                    manifests.append(dict(fit_id=fit_id,pol=pol,metric=metric,excluded_fold=int(q.split_fold),
                        query_time=q.time,query_overpass=q.overpass,source_obs_ids=src.obs_id.astype(int).tolist(),
                        source_ships=src.mmsi.nunique(),source_max_time=src.time.max(),
                        coefficients={name:dict(zip(ARMS[name],coef.tolist())) for name,coef in fitted.items()}))
                    cache[key]=(src,fitted,fit_id)
                src,fitted,fit_id=cache[key]
                models={'history_mean':float(np.log(hist[metric]).mean()),'last_observation':float(np.log(hist[metric].iloc[-1]))}
                own=fit_relation(hist,metric,GEOM,1)
                models['own_geometry']=predict(hist,q,metric,GEOM,own)
                own_fixed=fit_relation(hist,metric,GEOM,1,fixed_scale=[5.,1.,1.])
                models['own_geometry_fixedscale']=predict(hist,q,metric,GEOM,own_fixed)
                for name,columns in ARMS.items(): models[name]=predict(hist,q,metric,columns,fitted[name])
                models['transfer_geometry_lastref']=predict(hist,q,metric,ARMS['transfer_geometry'],fitted['transfer_geometry'],True)
                actual=float(np.log(q[metric]))
                geometry_out=bool(((q[GEOM]<hist[GEOM].min()-1e-8)|(q[GEOM]>hist[GEOM].max()+1e-8)).any())
                for name,value in models.items():
                    results.append(dict(variant=variant,pol=pol,metric=metric,mmsi=q.mmsi,obs_id=int(oid),
                        class_id=int(q.class_id),port=q.port,time=q.time,split_fold=int(q.split_fold),
                        history_ids=' '.join(hist.obs_id.astype(str)),history_n=len(hist),fit_id=fit_id,
                        source_n=len(src),source_ships=src.mmsi.nunique(),arm=name,actual_log=actual,pred_log=value,
                        actual_m=float(q[metric]),pred_m=float(np.exp(value)),abs_log_error=abs(value-actual),
                        abs_m_error=abs(np.exp(value)-q[metric]),own_geometry_outside_history_box=geometry_out,
                        classifier_prediction=int(q.prediction),classifier_correct=bool(q.correct)))
    return pd.DataFrame(results),manifests


def summarize(pred):
    rng=np.random.default_rng(20261021);rows=[]
    for (variant,pol,metric),f in pred.groupby(['variant','pol','metric']):
        wide=f.pivot(index='mmsi',columns='arm',values='abs_log_error')
        draws=rng.integers(0,len(wide),size=(10000,len(wide)))
        for arm in wide:
            for base in ('history_mean','last_observation','transfer_metadata'):
                gain=(wide[base]-wide[arm]).to_numpy()
                boot=gain[draws].mean(1)
                rows.append(dict(variant=variant,pol=pol,metric=metric,arm=arm,baseline=base,ships=len(wide),
                    log_mae=wide[arm].mean(),baseline_log_mae=wide[base].mean(),mean_gain=gain.mean(),
                    relative_mae_reduction_pct=100*gain.mean()/wide[base].mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),
                    improved_ships=int((gain>1e-12).sum()),harmed_ships=int((gain < -1e-12).sum()),
                    metre_mae=f[f.arm.eq(arm)].abs_m_error.mean()))
    return pd.DataFrame(rows)


def main():
    frame,folds=load();frames=[];members=[]
    for variant,subset in [('all',frame),('omit_flagged_ship',frame[frame.mmsi.ne('533150044')])]:
        p,m=run_variant(subset,variant);frames.append(p);members.extend(m)
        print(variant,'predictions',len(p),'source fits',len(m),flush=True)
    pred=pd.concat(frames,ignore_index=True);pred.to_csv(OUT/'predictions.csv',index=False,lineterminator='\n')
    summary=summarize(pred);summary.to_csv(OUT/'summary.csv',index=False,lineterminator='\n')
    (OUT/'fit_manifest.json').write_text(json.dumps(members,indent=2),encoding='utf-8',newline='\n')
    (OUT/'folds.json').write_text(json.dumps(folds,indent=2),encoding='utf-8',newline='\n')
    keep=summary[summary.variant.eq('all')&summary.baseline.eq('history_mean')&summary.arm.isin(['own_geometry','transfer_direction','transfer_geometry','last_observation'])]
    print(keep[['pol','metric','arm','ships','relative_mae_reduction_pct','ci_low','ci_high']].to_string(index=False),flush=True)


if __name__=='__main__': main()
