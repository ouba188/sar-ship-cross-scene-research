
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
import json,math,importlib.util,time
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image
R=Path('E:/临时会话/same_vessel_gallery_20261019')
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=module('morph',R/'condition_morphometry_20261021/run.py')
nxt=module('nextobs',R/'next_observation_20261021/run.py')
f,folds=nxt.load()
valid=f[f.r3_valid.eq(True)].copy()
data=json.loads((R/'gallery_data.js').read_text(encoding='utf8').split('=',1)[1].strip().removesuffix(';'))['observations']
rows=[];start=time.time()
for i,r in enumerate(valid.itertuples()):
 view=data[str(r.obs_id)]['images'][r.pol];pri=view['priors']
 im=np.asarray(Image.open(r.image_path).convert('L'),float)
 g=np.asarray(pri['scale']['transform'],float)
 a=g[[1,2,4,5]].reshape(2,2);area=abs(np.linalg.det(a))
 yy,xx=np.indices(im.shape);pixels=np.column_stack([xx.ravel()+.5,yy.ravel()+.5])
 center=np.asarray(pri['radar_center']);points=(pixels-center)@a.T
 box=pri['detections'][pri['target_index']]
 verts=(np.asarray(box['corners'])-center)@a.T
 edge=np.roll(verts,-1,axis=0)-verts
 lengths=np.linalg.norm(edge[:2],axis=1);axis=edge[np.argmax(lengths)]/max(lengths)
 other=np.zeros(len(points),bool)
 for j,d in enumerate(pri['detections']):
  if j!=pri['target_index'] and d['confidence']>=.25 and d['shape_usable']:
   other|=old.polygon_inside(points,(np.asarray(d['corners'])-center)@a.T)
 rad=np.linalg.norm(points,axis=1);core=(rad<=180)&~other;bg=(rad>=220)&(rad<300)&~other
 intensity=im.ravel();b=intensity[bg];med=np.median(b);mad=max(1.,1.4826*np.median(abs(b-med)))
 threshold=med+3*mad;weights=np.maximum(intensity-threshold,0)*core;use=weights>0
 pp=points[use];ww=weights[use];original=old.weighted_span(pp@axis,ww)
 if not np.isclose(original,r.r3_length_m,rtol=1e-8,atol=1e-6):
  raise RuntimeError(('replay mismatch',r.obs_id,r.pol,original,r.r3_length_m))
 mu=np.average(pp,weights=ww,axis=0);pc=pp-mu
 cov=(pc.T*ww)@pc/ww.sum();ev,evec=np.linalg.eigh(cov);paxis=evec[:,-1]
 row=dict(obs_id=r.obs_id,pol=r.pol,mmsi=r.mmsi,obb_span=original,
  intrinsic_span=old.weighted_span(pp@paxis,ww),radius_rms=float(np.sqrt(ev.sum())),
  response_area=float(use.sum()*area),eigen_major=float(np.sqrt(ev[-1])),
  axis_gap=float(np.degrees(np.arccos(np.clip(abs(axis@paxis),0,1)))),
  replay_abs_error=abs(original-r.r3_length_m))
 for deg in (5,10):
  vals=[]
  for sign in (-1,1):
   t=math.radians(sign*deg);rot=np.array([[math.cos(t),-math.sin(t)],[math.sin(t),math.cos(t)]])
   span=old.weighted_span(pp@(rot@axis),ww);vals.append(abs(math.log(span/original)))
  row['axis_'+str(deg)+'_abslog']=float(np.mean(vals))
 rows.append(row)
 if i%250==0: print('measured',i,'seconds',round(time.time()-start,1),flush=True)
new=pd.DataFrame(rows);print('MEASUREMENT',len(new),'max_replay_error',new.replay_abs_error.max(),flush=True)
for pol,g in new.groupby('pol'):
 print('AXIS_SENSITIVITY',pol,json.dumps(dict(n=len(g),
  median_abs_change_5deg_pct=float(100*np.expm1(g.axis_5_abslog.median())),
  median_abs_change_10deg_pct=float(100*np.expm1(g.axis_10_abslog.median())),
  p90_abs_change_10deg_pct=float(100*np.expm1(g.axis_10_abslog.quantile(.9))),
  median_detector_intrinsic_axis_gap=float(g.axis_gap.median()))),flush=True)
f=f.merge(new,on=['obs_id','pol','mmsi'],how='left',validate='one_to_one')
prior=pd.read_csv(R/'next_observation_20261021/predictions.csv',dtype={'mmsi':str})
oldpred=prior[(prior.variant=='all')&(prior.metric=='r3_length_m')&(prior.arm=='transfer_geometry')]
output=[]
for pol,full in f.groupby('pol'):
 qids=oldpred[oldpred.pol.eq(pol)].obs_id.tolist()
 for metric in ('obb_span','intrinsic_span','radius_rms','response_area'):
  v=full[full.r3_valid.eq(True)&full[metric].gt(0)&np.isfinite(full[metric])].copy()
  cache={}
  for oid in qids:
   q=v[v.obs_id.eq(oid)].iloc[0]
   hist=v[v.mmsi.eq(q.mmsi)&v.timestamp.lt(q.timestamp)]
   assert len(hist)>=3
   key=(int(q.split_fold),q.time,q.overpass)
   if key not in cache:
    src=v[v.split_fold.ne(q.split_fold)&v.timestamp.lt(q.timestamp)&v.overpass.ne(q.overpass)].copy()
    src=src.groupby('mmsi').filter(lambda x:len(x)>=2)
    assert q.mmsi not in set(src.mmsi)
    fitted={name:nxt.fit_relation(src,metric,nxt.ARMS[name],10) for name in ('transfer_metadata','transfer_direction','transfer_geometry')}
    cache[key]=fitted
   fitted=cache[key]
   predictions={'history_mean':float(np.log(hist[metric]).mean())}
   for name,beta in fitted.items():predictions[name]=nxt.predict(hist,q,metric,nxt.ARMS[name],beta)
   actual=float(np.log(q[metric]))
   for name,pred in predictions.items():
    output.append(dict(pol=pol,metric=metric,obs_id=int(oid),mmsi=q.mmsi,fold=int(q.split_fold),overpass=q.overpass,
     arm=name,pred_log=pred,actual_log=actual,error=abs(pred-actual)))
  print('fit',pol,metric,'queries',len(qids),flush=True)
out=pd.DataFrame(output)
rep=out[out.metric.eq('obb_span')&out.arm.eq('transfer_geometry')].merge(oldpred,on=['obs_id','pol'],suffixes=('_new','_old'))
print('PREDICTION_REPLAY_MAX',float(np.max(abs(rep.pred_log_new-rep.pred_log_old))),flush=True)
assert np.max(abs(rep.pred_log_new-rep.pred_log_old))<1e-8
summary=[]
rng=np.random.default_rng(20261024)
for (pol,metric),g in out.groupby(['pol','metric']):
 w=g.pivot(index=['obs_id','mmsi','fold','overpass'],columns='arm',values='error').reset_index()
 for baseline in ('history_mean','transfer_metadata'):
  for arm in ('transfer_direction','transfer_geometry'):
   gain=(w[baseline]-w[arm]).to_numpy();n=len(gain)
   boot=gain[rng.integers(n,size=(2000,n))].mean(1)
   groups=w.assign(gain=gain).groupby('overpass').agg(total=('gain','sum'),n=('gain','size'))
   draws=rng.integers(len(groups),size=(2000,len(groups)))
   cluster=groups.total.to_numpy()[draws].sum(1)/groups.n.to_numpy()[draws].sum(1)
   fg=w.assign(gain=gain).groupby('fold').gain.mean()
   s=dict(pol=pol,metric=metric,baseline=baseline,arm=arm,n=n,
     baseline_mae=float(w[baseline].mean()),model_mae=float(w[arm].mean()),
     relative_gain_pct=float(100*gain.mean()/w[baseline].mean()),
     improved=int((gain>0).sum()),positive_folds=int((fg>0).sum()),
     ship_ci=np.quantile(boot,[.025,.975]).tolist(),pass_ci=np.quantile(cluster,[.025,.975]).tolist())
   summary.append(s)
print('RESULTS_JSON',json.dumps(summary),flush=True)
# Fixed illustrations already reviewed, not selected by new outcomes.
cases=out[out.obs_id.isin([28992,383,25586,26309])&out.arm.isin(['history_mean','transfer_geometry'])]
print('FIXED_CASES',cases.to_json(orient='records'),flush=True)
print('SECONDS',time.time()-start,flush=True)

SAVE=Path(__file__).resolve().parent/'round02_axis'
SAVE.mkdir(parents=True,exist_ok=True)
new.to_csv(SAVE/'measurements.csv',index=False)
out.to_csv(SAVE/'predictions.csv',index=False)
(SAVE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
cases.to_csv(SAVE/'fixed_cases.csv',index=False)

