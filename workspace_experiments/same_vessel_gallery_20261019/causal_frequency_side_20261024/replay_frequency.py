import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import json,importlib.util,time
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image
R=Path('E:/临时会话/same_vessel_gallery_20261019')
sp=importlib.util.spec_from_file_location('nxt',R/'next_observation_20261021/run.py')
nxt=importlib.util.module_from_spec(sp);sp.loader.exec_module(nxt)
f,_=nxt.load()
data=json.loads((R/'gallery_data.js').read_text(encoding='utf8').split('=',1)[1].strip().removesuffix(';'))['observations']
rows=[];t=time.time()
def spectrum(im,points,shape,a,region):
 rad=np.linalg.norm(points,axis=1).reshape(shape)
 if region=='core':w=np.where(rad<180,.5*(1+np.cos(np.pi*np.minimum(rad,180)/180)),0)
 else:w=np.where((rad>220)&(rad<300),np.sin(np.pi*np.clip((rad-220)/80,0,1))**2,0)
 mean=(im*w).sum()/w.sum();z=(im-mean)*w
 power=abs(np.fft.fft2(z))**2
 fy,fx=np.meshgrid(np.fft.fftfreq(shape[0]),np.fft.fftfreq(shape[1]),indexing='ij')
 kv=np.column_stack([fx.ravel(),fy.ravel()])@np.linalg.inv(a)
 freq=np.linalg.norm(kv,axis=1).reshape(shape)
 use=(freq>1/600)&(freq<=.05)
 pp=power[use];ff=freq[use]
 return float((pp*ff).sum()/pp.sum()),float(pp[ff>=.025].sum()/pp.sum())
checks=[]
for r in f[f.r3_valid.eq(True)].itertuples():
 v=data[str(r.obs_id)]['images'][r.pol];p=v['priors']
 im=np.asarray(Image.open(r.image_path).convert('L'),float)
 a=np.asarray(p['scale']['transform'])[[1,2,4,5]].reshape(2,2)
 yy,xx=np.indices(im.shape)
 pts=(np.column_stack([xx.ravel()+.5,yy.ravel()+.5])-np.asarray(p['radar_center']))@a.T
 row=dict(obs_id=r.obs_id,pol=r.pol)
 for region in ('core','outer'):
  cen,hi=spectrum(im,pts,im.shape,a,region)
  row[region+'_centroid']=cen;row[region+'_high']=hi
  if len(checks)<8:
   cc,hh=spectrum(im*1.7+23,pts,im.shape,a,region)
   checks.append(max(abs(cc-cen),abs(hh-hi)))
 rows.append(row)
print('EXTRACTED',len(rows),'affine_intensity_invariance_error',max(checks),'seconds',time.time()-t,flush=True)
f=f.merge(pd.DataFrame(rows),on=['obs_id','pol'],how='left',validate='one_to_one')
f['log_length']=np.log(f.r3_length_m);f['log_width']=np.log(f.r3_width_m)
f['log_area']=np.log(f.r3_area_m2)
spatial=['log_length','log_width','log_area','contrast','target_saturated','far_mean','far_sd']
arms={'metadata':nxt.META,'geometry':nxt.META+nxt.GEOM,
      'spatial':nxt.META+spatial,'spatial_geometry':nxt.META+spatial+nxt.GEOM}
prior=pd.read_csv(R/'next_observation_20261021/predictions.csv')
qprior=prior[(prior.variant=='all')&(prior.metric=='r3_length_m')&(prior.arm=='transfer_geometry')]
out=[]
for pol,full in f.groupby('pol'):
 for metric in ('core_centroid','core_high','outer_centroid','outer_high'):
  valid=full[full.r3_valid.eq(True)&full[metric].gt(0)].dropna(subset=spatial).copy();cache={}
  for oid in qprior[qprior.pol.eq(pol)].obs_id:
   if not valid.obs_id.eq(oid).any():continue
   q=valid[valid.obs_id.eq(oid)].iloc[0]
   hist=valid[valid.mmsi.eq(q.mmsi)&valid.timestamp.lt(q.timestamp)]
   if len(hist)<3:continue
   key=(q.split_fold,q.time,q.overpass)
   if key not in cache:
    src=valid[valid.split_fold.ne(q.split_fold)&valid.timestamp.lt(q.timestamp)&valid.overpass.ne(q.overpass)]
    src=src.groupby('mmsi').filter(lambda g:len(g)>=2)
    cache[key]={k:nxt.fit_relation(src,metric,cols,10) for k,cols in arms.items()}
   actual=np.log(q[metric]);preds={'history':np.log(hist[metric]).mean()}
   for k,b in cache[key].items():preds[k]=nxt.predict(hist,q,metric,arms[k],b)
   out.append(dict(pol=pol,metric=metric,obs_id=int(oid),mmsi=q.mmsi,fold=int(q.split_fold),overpass=q.overpass,actual=q[metric],
       history_value=float(np.exp(preds['history'])),geometry_value=float(np.exp(preds['geometry'])),
       **{k:abs(v-actual) for k,v in preds.items()}))
  print('FITTED',pol,metric,flush=True)
out=pd.DataFrame(out)
summ=[]
rng=np.random.default_rng(20261024)
for (pol,metric),g in out.groupby(['pol','metric']):
 for base,arm in [('history','geometry'),('metadata','geometry'),('spatial','spatial_geometry')]:
  gain=g[base]-g[arm];gr=g.assign(gain=gain).groupby('overpass').agg(s=('gain','sum'),n=('gain','size'))
  d=rng.integers(len(gr),size=(2000,len(gr)));boot=gr.s.to_numpy()[d].sum(1)/gr.n.to_numpy()[d].sum(1)
  summ.append(dict(pol=pol,metric=metric,n=len(g),baseline=base,arm=arm,baseline_mae=g[base].mean(),model_mae=g[arm].mean(),
  reduction_pct=100*gain.mean()/g[base].mean(),positive_folds=int((g.assign(gain=gain).groupby('fold').gain.mean()>0).sum()),
  pass_ci=np.quantile(boot,[.025,.975]).tolist()))
print('SUMMARY',json.dumps(summ),flush=True)
print('FIXED_CASES',out[out.obs_id.isin([28992,383,25586,26309])].to_json(orient='records'),flush=True)
print('ALL_CASE_VALUES',pd.DataFrame(rows)[pd.DataFrame(rows).obs_id.isin([6273,6275,29682,29683,28992,383,25586,26309])].to_json(orient='records'),flush=True)
print('TIME',time.time()-t,flush=True)

SAVE=Path(__file__).resolve().parent/'round04_frequency'
SAVE.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(SAVE/'measurements.csv',index=False)
out.to_csv(SAVE/'predictions.csv',index=False)
pd.DataFrame(summ).to_csv(SAVE/'summary.csv',index=False)
(SAVE/'summary.json').write_text(json.dumps(summ,indent=2),encoding='utf8')
f[['obs_id','pol','mmsi','image_path','class_id','port','time','overpass','split_fold','r3_valid']].to_csv(SAVE/'cohort.csv',index=False)

