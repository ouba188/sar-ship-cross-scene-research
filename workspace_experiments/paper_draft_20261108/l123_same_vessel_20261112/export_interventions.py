"""Recover eight exact historical pixel interventions without model inference."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, distance_transform_edt

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
OUT=ROOT/'expanded'; OUT.mkdir(exist_ok=True)
AS=OUT/'l2_images'; AS.mkdir(exist_ok=True)
H=WORK/'paper_evidence_20261027/full_model_interventions_20261027'
G=WORK/'same_vessel_gallery_20261019'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
loader=module('cpu_loader',ROOT.parent/'l123_editable_20261112/prepare_l2.py')
native=module('native_pixels',WORK/'sar_m123_standalone/sar_m123/input.py')
cpu=loader.cpu_functions
body=cpu(G/'local_condition_module/prepare.py',['masks_for'],dict(np=np,Image=Image,ImageDraw=ImageDraw))
axis=cpu(G/'causal_frequency_side_20261024/joint_probe.py',['masks_for'],dict(np=np))
probe=cpu(G/'causal_frequency_side_20261024/classification_joint_probe.py',['regions','bands'],dict(
    np=np,gaussian_filter=gaussian_filter,d=SimpleNamespace(ip=SimpleNamespace(masks_for=body['masks_for'])),j=SimpleNamespace(masks_for=axis['masks_for'])))
ops=cpu(H/'run_interventions.py',['changes_for'],dict(np=np,distance_transform_edt=distance_transform_edt,probe=SimpleNamespace(**probe)))
snapshot=next(x for x in json.loads((G/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if x['obs_id']==12457)
obs=pd.read_csv(ROOT/'ports24/l1_observations.csv').set_index('obs_id').loc[12457]
source=[snapshot['images'][p] for p in ('vv','vh')]
views=[]
for p,s in zip(('vv','vh'),source):
    pr=s['priors']
    views.append(native.prepare_view(dict(image=obs['image_relpath_'+p],detections=pr['detections'],target_index=pr['target_index'])))
changes,excluded=ops['changes_for'](views,source)
assert len(changes)==8 and not excluded
saved=json.loads((H/'predictions/12457.json').read_text(encoding='utf-8'))['variants']
z=np.load(H/'predictions/12457.npz')
scores=z['score_17_full']; y=int(z['class_id'])
def margin(v): return float(v[y]-np.max(np.delete(v,y)))
base=margin(scores[0]); records=[]; arrays={}; maxerr=0
limit=.02
actualmax=max(float(abs(a-views[p]['raw']/255).max()) for p,a,_,_ in changes)
if actualmax>limit: limit=float(np.ceil(actualmax*100)/100)
def save_gray(name,a):
    Image.fromarray(np.rint(np.clip(a,0,1)*65535).astype(np.uint16)).save(AS/f'{name}_16bit.png')
    Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).save(AS/f'{name}.png')
    return str(AS/f'{name}.png')
for p,altered,level,meta in changes:
    old=next(r for r in saved if r.get('kind')=='structure' and all(r.get(k)==meta[k] for k in ('pol','region','band')))
    ctl=next(r for r in saved if r.get('kind')=='mean_control' and all(r.get(k)==meta[k] for k in ('pol','region','band')))
    errors={k:abs(float(meta[k])-float(old[k])) for k in ('pixels','l2','level_l2','mean_delta')}
    maxerr=max(maxerr,max(errors.values())); assert max(errors.values())<1e-12,errors
    original=views[p]['raw']/255.; delta=altered-original
    key='_'.join(meta[k] for k in ('pol','region','band'))
    mask=probe['regions'](original,source[p])[meta['region']]
    paths={k:save_gray(key+'_'+k,a) for k,a in [('original',original),('after',altered),('control',level)]}
    rgb=np.repeat(original[...,None],3,axis=-1)
    rgb[mask]=.62*rgb[mask]+.38*np.array([.05,.85,.65])
    paths['region']=str(AS/f'{key}_region.png'); Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(paths['region'])
    signed=delta/limit
    diff=np.stack((1+np.minimum(signed,0),1-abs(signed),1-np.maximum(signed,0)),axis=-1)
    paths['difference']=str(AS/f'{key}_difference.png'); Image.fromarray(np.rint(diff*255).astype(np.uint8)).save(paths['difference'])
    arrays.update({key+'_original':original,key+'_after':altered,key+'_control':level,key+'_difference':delta,key+'_mask':mask})
    v,c=int(old['variant']),int(ctl['variant'])
    records.append(dict(**meta,variant=v,control_variant=c,base_margin=base,after_margin=margin(scores[v]),
        delta_margin=margin(scores[v])-base,control_margin=margin(scores[c]),control_delta=margin(scores[c])-base,
        base_pred=int(scores[0].argmax()),after_pred=int(scores[v].argmax()),control_pred=int(scores[c].argmax()),
        changed_pixels=int(np.count_nonzero(delta)),difference_min=float(delta.min()),difference_max=float(delta.max()),
        errors=errors,paths=paths,shape=list(original.shape)))
np.savez_compressed(OUT/'l2_exact_pixels.npz',**arrays)
info=dict(obs_id=12457,class_id=y,model='20261027 complete M1-M3 seed17',source_results=str(H),
    reference_count=16,records=records,max_reconstruction_error=maxerr,difference_limit=limit,actual_max=actualmax,
    display='Original/after/control: fixed [0,1], no individual contrast stretch; uint8 display and uint16 alternatives. Exact float arrays saved in NPZ. Difference is after-original with shared signed scale.',
    intervention='Historical changes_for(): single polarization, body/side, fine/middle; all eight operations share within-query pixel L2. Other polarization unchanged.',
    note='Historical full-route scores include recomputed frequency descriptors; not a new frequency-free model test. All eight conditions remain correct.')
(OUT/'l2_assets.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(operations=len(records),max_reconstruction_error=maxerr,difference_limit=limit,actual_max=actualmax),indent=2))
