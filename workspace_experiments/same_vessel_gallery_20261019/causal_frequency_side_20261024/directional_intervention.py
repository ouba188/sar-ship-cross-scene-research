"""Matched local directional smoothing against an existing frozen diagnostic head."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='2'
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates, distance_transform_edt
import torch
from torchvision.models import resnet50
import joint_probe as j

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'round10_directional_intervention'
CONTROL=ROOT.parent/'crude_input_controls_20261021'
sys.path.insert(0,str(CONTROL))
import input_controls as ip


def directional_smooth(im,angle):
    yy,xx=np.indices(im.shape,dtype=float)
    values=np.arange(-3,4,dtype=float);weights=np.exp(-values**2/2);weights/=weights.sum()
    result=np.zeros_like(im,dtype=float)
    for step,weight in zip(values,weights):
        result+=weight*map_coordinates(im,[yy+step*np.sin(angle),xx+step*np.cos(angle)],order=1,mode='reflect',prefilter=False)
    return result


def perturbations(im,view):
    masks=ip.masks_for(view);sh=view['priors']['ship_axis']
    region=j.masks_for(im,sh,sh['length_px'],sh['width_px'])
    areas={'body':masks[0].astype(bool),'end':region['end']&~(masks[0]+masks[1]).astype(bool)}
    radar=view['priors']['radar_axis']
    angles={'ship':sh['angle'],'ship_perp':sh['angle']+np.pi/2,'radar':radar,'radar_perp':radar+np.pi/2}
    deltas={}
    for name,mask in areas.items():
        taper=np.minimum(distance_transform_edt(mask)/2,1)*mask
        for direction,angle in angles.items():
            deltas[(name,direction)]=taper*(directional_smooth(im,angle)-im)
    norms={key:float(np.linalg.norm(x)) for key,x in deltas.items()}
    target=.5*min(norms.values())
    if target<1e-10:return [],dict(zero=True,target=target)
    items=[]
    for (region,direction),delta in deltas.items():
        mask=areas[region];g=j.tensor(*j.gradients(im,1),mask);e2=np.trace(j.tensor(*j.gradients(im,2),mask))
        for dose in (.5,1.):
            strength=dose*target/norms[(region,direction)]
            altered=im+strength*delta
            g2=j.tensor(*j.gradients(altered,1),mask);e22=np.trace(j.tensor(*j.gradients(altered,2),mask))
            outside_error=float(abs((altered-im)[~mask]).max())
            assert outside_error==0 and altered.min()>=-1e-12 and altered.max()<=1+1e-12
            actual=float(np.linalg.norm(altered-im))
            assert abs(actual-dose*target)<1e-10
            items.append((altered,dict(region=region,direction=direction,dose=dose,weight=strength,
                input_l2=actual,global_rms=actual/np.sqrt(im.size),region_pixels=int(mask.sum()),
                mean_delta=float((altered-im).mean()),outside_error=outside_error,
                log_fine_coarse_before=float(np.log(np.trace(g)/e2)),
                log_fine_coarse_after=float(np.log(np.trace(g2)/e22)),
                directional_share_before=j.share(g,angles[direction]),
                directional_share_after=j.share(g2,angles[direction]))))
    return items,dict(zero=False,target=target)


def prepare_batch(images,view):
    masks=ip.masks_for(view);xs=[];ts=[];vs=[]
    for im in images:
        x,m,place=ip.crop_pad(im.astype(np.float32),masks,160)
        assert place['native_target_pixels']==place['retained_target_pixels']
        xs.append(x[None]);ts.append(m[:1]);vs.append(m.sum(0)[None])
    return [torch.from_numpy(np.stack(z)) for z in (xs,ts,vs)]


def score(raw,head):
    z=(raw-head['mean'])/head['std']/np.sqrt(raw.shape[-1])
    return z@head['coef'].T+head['intercept']


def summarize(table):
    keys=['region','pol','direction','dose']
    table.groupby(keys).agg(n=('obs_id','size'),ships=('mmsi','nunique'),
        mean_fixed_delta=('delta_fixed','mean'),median_fixed_delta=('delta_fixed','median'),
        mean_abs_delta=('delta_fixed',lambda a:abs(a).mean()),
        positive_fraction=('delta_fixed',lambda a:(a>0).mean()),
        rescued=('rescued','sum'),harmed=('harmed','sum'),
        mean_rms=('global_rms','mean'),mean_log_ratio_change=('delta_log_ratio','mean')).to_csv(OUT/'summary.csv')
    table.groupby(keys+['class_id']).agg(n=('obs_id','size'),ships=('mmsi','nunique'),
        mean_fixed_delta=('delta_fixed','mean'),mean_abs_delta=('delta_fixed',lambda a:abs(a).mean()),
        rescued=('rescued','sum'),harmed=('harmed','sum')).to_csv(OUT/'by_class.csv')
    wide=table.pivot(index=['obs_id','mmsi','product','pol','region','dose','class_id'],columns='direction',values='delta_fixed').reset_index()
    result=[];rng=np.random.default_rng(20261024)
    for (pol,region,dose),g in wide.groupby(['pol','region','dose']):
        for axis in ('ship','radar'):
            effect=g[axis]-g[axis+'_perp']
            ship=g.assign(effect=effect).groupby('mmsi').effect.mean()
            draws=rng.integers(len(ship),size=(2000,len(ship)))
            boot=ship.to_numpy()[draws].mean(1)
            result.append(dict(pol=pol,region=region,dose=dose,axis=axis,n=len(g),ships=len(ship),
                mean_difference=effect.mean(),median_abs_difference=abs(effect).median(),
                ship_mean=ship.mean(),ship_ci_low=np.quantile(boot,.025),ship_ci_high=np.quantile(boot,.975)))
    pd.DataFrame(result).to_csv(OUT/'orthogonal_contrasts.csv',index=False)
    table.groupby(keys+['product']).delta_fixed.agg(['count','mean']).to_csv(OUT/'by_product.csv')
    keys=['obs_id','pol','region','direction']
    d=table.pivot(index=keys,columns='dose',values='delta_fixed')
    d['same_sign']=d[.5]*d[1.]>0;d['high_larger']=abs(d[1.])>=abs(d[.5])
    d.to_csv(OUT/'dose_pairs.csv')
    return dict(dose_sign_consistency=float(d.same_sign.mean()),dose_magnitude_monotonic=float(d.high_larger.mean()))


def main():
    OUT.mkdir(exist_ok=True);started=time.perf_counter();torch.set_num_threads(2)
    frame,views,disabled=ip.load_data()
    pairs=pd.read_csv(ROOT/'round08_boundary_reference/pairs.csv')
    ids=sorted(set(pairs.history_id)|set(pairs.query_id))
    keep=frame[frame.obs_id.isin(ids)]
    cache=np.load(CONTROL/'features.npz')['full_global']
    old=pd.read_csv(CONTROL/'predictions.csv');old=old[old.arm.eq('full_global')].set_index('obs_id')
    heads=[np.load(CONTROL/f'head_f{k}_full_global.npz') for k in range(3)]
    model=resnet50(weights=None);model.load_state_dict(torch.load(ip.WEIGHTS,map_location='cpu',weights_only=True))
    stem=torch.nn.Sequential(model.conv1,model.bn1,model.relu).eval();del model
    checks=dict(feature_replay=0.,margin_replay=0.,constant_filter=0.);rows=[];skipped=[]
    checks['constant_filter']=float(abs(directional_smooth(np.ones((31,31)),.37)-1).max())
    assert checks['constant_filter']<1e-12
    with torch.inference_mode():
        for pos,(i,row) in enumerate(keep.iterrows()):
            head=heads[int(row.fold)];base=score(cache[i],head);y=int(row.class_id)
            wrong=base.copy();wrong[y]=-np.inf;rival=int(wrong.argmax())
            base_margin=float(base[y]-base[rival]);base_pred=int(base.argmax())
            checks['margin_replay']=max(checks['margin_replay'],abs(base_margin-old.loc[row.obs_id,'margin']))
            for polidx,pol in enumerate(('vv','vh')):
                view=views[i]['images'][pol]
                if polidx==1 and disabled[i]:
                    skipped.append(dict(obs_id=int(row.obs_id),pol=pol,reason='existing_crossview_disabled'));continue
                im=np.asarray(ip.Image.open(row['image_'+pol]).convert('L'),dtype=float)/255
                items,status=perturbations(im,view)
                if status['zero']:
                    skipped.append(dict(obs_id=int(row.obs_id),pol=pol,reason='zero_matched_change'));continue
                batch=prepare_batch([im]+[a for a,_ in items],view)
                desc=ip.describe(stem,*batch,inputs=('full',))['full_global'].numpy()
                sl=slice(polidx*128,(polidx+1)*128)
                err=float(abs(desc[0]-cache[i,sl]).max());checks['feature_replay']=max(checks['feature_replay'],err)
                assert err<2e-5,(row.obs_id,pol,err)
                raw=np.tile(cache[i],(len(items),1));raw[:,sl]=desc[1:]
                scores=score(raw,head)
                for (altered,meta),s in zip(items,scores):
                    pred=int(s.argmax());other=s.copy();other[y]=-np.inf
                    rows.append(dict(obs_id=int(row.obs_id),mmsi=row.mmsi,fold=int(row.fold),class_id=y,
                        product=row['product'],port=row.port,pol=pol,**meta,
                        relative_axis=float(view['priors']['ship_axis']['angle']-view['priors']['radar_axis']),
                        incidence=float(view['priors']['radar_incidence']),
                        baseline_prediction=base_pred,prediction=pred,baseline_margin=base_margin,
                        fixed_rival=rival,delta_fixed=float(s[y]-s[rival]-base_margin),
                        margin=float(s[y]-other.max()),rescued=bool(base_pred!=y and pred==y),
                        harmed=bool(base_pred==y and pred!=y),
                        delta_log_ratio=meta['log_fine_coarse_after']-meta['log_fine_coarse_before']))
            if pos%25==0:print('processed',pos+1,'/',len(keep),'seconds',round(time.perf_counter()-started,1),flush=True)
    assert checks['margin_replay']<1e-5,checks
    out=pd.DataFrame(rows);out.to_csv(OUT/'interventions.csv',index=False)
    pd.DataFrame(skipped,columns=['obs_id','pol','reason']).to_csv(OUT/'skipped.csv',index=False)
    dose=summarize(out)
    manifest=dict(observations=len(keep),ships=keep.mmsi.nunique(),interventions=len(out),
        views=len(out[['obs_id','pol']].drop_duplicates()),checks=checks,**dose,
        seconds=time.perf_counter()-started,classifier='frozen full_global C4 stem diagnostic, not main model',
        new_training=False,physical_causal_effect=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps(manifest),flush=True)
    print(pd.read_csv(OUT/'orthogonal_contrasts.csv').query('dose==1').to_string(index=False),flush=True)


if __name__=='__main__':main()
