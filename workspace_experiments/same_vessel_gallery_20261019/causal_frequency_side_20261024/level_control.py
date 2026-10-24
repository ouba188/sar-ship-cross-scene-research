"""Match each directional intervention's mean-level shift without retraining."""
import os
os.environ['MPLCONFIGDIR']=str(__import__('pathlib').Path(__file__).resolve().parent/'round11_level_control/.mplcache')
import json
import time
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr
import directional_intervention as d
import joint_probe as j

ROOT=d.ROOT
OUT=ROOT/'round11_level_control'


def level_image(im,view,region,mean_delta):
    masks=d.ip.masks_for(view);sh=view['priors']['ship_axis']
    end=j.masks_for(im,sh,sh['length_px'],sh['width_px'])['end']&~(masks[0]+masks[1]).astype(bool)
    mask=masks[0].astype(bool) if region=='body' else end
    taper=np.minimum(distance_transform_edt(mask)/2,1)*mask
    basis=taper*((1-im) if mean_delta>=0 else im)
    strength=mean_delta*im.size/basis.sum() if basis.any() else 0.
    assert abs(strength)<=1,(region,strength)
    out=im+strength*basis
    assert out.min()>=-1e-12 and out.max()<=1+1e-12
    assert abs((out-im).mean()-mean_delta)<1e-12
    assert np.max(abs((out-im)[~mask]))==0
    return out


def bootstrap(g,col,group,rng):
    means=g.groupby(group)[col].mean().to_numpy()
    draws=rng.integers(len(means),size=(2000,len(means)))
    boot=means[draws].mean(1)
    return float(means.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975))


def analyze(table,frame):
    table['residual_effect']=table.delta_fixed-table.level_delta_fixed
    keys=['pol','region','direction','dose']
    table.groupby(keys).agg(n=('obs_id','size'),smooth_delta=('delta_fixed','mean'),
        level_delta=('level_delta_fixed','mean'),difference=('residual_effect','mean'),
        absolute_smooth=('delta_fixed',lambda a:abs(a).mean()),
        absolute_level=('level_delta_fixed',lambda a:abs(a).mean()),
        level_l2=('level_l2','mean'),smooth_l2=('input_l2','mean')).to_csv(OUT/'summary.csv')
    table['overpass']=table['product'].map(lambda x:'_'.join([x.split('_')[0],x.split('_')[6],x.split('_')[7]]))
    wide=table.pivot(index=['obs_id','mmsi','overpass','pol','region','dose','class_id'],columns='direction',values='residual_effect').reset_index()
    rng=np.random.default_rng(20261024);rows=[];groups=[]
    for (pol,region,dose),g in wide.groupby(['pol','region','dose']):
        for axis in ('ship','radar'):
            z=g.assign(difference=g[axis]-g[axis+'_perp'])
            row=dict(pol=pol,region=region,dose=dose,axis=axis,n=len(z))
            for group in ('mmsi','overpass'):
                mean,lo,hi=bootstrap(z,'difference',group,rng)
                row.update({group+'_mean':mean,group+'_lo':lo,group+'_hi':hi})
            rows.append(row)
            for c,h in z.groupby('class_id'):
                groups.append(dict(pol=pol,region=region,dose=dose,axis=axis,class_id=c,n=len(h),
                    mean=h.difference.mean(),ships=h.mmsi.nunique()))
    pd.DataFrame(rows).to_csv(OUT/'contrasts.csv',index=False)
    pd.DataFrame(groups).to_csv(OUT/'contrasts_by_class.csv',index=False)
    old=pd.read_csv(ROOT/'round08_boundary_reference/pairs.csv')
    old=old[old.region.eq('body')&old.scale_m.eq(10)]
    stat=table[table.direction.eq('ship')&table.dose.eq(1.)]
    p=[]
    for (pol,region),g in stat.groupby(['pol','region']):
        lookup=g.set_index('obs_id')
        for r in old[old.pol.eq(pol)].itertuples():
            a,z=lookup.loc[r.history_id],lookup.loc[r.query_id]
            p.append(dict(query_id=r.query_id,history_id=r.history_id,mmsi=r.mmsi,pol=pol,region=region,
                overpass=r.query_pass,class_id=r.class_id,
                delta_log_ratio=z.log_fine_coarse_before-a.log_fine_coarse_before,
                delta_margin=z.baseline_margin-a.baseline_margin,
                delta_sensitivity=z.residual_effect-a.residual_effect,
                delta_relative_cos=np.cos(2*z.relative_axis)-np.cos(2*a.relative_axis),
                delta_incidence=z.incidence-a.incidence))
    pairs=pd.DataFrame(p);pairs.to_csv(OUT/'natural_pairs.csv',index=False)
    coverage=[]
    for (pol,region),g in pairs.groupby(['pol','region']):
        measured=table[table.pol.eq(pol)&table.region.eq(region)&table.dose.eq(1)].groupby('obs_id').delta_log_ratio.apply(lambda x:abs(x).max())
        reached=np.maximum(g.query_id.map(measured),g.history_id.map(measured))
        coverage.append(dict(pol=pol,region=region,n=len(g),natural_median=abs(g.delta_log_ratio).median(),
            intervention_median=np.median(reached),within_magnitude=int((abs(g.delta_log_ratio)<=reached).sum())))
    pd.DataFrame(coverage).to_csv(OUT/'amplitude_coverage.csv',index=False)
    corr=[]
    for (pol,region),g in pairs.groupby(['pol','region']):
        for x,y in [('delta_log_ratio','delta_margin'),('delta_relative_cos','delta_log_ratio'),
                    ('delta_incidence','delta_log_ratio'),('delta_log_ratio','delta_sensitivity')]:
            corr.append(dict(pol=pol,region=region,x=x,y=y,n=len(g),rho=float(spearmanr(g[x],g[y]).statistic)))
    pd.DataFrame(corr).to_csv(OUT/'natural_associations.csv',index=False)
    return table


def panels(table,frame,views):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Select median-strength cases in each observed intervention outcome, not extremes.
    h=table[table.dose.eq(1)].copy()
    h['state']=np.where(h.baseline_prediction.eq(h.class_id),'C','W')+np.where(h.prediction.eq(h.class_id),'C','W')
    selection=[]
    for state,g in h.groupby('state'):
        median=abs(g.delta_fixed).median();r=g.loc[(abs(g.delta_fixed)-median).abs().idxmin()]
        selection.append(r.to_dict())
    selected=pd.DataFrame(selection);selected.to_csv(OUT/'case_selection.csv',index=False)
    for r in selected.itertuples():
        i=frame.index[frame.obs_id.eq(r.obs_id)][0];row=frame.loc[i]
        fig,axes=plt.subplots(2,4,figsize=(12,6),constrained_layout=True)
        for k,pol in enumerate(('vv','vh')):
            view=views[i]['images'][pol];im=np.asarray(d.ip.Image.open(row['image_'+pol]).convert('L'),dtype=float)/255
            z=table[table.obs_id.eq(r.obs_id)&table.pol.eq(pol)&table.region.eq(r.region)&table.direction.eq(r.direction)&table.dose.eq(1)].iloc[0]
            items,_=d.perturbations(im,view)
            altered=next(x for x,m in items if m['region']==r.region and m['direction']==r.direction and m['dose']==1)
            level=level_image(im,view,r.region,z.mean_delta)
            for col,(title,img) in enumerate([('native',im),('directional',altered),('mean-matched',level),('directional - native',altered-im)]):
                ax=axes[k,col];limit=max(abs(altered-im).max(),1e-10)
                ax.imshow(img,cmap='gray' if col<3 else 'RdBu_r',vmin=0 if col<3 else -limit,vmax=1 if col<3 else limit,interpolation='nearest')
                ax.contour(d.ip.masks_for(view)[0],levels=[.5],colors='lime',linewidths=.5)
                ax.set_title(f'{pol.upper()} {title}',fontsize=10)
            axes[k,1].set_xlabel(f'fixed margin change {z.delta_fixed:+.5f}')
            axes[k,2].set_xlabel(f'fixed margin change {z.level_delta_fixed:+.5f}')
        fig.suptitle(f'ID {r.obs_id}, class {r.class_id}, selected {r.pol.upper()} {r.state}; {r.region}/{r.direction}\n'
                     'Frozen full_global diagnostic; native pixels, no image resizing')
        fig.savefig(OUT/f'case_{r.obs_id}_{r.state}.png',dpi=125);plt.close(fig)
    prior=pd.read_csv(ROOT/'round07_joint_coverage/review_selection.csv')
    table[table.obs_id.isin(prior.query_id)&table.dose.eq(1)].to_csv(OUT/'previous_case_effects.csv',index=False)


def main():
    OUT.mkdir(exist_ok=True);started=time.perf_counter();d.torch.set_num_threads(2)
    original=pd.read_csv(ROOT/'round10_directional_intervention/interventions.csv',dtype={'mmsi':str})
    frame,views,disabled=d.ip.load_data();cache=np.load(d.CONTROL/'features.npz')['full_global']
    heads=[np.load(d.CONTROL/f'head_f{k}_full_global.npz') for k in range(3)]
    model=d.resnet50(weights=None);model.load_state_dict(d.torch.load(d.ip.WEIGHTS,map_location='cpu',weights_only=True))
    stem=d.torch.nn.Sequential(model.conv1,model.bn1,model.relu).eval();del model
    records=[]
    with d.torch.inference_mode():
        for count,((oid,pol),g) in enumerate(original.groupby(['obs_id','pol'])):
            i=frame.index[frame.obs_id.eq(oid)][0];row=frame.loc[i];view=views[i]['images'][pol]
            im=np.asarray(d.ip.Image.open(row['image_'+pol]).convert('L'),dtype=float)/255
            changed=[level_image(im,view,r.region,r.mean_delta) for r in g.itertuples()]
            desc=d.ip.describe(stem,*d.prepare_batch(changed,view),inputs=('full',))['full_global'].numpy()
            head=heads[int(row.fold)];base=d.score(cache[i],head)
            raw=np.tile(cache[i],(len(g),1));polidx=('vv','vh').index(pol);raw[:,polidx*128:(polidx+1)*128]=desc
            scores=d.score(raw,head)
            for r,altered,s in zip(g.itertuples(),changed,scores):
                records.append(dict(obs_id=oid,pol=pol,region=r.region,direction=r.direction,dose=r.dose,
                    level_delta_fixed=float(s[r.class_id]-s[r.fixed_rival]-(base[r.class_id]-base[r.fixed_rival])),
                    level_l2=float(np.linalg.norm(altered-im)),level_prediction=int(s.argmax())))
            if count%80==0:print('level views',count+1,'seconds',round(time.perf_counter()-started,1),flush=True)
    keys=['obs_id','pol','region','direction','dose']
    table=original.merge(pd.DataFrame(records),on=keys,validate='one_to_one')
    table=analyze(table,frame);table.to_csv(OUT/'interventions.csv',index=False)
    panels(table,frame,views)
    (OUT/'manifest.json').write_text(json.dumps(dict(observations=table.obs_id.nunique(),views=len(table[['obs_id','pol']].drop_duplicates()),
        ships=table.mmsi.nunique(),matched_interventions=len(table),seconds=time.perf_counter()-started,
        mean_matching=True,l2_matching=False,posthoc_followup=True,physical_causal_effect=False),indent=2),encoding='utf8')
    print(pd.read_csv(OUT/'contrasts.csv').query('dose==1').to_string(index=False))
    print(pd.read_csv(OUT/'natural_associations.csv').to_string(index=False))


if __name__=='__main__':main()
