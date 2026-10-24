"""Match a measured natural scale-ratio difference without choosing by labels."""
import os
import json
import time
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import directional_intervention as d
import level_control as l
import joint_probe as j

ROOT=d.ROOT
OUT=ROOT/'round13_natural_amplitude'


def ratio(im,mask):
    energies=[np.trace(j.tensor(*j.gradients(im,s),mask)) for s in (1,2)]
    return float(np.log(energies[0]/energies[1]))


def match(im,view,direction,target):
    body=d.ip.masks_for(view)[0].astype(bool)
    p=view['priors'];angle=p['ship_axis']['angle'] if direction.startswith('ship') else p['radar_axis']
    if direction.endswith('perp'):angle+=np.pi/2
    taper=np.minimum(distance_transform_edt(body)/2,1)*body
    delta=taper*(d.directional_smooth(im,angle)-im)
    start=ratio(im,body);prev_weight=0.;prev=start-target
    for weight in np.linspace(0,.9,17)[1:]:
        current=ratio(im+weight*delta,body)-target
        if current*prev<=0:
            lo,hi=prev_weight,weight
            for _ in range(12):
                mid=(lo+hi)/2
                if (ratio(im+mid*delta,body)-target)*prev>0:lo=mid
                else:hi=mid
            weight=(lo+hi)/2;altered=im+weight*delta
            return altered,dict(reachable=True,weight=weight,achieved=ratio(altered,body),target=target,start=start,
                                input_l2=float(np.linalg.norm(altered-im)),mean_delta=float((altered-im).mean()))
        prev_weight=weight;prev=current
    return None,dict(reachable=False,weight=.9,achieved=ratio(im+.9*delta,body),target=target,start=start)


def analyze(table):
    keys=['pol','direction']
    table['bridge_gain']=abs(table.natural_delta)-abs(table.natural_delta-table.delta_fixed)
    table['level_gain']=abs(table.natural_delta)-abs(table.natural_delta-table.level_delta)
    count=table.groupby(['query_id','history_id','pol']).direction.transform('nunique')
    table['common_four']=count.eq(4)
    summaries=[]
    for common in (False,True):
        z=table[table.common_four] if common else table
        for (pol,direction),g in z.groupby(keys):
            summaries.append(dict(pol=pol,direction=direction,common_four=common,n=len(g),ships=g.mmsi.nunique(),
                mean_weight=g.weight.mean(),mean_natural_abs=abs(g.natural_delta).mean(),mean_gain=g.bridge_gain.mean(),
                improved_fraction=(g.bridge_gain>0).mean(),mean_level_gain=g.level_gain.mean(),
                mean_gain_over_level=(g.bridge_gain-g.level_gain).mean(),mean_abs_input_l2=g.input_l2.mean(),
                max_ratio_error=abs(g.achieved-g.target).max()))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    return table


def panels(table,frame,views):
    os.environ['MPLCONFIGDIR']=str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selection=[]
    for pol in ('vv','vh'):
        z=table[table.pol.eq(pol)&table.direction.eq('ship')]
        for label,h in [('closer',z[z.bridge_gain>0]),('farther',z[z.bridge_gain<0])]:
            if h.empty:continue
            r=h.loc[(h.bridge_gain-h.bridge_gain.median()).abs().idxmin()]
            selection.append(dict(r,selection=label))
    pd.DataFrame(selection).to_csv(OUT/'case_selection.csv',index=False)
    for rec in selection:
        fig,axes=plt.subplots(2,4,figsize=(12,6),constrained_layout=True)
        for k,pol in enumerate(('vv','vh')):
            ih=frame.index[frame.obs_id.eq(rec['high_id'])][0];il=frame.index[frame.obs_id.eq(rec['low_id'])][0]
            view=views[ih]['images'][pol]
            im=np.asarray(d.ip.Image.open(frame.loc[ih,'image_'+pol]).convert('L'),dtype=float)/255
            low=np.asarray(d.ip.Image.open(frame.loc[il,'image_'+pol]).convert('L'),dtype=float)/255
            target=ratio(low,d.ip.masks_for(views[il]['images'][pol])[0].astype(bool))
            altered,meta=match(im,view,rec['direction'],target) if ratio(im,d.ip.masks_for(view)[0].astype(bool))>target else (None,{})
            imgs=[im,low,altered,(altered-im) if altered is not None else None]
            for col,(title,img) in enumerate(zip(['high endpoint','real other endpoint','ratio-matched edit','edit - high'],imgs)):
                ax=axes[k,col];ax.set_title(pol.upper()+' '+title,fontsize=9)
                if img is None:ax.text(.5,.5,'Not reachable / reverse order',ha='center',va='center',wrap=True);ax.set_axis_off();continue
                lim=max(abs(img).max(),1e-10)
                ax.imshow(img,cmap='gray' if col<3 else 'RdBu_r',vmin=0 if col<3 else -lim,vmax=1 if col<3 else lim,interpolation='nearest')
        fig.suptitle(f"{rec['high_id']} -> {rec['low_id']} | {rec['pol']} {rec['selection']}\n"
                     f"Natural margin change {rec['natural_delta']:+.4f}; edit {rec['delta_fixed']:+.4f}; brightness {rec['level_delta']:+.4f}")
        fig.savefig(OUT/f"case_{rec['high_id']}_{rec['pol']}_{rec['selection']}.png",dpi=125);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True);started=time.perf_counter();d.torch.set_num_threads(2)
    original=pd.read_csv(ROOT/'round10_directional_intervention/interventions.csv')
    stats=original[original.region.eq('body')&original.direction.eq('ship')&original.dose.eq(1)].set_index(['obs_id','pol'])
    pairs=pd.read_csv(ROOT/'round08_boundary_reference/pairs.csv',dtype={'mmsi':str})
    pairs=pairs[pairs.region.eq('body')&pairs.scale_m.eq(10)]
    frame,views,_=d.ip.load_data();loc={oid:i for i,oid in enumerate(frame.obs_id)}
    cache=np.load(d.CONTROL/'features.npz')['full_global'];heads=[np.load(d.CONTROL/f'head_f{k}_full_global.npz') for k in range(3)]
    model=d.resnet50(weights=None);model.load_state_dict(d.torch.load(d.ip.WEIGHTS,map_location='cpu',weights_only=True))
    stem=d.torch.nn.Sequential(model.conv1,model.bn1,model.relu).eval();del model
    rows=[];coverage=[]
    with d.torch.inference_mode():
        for n,rec in enumerate(pairs.itertuples()):
            high,low=sorted((rec.history_id,rec.query_id),key=lambda oid:stats.loc[(oid,rec.pol),'log_fine_coarse_before'],reverse=True)
            ih,il=loc[high],loc[low];row=frame.iloc[ih];view=views[ih]['images'][rec.pol]
            assert frame.iloc[ih].fold==frame.iloc[il].fold
            im=np.asarray(d.ip.Image.open(row['image_'+rec.pol]).convert('L'),dtype=float)/255
            target=stats.loc[(low,rec.pol),'log_fine_coarse_before'];head=heads[int(row.fold)]
            base=d.score(cache[ih],head);real=d.score(cache[il],head);y=int(row.class_id)
            wrong=base.copy();wrong[y]=-np.inf;rival=int(wrong.argmax())
            natural=float(real[y]-real[rival]-(base[y]-base[rival]))
            images=[];metadata=[]
            for direction in ('ship','ship_perp','radar','radar_perp'):
                altered,meta=match(im,view,direction,target)
                info=dict(query_id=rec.query_id,history_id=rec.history_id,mmsi=rec.mmsi,pol=rec.pol,
                    class_id=y,high_id=high,low_id=low,direction=direction,query_pass=rec.query_pass,**meta)
                coverage.append(info)
                if altered is None:continue
                images.extend([altered,l.level_image(im,view,'body',meta['mean_delta'])]);metadata.append(info)
            if images:
                desc=d.ip.describe(stem,*d.prepare_batch(images,view),inputs=('full',))['full_global'].numpy()
                raw=np.tile(cache[ih],(len(images),1));p=('vv','vh').index(rec.pol);raw[:,128*p:128*(p+1)]=desc
                scores=d.score(raw,head)
                for k,info in enumerate(metadata):
                    a,b=scores[2*k:2*k+2]
                    rows.append(dict(info,natural_delta=natural,baseline_margin=float(base[y]-base[rival]),fixed_rival=rival,
                        delta_fixed=float(a[y]-a[rival]-(base[y]-base[rival])),
                        level_delta=float(b[y]-b[rival]-(base[y]-base[rival]))))
            if n%40==0:print('amplitude pairs',n+1,'seconds',round(time.perf_counter()-started,1),flush=True)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    out=analyze(pd.DataFrame(rows));out.to_csv(OUT/'interventions.csv',index=False);panels(out,frame,views)
    (OUT/'manifest.json').write_text(json.dumps(dict(pair_views=len(pairs),directions_requested=len(coverage),
        reachable=len(out),seconds=time.perf_counter()-started,physical_causal_effect=False,new_training=False),indent=2),encoding='utf8')
    print(pd.read_csv(OUT/'summary.csv').to_string(index=False))


if __name__=='__main__':main()
