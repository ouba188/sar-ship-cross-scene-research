"""Factor image change and measurement-angle change before interpreting geometry."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='2'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import joint_probe as j

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'round12_region_counterfactual'


def values(im,p,angle,length,width):
    sh=dict(p['ship_axis'],angle=angle)
    regions=j.masks_for(im,sh,length,width)
    yy,xx=np.indices(im.shape);r=np.hypot(xx+.5-sh['center'][0],yy+.5-sh['center'][1])*10
    regions.update(disk60=r<=60,ring120_200=(r>=120)&(r<200))
    valid=(xx>=6)&(xx<im.shape[1]-6)&(yy>=6)&(yy<im.shape[0]-6)
    g1,g2=j.gradients(im,1),j.gradients(im,2);result={}
    for name,mask in regions.items():
        use=mask&valid
        if use.sum()<12 or use.sum()<.9*mask.sum():continue
        a,b=[np.trace(j.tensor(*g,use)) for g in (g1,g2)]
        if min(a,b)>1e-15:result[name]=float(np.log(a/b))
    return result


def analyze(rows):
    result=[]
    for (pol,region),g in rows.groupby(['pol','region']):
        for label,h in [('all',g),('within_pass_pair',g.groupby('pass_pair').filter(lambda z:len(z)>=3))]:
            if h.empty:continue
            cols=['delta_cos','total_delta','image_delta','mask_delta','margin_delta']
            z=h[cols].copy()
            if label=='within_pass_pair':z=z-h.groupby('pass_pair')[cols].transform('mean')
            for y in ('total_delta','image_delta','mask_delta'):
                rho=spearmanr(z.delta_cos,z[y]).statistic if z[y].std()>1e-12 else np.nan
                result.append(dict(pol=pol,region=region,stratum=label,outcome=y,n=len(h),
                    ships=h.mmsi.nunique(),pass_pairs=h.pass_pair.nunique(),rho=float(rho),
                    mean_absolute=float(abs(h[y]).mean()),
                    margin_rho=float(spearmanr(z[y],z.margin_delta).statistic) if z[y].std()>1e-12 else np.nan))
    pd.DataFrame(result).to_csv(OUT/'summary.csv',index=False)


def panels(data,frame):
    os.environ['MPLCONFIGDIR']=str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=pd.read_csv(ROOT/'round07_joint_coverage/review_selection.csv')
    for rec in selected.itertuples():
        fig,axes=plt.subplots(2,2,figsize=(8,7),constrained_layout=True)
        for row,pol in enumerate(('vv','vh')):
            items=[j.image_data(i,pol,data,frame) for i in (rec.history_id,rec.query_id)]
            length=np.mean([p['ship_axis']['length_px'] for _,p,_ in items]);width=np.mean([p['ship_axis']['width_px'] for _,p,_ in items])
            angles=[p['ship_axis']['angle'] for _,p,_ in items]
            for col,(im,p,_) in enumerate(items):
                ax=axes[row,col];ax.imshow(im,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                for angle,color in zip(angles,('cyan','orange')):
                    sh=dict(p['ship_axis'],angle=angle)
                    m=j.masks_for(im,sh,length,width)
                    ax.contour(m['body'],levels=[.5],colors=color,linewidths=.8)
                    ax.contour(m['end'],levels=[.5],colors=color,linewidths=.6,linestyles='dashed')
                ax.set_title(f'{(rec.history_id,rec.query_id)[col]} {pol.upper()}')
        fig.suptitle('Same image, different readout masks\nCyan: historical axis; orange: current axis; paired mean size')
        fig.savefig(OUT/f'case_{rec.query_id}.png',dpi=125);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True);data,frame=j.load()
    old=pd.read_csv(ROOT/'round08_boundary_reference/pairs.csv',dtype={'mmsi':str})
    old=old[old.region.eq('body')&old.scale_m.eq(10)]
    preds=pd.read_csv(ROOT.parent/'crude_input_controls_20261021/predictions.csv')
    preds=preds[preds.arm.eq('full_global')].set_index('obs_id')
    rows=[];maxerror=0.
    for rec in old.itertuples():
        items=[j.image_data(i,rec.pol,data,frame) for i in (rec.history_id,rec.query_id)]
        length=np.mean([p['ship_axis']['length_px'] for _,p,_ in items]);width=np.mean([p['ship_axis']['width_px'] for _,p,_ in items])
        angles=[p['ship_axis']['angle'] for _,p,_ in items]
        f={(i,k):values(im,p,angle,length,width) for i,(im,p,_) in enumerate(items) for k,angle in enumerate(angles)}
        common=set.intersection(*(set(a) for a in f.values()))
        cos=[np.cos(2*(p['ship_axis']['angle']-p['radar_axis'])) for _,p,_ in items]
        for region in sorted(common):
            a,b,c,d=[f[key][region] for key in [(0,0),(0,1),(1,0),(1,1)]]
            image=.5*((d-b)+(c-a));mask=.5*((d-c)+(b-a))
            maxerror=max(maxerror,abs(d-a-image-mask))
            rows.append(dict(query_id=rec.query_id,history_id=rec.history_id,mmsi=rec.mmsi,pol=rec.pol,region=region,
                pass_pair=rec.history_pass+'__'+rec.query_pass,class_id=rec.class_id,
                f00=a,f01=b,f10=c,f11=d,total_delta=d-a,image_delta=image,mask_delta=mask,
                delta_cos=cos[1]-cos[0],margin_delta=preds.loc[rec.query_id,'margin']-preds.loc[rec.history_id,'margin']))
    assert maxerror<1e-12,maxerror
    out=pd.DataFrame(rows);out.to_csv(OUT/'pairs.csv',index=False);analyze(out);panels(data,frame)
    (OUT/'manifest.json').write_text(json.dumps(dict(pair_views=len(old),identity_error=maxerror,
        physical_causal_effect=False,new_training=False),indent=2),encoding='utf8')
    print(pd.read_csv(OUT/'summary.csv').query('region in ["end","disk60","ring120_200"]').to_string(index=False))


if __name__=='__main__':main()
