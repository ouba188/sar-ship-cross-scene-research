"""Exact input-source decomposition at unchanged spatial readout locations."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joint_probe as j
import boundary_reference as b

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'round09_fixed_support'


def fields(im, body, anchor):
    g = np.stack(j.gradients(im, 1))
    gb = np.stack(j.gradients(im * body, 1))
    if anchor == 'local':
        gb -= im[None] * np.stack(j.gradients(body.astype(float), 1))
    return g, gb, g-gb


def measure(oid, pol, data, frame):
    im, p, rec = j.image_data(oid, pol, data, frame)
    masks, valid, *_ = b.region_masks(im, p, rec, 1)
    rows, tensors, errors = [], {}, {}
    for anchor in ('zero', 'local'):
        g, gb, go = fields(im, masks['body'], anchor)
        errors['sum'] = float(abs(g-gb-go).max())
        if anchor == 'local':
            shifted = fields(im+.2, masks['body'], anchor)
            errors['offset'] = max(float(abs(a-c).max()) for a,c in zip((g,gb,go),shifted))
        for region in ('body', 'end', 'side'):
            m = masks[region]&valid
            tt, tb, to = [j.tensor(*v, m) for v in (g,gb,go)]
            cross = tt-tb-to
            total = np.trace(tt)
            rows.append(dict(obs_id=oid,pol=pol,anchor=anchor,region=region,n=int(m.sum()),
                body_ratio=float(np.trace(tb)/total),outside_ratio=float(np.trace(to)/total),
                cross_ratio=float(np.trace(cross)/total),energy=float(total),
                body_anisotropy=float(np.diff(np.linalg.eigvalsh(tb))[0]/max(np.trace(tb),1e-15)),
                outside_anisotropy=float(np.diff(np.linalg.eigvalsh(to))[0]/max(np.trace(to),1e-15))))
            for source, t in [('total',tt),('body',tb),('outside',to)]:
                tensors[(anchor,region,source)] = t
    return rows, tensors, p, errors


def panels(data,frame):
    os.environ['MPLCONFIGDIR']=str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=pd.read_csv(ROOT/'round07_joint_coverage/review_selection.csv')
    for rec in selected.itertuples():
        fig,axes=plt.subplots(2,6,figsize=(17,6),constrained_layout=True)
        for row,pol in enumerate(('vv','vh')):
            objects=[]
            for oid in (rec.history_id,rec.query_id):
                im,p,r=j.image_data(oid,pol,data,frame)
                masks,valid,*_=b.region_masks(im,p,r,1)
                _,gb,go=fields(im,masks['body'],'local')
                objects.append((oid,im,masks,np.sum(gb**2,0),np.sum(go**2,0)))
            vmax=max(np.quantile(z,.99) for o in objects for z in o[3:])
            for col,(oid,im,masks,eb,eo) in enumerate(objects):
                for k,img in enumerate((im,eb,eo)):
                    ax=axes[row,col*3+k]
                    ax.imshow(img,cmap='gray' if k==0 else 'magma',vmin=0,vmax=1 if k==0 else vmax,interpolation='nearest')
                    ax.contour(masks['body'],levels=[.5],colors='cyan',linewidths=.6)
                    ax.contour(masks['end'],levels=[.5],colors='lime',linewidths=.5)
                    ax.set_title(f'{oid} {pol.upper()}\n'+('native','body-input contrast','outside-input contrast')[k],fontsize=9)
                    ax.set_aspect('equal')
        fig.suptitle(f'Fixed positions: {rec.history_id}->{rec.query_id}; old classifier {rec.state}\n'
                     'Contributions are operator bookkeeping, not physical scattering sources')
        fig.savefig(OUT/f'case_{rec.query_id}.png',dpi=120);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True);data,frame=j.load()
    source=pd.read_csv(ROOT/'round08_boundary_reference/pairs.csv',dtype={'mmsi':str})
    requests=source[source.region.eq('body')&source.scale_m.eq(10)]
    cache={};measurements=[];pairs=[];checks={}
    for rec in requests.itertuples():
        objects=[]
        for oid in (rec.history_id,rec.query_id):
            key=(oid,rec.pol)
            if key not in cache:
                rows,t,p,err=measure(oid,rec.pol,data,frame)
                cache[key]=(t,p);measurements+=rows
                for k,v in err.items(): checks[k]=max(checks.get(k,0),v)
            objects.append(cache[key])
        (t0,p0),(t1,p1)=objects
        angles={'image':0,'ship':p1['ship_axis']['angle']-p0['ship_axis']['angle'],
                'radar':p1['radar_axis']-p0['radar_axis']}
        for (anchor,region,part),a in t0.items():
            z=t1[(anchor,region,part)]
            row=dict(query_id=rec.query_id,history_id=rec.history_id,mmsi=rec.mmsi,pol=rec.pol,
                anchor=anchor,region=region,part=part,query_pass=rec.query_pass,
                class_id=rec.class_id,state=rec.state)
            for name,angle in angles.items():
                rot=j.rotation(angle)
                row[name+'_error']=float(np.linalg.norm(z/np.trace(z)-rot@(a/np.trace(a))@rot.T))
            pairs.append(row)
    m=pd.DataFrame(measurements);q=pd.DataFrame(pairs)
    replay=q[q.anchor.eq('local')&q.part.eq('total')].merge(
        source[source.scale_m.eq(10)&source.region.isin(['body','end','side'])],
        on=['query_id','pol','region'],suffixes=('_new','_old'),validate='one_to_one')
    checks['round08_replay']=max(float(abs(replay[a+'_error_new']-replay[a+'_error_old']).max()) for a in ('image','ship','radar'))
    assert max(checks.values())<1e-12,checks
    m.to_csv(OUT/'measurements.csv',index=False);q.to_csv(OUT/'pairs.csv',index=False)
    m.groupby(['pol','anchor','region']).agg(n=('obs_id','size'),body_ratio=('body_ratio','median'),
        outside_ratio=('outside_ratio','median'),cross_ratio=('cross_ratio','median')).to_csv(OUT/'source_summary.csv')
    result=[];pass_rows=[]
    for (pol,anchor,region,part),g in q.groupby(['pol','anchor','region','part']):
        gain=g.image_error-g.radar_error
        h=g.assign(gain=gain).groupby('query_pass').gain.mean()
        result.append(dict(pol=pol,anchor=anchor,region=region,part=part,n=len(g),
            image_error=g.image_error.mean(),ship_error=g.ship_error.mean(),radar_error=g.radar_error.mean(),
            radar_gain=gain.mean(),pass_mean_gain=h.mean(),positive_passes=int((h>0).sum()),passes=len(h)))
        pass_rows.extend(dict(pol=pol,anchor=anchor,region=region,part=part,query_pass=k,gain=v) for k,v in h.items())
    pd.DataFrame(result).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(pass_rows).to_csv(OUT/'pass_gains.csv',index=False)
    panels(data,frame)
    (OUT/'manifest.json').write_text(json.dumps(dict(pair_views=len(requests),ships=requests.mmsi.nunique(),
        checks=checks,physical_causal_effect=False,neural_training=False),indent=2),encoding='utf8')
    print(pd.DataFrame(result).query('anchor=="local" and region=="end"').to_string(index=False))
    print(m.groupby(['pol','anchor','region'])[['body_ratio','outside_ratio','cross_ratio']].median().to_string())


if __name__=='__main__':main()
