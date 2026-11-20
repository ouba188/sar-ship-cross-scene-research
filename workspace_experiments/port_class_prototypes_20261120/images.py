"""Native PNG evidence, AIS-matched controls, not physical backscatter claims."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr, wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from analyze import ROOT, WORK, STORE, csv, write

PAIRS=[('bulk_carrier','Melbourne','Port Hedland'),
       ('container','Port Said','New York'),
       ('chemical_tanker','Singapore','Houston'),
       ('general_cargo','Hamburg','Santos')]


def native(row,pol):
    view=row.get(pol)
    if view is None:
        return None
    with Image.open(view['image']) as im:
        image=np.asarray(im.convert('L')).copy()
    axis=view.get('ship_axis')
    if not axis:
        return None
    det=view['detections'][view['target_index']]
    yy,xx=np.mgrid[:image.shape[0],:image.shape[1]]
    cx,cy=axis['center']; theta=axis['angle']
    along=(xx-cx)*np.cos(theta)+(yy-cy)*np.sin(theta)
    across=-(xx-cx)*np.sin(theta)+(yy-cy)*np.cos(theta)
    length,width=axis['length_px'],axis['width_px']
    body=(abs(along)<=length/2)&(abs(across)<=width/2)
    nearby=(abs(along)<=length/2+8)&(abs(across)<=width/2+8)&~body
    if body.sum()<4 or nearby.sum()<4:
        return None
    x=image.astype(float)/255
    contrast=float(x[body].mean()-x[nearby].mean())
    profile=[]
    for a,b in zip(np.linspace(-.5,.5,7)[:-1],np.linspace(-.5,.5,7)[1:]):
        mask=body&(along>=a*length)&(along<b*length)
        profile.append(float(x[mask].mean()) if mask.any() else np.nan)
    confidence=float(det['confidence'])
    return dict(image=image,corners=det['corners'],axis=axis,profile=profile,
        metrics=dict(obb_length_px=float(length),obb_width_px=float(width),
                     contrast=contrast,body_mean=float(x[body].mean()),
                     outside_mean=float(x[nearby].mean()),saturation=float((x[body]>=.99).mean()),
                     confidence=confidence,shape_usable=bool(det.get('shape_usable',False)),
                     width_px=image.shape[1],height_px=image.shape[0]))


def representative(df,rows):
    # Select an observation without inspecting pixels or classifier outcomes.
    sorted_df=df.sort_values(['ais_size_status','product','index'],kind='stable')
    return sorted_df.drop_duplicates(['class_name','port','mmsi'])


def panel(items,rows,name,title,overlays=False):
    fig,axes=plt.subplots(2,len(items),figsize=(3.1*len(items),6.4),squeeze=False,layout='constrained')
    records=[]
    for col,r in enumerate(items):
        row=rows[int(r['index'])]
        paths={}
        for p,pol in enumerate(('vv','vh')):
            ax=axes[p,col]; view=native(row,pol)
            if view is None:
                ax.text(.5,.5,'Unavailable',ha='center');ax.set_axis_off();continue
            paths[pol]=row[pol]['image']
            h,w=view['image'].shape
            ax.imshow(view['image'],cmap='gray',vmin=0,vmax=255,interpolation='nearest',
                      extent=(-w/2,w/2,h/2,-h/2))
            if overlays:
                corners=np.asarray(view['corners'])-[w/2,h/2]
                ax.add_patch(Polygon(corners,fill=False,edgecolor='#10b981',linewidth=.7))
            ax.set_xlim(-64,64);ax.set_ylim(64,-64);ax.set_aspect('equal');ax.set_facecolor('#dddddd')
            ax.set_xticks([-64,0,64]);ax.set_yticks([-64,0,64]);ax.tick_params(labelsize=7)
            ax.set_title(f"{r['port']} / {pol.upper()}\nAIS {r['ais_length_m']:g} x {r['ais_width_m']:g} m",fontsize=10)
            ax.set_xlabel(f"{row['id']} / {h}x{w} px / OBB {view['metrics']['obb_length_px']:.1f}px",fontsize=8)
        records.append(dict(index=int(r['index']),id=row['id'],port=r['port'],mmsi=r['mmsi'],
            class_name=r['class_name'],length_m=float(r['ais_length_m']),beam_m=float(r['ais_width_m']),
            ais_size_status=r['ais_size_status'],paths=paths,product=row['product'],
            pixel_x_m_vv=None if pd.isna(r['pixel_x_m_vv']) else float(r['pixel_x_m_vv'])))
    fig.suptitle(title+'\nNative pixels, same 128-pixel canvas, fixed 0-255 display; gray is unused canvas',fontsize=12)
    fig.savefig(ROOT/f'{name}_local.png',dpi=150)
    plt.close(fig)
    return records


def main():
    rows=json.loads((STORE/'manifest.json').read_text(encoding='utf-8'))
    df=pd.read_csv(ROOT/'source_observations_local.csv',dtype={'mmsi':str})
    df=df[df.valid & df.plausible]
    candidates=representative(df,rows)
    all_metrics=[]; gallery=[]; matchrows=[]
    rng=np.random.default_rng(20261120)
    for name,a,b in PAIRS:
        sub=candidates[(candidates.class_name==name)&candidates.port.isin([a,b])]
        shared=set(sub[sub.port==a].mmsi)&set(sub[sub.port==b].mmsi)
        sub=sub[~sub.mmsi.isin(shared)]
        exemplars=[]
        measured={}
        for port in (a,b):
            pool=sub[sub.port==port]
            median=pool.ais_length_m.median()
            ids=(pool.ais_length_m-median).abs().sort_values(kind='stable').index[:3]
            exemplars.extend(pool.loc[ids].to_dict('records'))
            # Up to 80 independently sampled vessels per class/port; one original observation each.
            sample=pool.iloc[rng.permutation(len(pool))[:80]]
            measured[port]=sample
            for r in sample.to_dict('records'):
                for pol in ('vv','vh'):
                    view=native(rows[int(r['index'])],pol)
                    if view is not None:
                        all_metrics.append(dict(index=int(r['index']),mmsi=r['mmsi'],
                            class_name=name,port=port,pol=pol,length_m=r['ais_length_m'],beam_m=r['ais_width_m'],
                            quarter=r['quarter'],pixel_x_m=r[f'pixel_x_m_{pol}'],
                            pixel_y_m=r[f'pixel_y_m_{pol}'],scale_valid=r[f'scale_valid_{pol}'],
                            **view['metrics']))
        gallery.append(dict(kind='median_population',class_name=name,
            cases=panel(exemplars,rows,f'{name}_median',f'{name}: vessels closest to each port median AIS length')))
        # Greedy one-to-one match within 10% in both AIS dimensions, no image or outcome selection.
        left,right=[sub[sub.port==port].sort_values('index') for port in (a,b)]
        edges=[]
        for i,x in left.iterrows():
            for j,y in right.iterrows():
                delta=np.abs(np.log([x.ais_length_m/y.ais_length_m,x.ais_width_m/y.ais_width_m]))
                if delta.max()<=np.log(1.1):
                    edges.append((float(delta.sum()),int(i),int(j)))
        used=set();matches=[]
        for distance,i,j in sorted(edges):
            if i in used or j in used:continue
            used.update((i,j));matches.append((i,j,distance))
        selected=[]
        for rank,(i,j,distance) in enumerate(matches):
            matchrows.append(dict(class_name=name,pair_rank=rank,index_a=int(left.loc[i,'index']),
                index_b=int(right.loc[j,'index']),port_a=a,port_b=b,distance=distance))
            if rank<2:selected.extend([left.loc[i].to_dict(),right.loc[j].to_dict()])
        if selected:
            gallery.append(dict(kind='AIS_size_matched',class_name=name,
                pairs=len(matches),cases=panel(selected,rows,f'{name}_matched',
                    f'{name}: different vessels, cross-port AIS length AND beam matched (<=10%)')))
    metrics=pd.DataFrame(all_metrics)
    csv('native_metrics_local.csv',metrics)
    csv('ais_matched_pairs_local.csv',pd.DataFrame(matchrows))
    write('gallery_local.json',gallery)
    records=[]
    for keys,part in metrics.groupby(['class_name','port','pol']):
        rho,p=spearmanr(part.length_m,part.obb_length_px)
        records.append(dict(class_name=keys[0],port=keys[1],pol=keys[2],n=len(part),
            length_m_median=part.length_m.median(),obb_length_px_median=part.obb_length_px.median(),
            contrast_median=part.contrast.median(),outside_mean_median=part.outside_mean.median(),
            length_vs_obb_spearman=rho,scale_valid_n=int(part.scale_valid.sum()),
            verified_10m_n=int((part.scale_valid & np.isclose(part.pixel_x_m,10)&np.isclose(part.pixel_y_m,10)).sum()),
            low_confidence_n=int((part.confidence<.25).sum())))
    csv('native_metrics_summary.csv',pd.DataFrame(records))
    print(pd.DataFrame(records).to_string(index=False))
    print('size-matched independent pairs:',pd.DataFrame(matchrows).groupby('class_name').size().to_dict())


if __name__=='__main__':main()
