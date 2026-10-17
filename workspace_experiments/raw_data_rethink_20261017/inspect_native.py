"""Outcome-blind native-chip inspection before revising local relation design."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation, label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PREV=ROOT/'obb_consistent_repair_20261016'
TASKS=ROOT/'prepool_local_probe_20261015/body_context_expansion/Singapore'
sys.path.insert(0,str(PREV))
from extract_regions import masks_for


def native(meta,grow,pol):
    with Image.open(meta.loc[grow,'image_relpath_'+pol]) as im:
        return np.array(im.convert('L'))


def show(ax,meta,det,grow,pol,title):
    if grow is None:
        ax.axis('off'); ax.text(.5,.5,title,ha='center',va='center',transform=ax.transAxes,fontsize=8)
        return
    im=native(meta,grow,pol)
    ax.imshow(im,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    x0,y0,x1,y1=det[(int(grow),pol)]['fixed_box']
    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='lime',linewidth=.7))
    m=meta.loc[grow]
    ax.set_title(f'{title} | {pol.upper()} | row {grow}\n{m.class_name} | {im.shape[1]}px | {m.date}',fontsize=7)
    ax.set_xticks([]);ax.set_yticks([])


def select(meta):
    rng=np.random.default_rng(20261017)
    counts=meta.groupby('mmsi').agg(products=('product','nunique'),classes=('class_id','nunique'))
    eligible=counts.index[(counts.products>1)&(counts.classes==1)]
    pairs=[]
    for cls,g in meta.loc[meta.mmsi.isin(eligible)].groupby('class_id'):
        ships=sorted(g.mmsi.unique())
        mmsi=ships[int(rng.integers(len(ships)))]
        observations=g[g.mmsi==mmsi].sort_values(['date','product']).drop_duplicates('product')
        a,b=observations.iloc[0],observations.iloc[-1]
        pairs.append(dict(class_id=int(cls),mmsi=mmsi,first=int(a.name),second=int(b.name),
                          first_product=a['product'],second_product=b['product']))
    tm=pd.read_csv(TASKS/'metadata.csv',dtype={'mmsi':str})
    tasks=[t for t in json.loads((TASKS/'tasks.json').read_text()) if t['budget']==32 and t['draw']==0]
    qrows=sorted({int(tm.iloc[q].global_row) for t in tasks for q in t['q']})
    cases=[]; all_supports=[]
    for cls,g in meta.loc[qrows].groupby('class_id'):
        grow=int(rng.choice(g.index.to_numpy()))
        row=dict(class_id=int(cls),query=grow)
        for origin in ('same','cross'):
            task=next(t for t in tasks if t['origin']==origin and grow in tm.iloc[t['q']].global_row.to_list())
            supports=tm.iloc[task['s']].global_row.to_numpy(int)
            same=meta.loc[supports].query('class_id == @cls')
            assert not set(meta.loc[supports].mmsi)&{meta.loc[grow,'mmsi']}
            row[origin+'_support']=int(rng.choice(same.index.to_numpy())) if len(same) else None
            row[origin+'_sameclass_count']=len(same)
            row[origin+'_acquisition']=task['acquisition']
            all_supports.extend(dict(query=grow,origin=origin,support=int(s),same_class=bool(meta.loc[s,'class_id']==cls)) for s in supports)
        cases.append(row)
    pd.DataFrame(pairs).to_csv(HERE/'same_vessel_selection.csv',index=False)
    pd.DataFrame(cases).to_csv(HERE/'query_support_selection.csv',index=False)
    pd.DataFrame(all_supports).to_csv(HERE/'actual_support_sets.csv',index=False)
    return pairs,cases,dict(repeated_ships=len(eligible),repeated_ships_by_class=
        meta.loc[meta.mmsi.isin(eligible)].groupby('class_id').mmsi.nunique().to_dict(),query_rows=len(qrows))


def plots(meta,det,pairs,cases):
    for start in range(0,len(pairs),3):
        batch=pairs[start:start+3]
        fig,axes=plt.subplots(len(batch),4,figsize=(12,3.3*len(batch)),squeeze=False,layout='constrained')
        for i,p in enumerate(batch):
            for j,(key,pol) in enumerate((('first','vv'),('first','vh'),('second','vv'),('second','vh'))):
                show(axes[i,j],meta,det,p[key],pol,f"MMSI {p['mmsi']} | {key}")
        fig.suptitle('Same vessel, distinct products; native uint8, fixed 0-255; not physical radiometry',fontsize=10)
        fig.savefig(HERE/f'same_vessel_{start//3+1}.png',dpi=160);plt.close(fig)
    for start in range(0,len(cases),4):
        batch=cases[start:start+4]
        fig,axes=plt.subplots(len(batch),6,figsize=(17,3.0*len(batch)),squeeze=False,layout='constrained')
        for i,p in enumerate(batch):
            for j,(role,pol) in enumerate((('query','vv'),('query','vh'),('same_support','vv'),('same_support','vh'),('cross_support','vv'),('cross_support','vh'))):
                title=role.replace('_',' ')
                if role!='query':
                    title+=f" (same class n={p[role.split('_')[0]+'_sameclass_count']})"
                show(axes[i,j],meta,det,p[role],pol,title)
        fig.suptitle('Actual S32/draw0 tasks; random within-class selection before reading predictions',fontsize=11)
        fig.savefig(HERE/f'query_support_{start//4+1}.png',dpi=160);plt.close(fig)


def statistics(meta,det):
    rows=meta.index.to_numpy(int)
    masks=masks_for(rows,det,'fixed')
    coverage=[]; pixels=[]
    for idx,(grow,body) in enumerate(zip(rows,masks)):
        extended=np.pad(body,16)
        d=[extended]+[binary_dilation(extended,iterations=n) for n in (2,8,16)]
        regions=[d[0],d[1]&~d[0],d[2]&~d[1],d[3]&~d[2]]
        cropped=[x[16:-16,16:-16] for x in regions]
        for name,region,observed in zip(('body','touch','near','far'),regions,cropped):
            coverage.append(dict(global_row=int(grow),region=name,observed_pixels=int(observed.sum()),
                full_pixels=int(region.sum()),fraction_observed=float(observed.sum()/region.sum())))
        for pol in ('vv','vh'):
            im=native(meta,grow,pol); h,w=im.shape
            iy=np.minimum(((np.arange(h)+.5)/h*56).astype(int),55)
            ix=np.minimum(((np.arange(w)+.5)/w*56).astype(int),55)
            native_regions=[x[iy[:,None],ix[None,:]] for x in cropped]
            bright=im>=200
            components,_=label(bright,structure=np.ones((3,3),int))
            linked=np.unique(components[native_regions[0]&bright])
            linked=linked[linked!=0]
            attached=np.isin(components,linked)&bright
            for name,mask in zip(('body','touch','near','far'),native_regions):
                val=im[mask].astype(float)
                pixels.append(dict(global_row=int(grow),pol=pol,region=name,pixels=len(val),
                    mean=float(val.mean()) if len(val) else np.nan,
                    std=float(val.std()) if len(val) else np.nan,
                    q90=float(np.quantile(val,.9)) if len(val) else np.nan,
                    bright_pixels=int((bright&mask).sum()),bright_attached_to_body=int((attached&mask).sum()),
                    saturated_pixels=int(((im==255)&mask).sum())))
        if idx%500==0:
            print('NATIVE',idx,'/',len(rows),flush=True)
    cov=pd.DataFrame(coverage);pix=pd.DataFrame(pixels)
    cov.to_csv(HERE/'region_coverage.csv',index=False)
    pix.to_csv(HERE/'native_region_statistics.csv',index=False)
    roles=pd.read_csv(PREV/'roles.csv')
    cr=cov.merge(roles,on='global_row')
    cr.groupby(['split','role','region']).fraction_observed.agg(
        count='count',mean='mean',median='median',
        fraction_truncated=lambda s:float((s<.999).mean()),
        fraction_below_half=lambda s:float((s<.5).mean())).to_csv(HERE/'coverage_by_role.csv')
    pr=pix.merge(roles,on='global_row')
    pr.groupby(['split','role','region'])[['pixels','bright_pixels','bright_attached_to_body','saturated_pixels']].sum().to_csv(HERE/'display_brightness_by_role.csv')
    return cov,pix


def region_plot(meta,det,cov):
    target=pd.read_csv(PREV/'roles.csv').query("split == 'target' and role == 'q'").global_row.unique()
    far=cov[cov.region.eq('far')&cov.global_row.isin(target)].sort_values(['fraction_observed','global_row'])
    # Coverage quantiles select geometry cases, not classification outcomes.
    selected=far.iloc[np.linspace(0,len(far)-1,4).round().astype(int)].global_row.to_numpy(int)
    records=[]
    fig,axes=plt.subplots(4,3,figsize=(11,13),layout='constrained')
    for i,grow in enumerate(selected):
        body=masks_for([grow],det,'fixed')[0]
        d2,d8,d16=[binary_dilation(body,iterations=n) for n in (2,8,16)]
        region=np.zeros((56,56),int)
        for j,m in enumerate((d16,d8,d2,body),1):
            region[m]=j
        for j,pol in enumerate(('vv','vh')):
            show(axes[i,j],meta,det,grow,pol,'Coverage quantile')
        axes[i,2].imshow(region,cmap='viridis',vmin=0,vmax=4,interpolation='nearest')
        row=cov[(cov.global_row==grow)&cov.region.eq('far')].iloc[0]
        axes[i,2].set_title(f'Actual 56x56 masks\nFar region observed: {100*row.fraction_observed:.1f}%',fontsize=9)
        axes[i,2].set_xticks([]);axes[i,2].set_yticks([])
        records.append(dict(global_row=int(grow),far_fraction=float(row.fraction_observed)))
    fig.suptitle('Image boundary truncates nominal neighborhoods; colors show distance zones, not semantics',fontsize=10)
    fig.savefig(HERE/'region_coverage_examples.png',dpi=150);plt.close(fig)
    pd.DataFrame(records).to_csv(HERE/'coverage_visual_selection.csv',index=False)


def main():
    meta=pd.read_csv(PREV/'inventory.csv',dtype={'mmsi':str}).set_index('global_row')
    meta['mmsi']=meta.mmsi.str.replace(r'\.0$','',regex=True)
    meta['class_name']=meta.image_relpath_vv.map(lambda p:Path(p).parent.name)
    meta['date']=meta['product'].str.split('_').str[4].str[:8]
    det={(r['global_row'],r['pol']):r for r in map(json.loads,(PREV/'detections.jsonl').read_text().splitlines())}
    pairs,cases,summary=select(meta)
    plots(meta,det,pairs,cases)
    cov,pix=statistics(meta,det)
    region_plot(meta,det,cov)
    summary.update(rows=len(meta),native_polarization_images=2*len(meta),
        selection_seed=20261017,outcome_based_selection=False,trained_models=0,
        threshold=200,threshold_meaning='Fixed uint8 display threshold; not calibrated backscatter',
        mask_meaning='Current 56-grid region membership mapped to native pixel centers',
        figure_rule='Native arrays, fixed 0-255, nearest display; no per-panel rescaling',
        same_vessel_pairs=len(pairs),query_support_cases=len(cases))
    (HERE/'INSPECTION_DONE.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
