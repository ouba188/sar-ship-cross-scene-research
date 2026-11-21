"""Native OSS2 evidence: common OBB, six longitudinal bins, no resizing or training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from analyze import ROOT,WORK,MANIFEST,REFS,SEED,csv,write

ROWS=json.loads(MANIFEST.read_text(encoding='utf-8'))
LOOK={r['id']:r for r in ROWS}

def read(row,pol):
    path=row.get('source_paths',{}).get(pol)
    if not path or not Path(path).is_file():return None
    image=tifffile.imread(path)
    if image.ndim!=2 or image.dtype!=np.uint8:
        raise ValueError(f'Expected original grayscale uint8 chip: {path}')
    return image

def masks(row,shape):
    # Same VV-derived box and signed coordinate system for both polarizations.
    # This removes inter-polar detector/axis flips as an explanation of profile differences.
    view=row.get('vv') or row.get('vh')
    if not view or not view.get('ship_axis'):return None
    axis=view['ship_axis'];cx,cy=axis['center'];theta=axis['angle']
    yy,xx=np.mgrid[:shape[0],:shape[1]]
    along=(xx-cx)*np.cos(theta)+(yy-cy)*np.sin(theta)
    across=-(xx-cx)*np.sin(theta)+(yy-cy)*np.cos(theta)
    length,width=axis['length_px'],axis['width_px']
    body=(abs(along)<=length/2)&(abs(across)<=width/2)
    ring=(abs(along)<=length/2+8)&(abs(across)<=width/2+8)&~body
    bins=[body&(along>=a*length)&(along<b*length) for a,b in zip(np.linspace(-.5,.5,7)[:-1],np.linspace(-.5,.5,7)[1:])]
    det=view['detections'][view['target_index']]
    return body,ring,bins,axis,det

def features(row):
    ims={p:read(row,p) for p in ['vv','vh']}
    first=next((x for x in ims.values() if x is not None),None)
    if first is None:return [],None
    ms=masks(row,first.shape)
    if ms is None:return [],None
    body,ring,bins,axis,det=ms
    if body.sum()<12 or ring.sum()<12 or min(x.sum() for x in bins)<2:return [],None
    out=[];distributions={}
    for pol,im in ims.items():
        if im is None or im.shape!=first.shape:continue
        x=im.astype(float)/255;bg=float(np.median(x[ring]));excess=np.maximum(x-bg,0)
        mass=np.array([excess[b].sum() for b in bins]);means=np.array([x[b].mean() for b in bins])
        q=mass/mass.sum() if mass.sum()>0 else np.full(6,np.nan)
        distributions[pol]=q
        out.append(dict(pol=pol,confidence=det['confidence'],length_px=axis['length_px'],width_px=axis['width_px'],body_pixels=int(body.sum()),min_bin_pixels=int(min(b.sum() for b in bins)),body_mean=x[body].mean(),ring_mean=x[ring].mean(),contrast=x[body].mean()-x[ring].mean(),saturation=(x[body]>=.99).mean(),top_bin=int(np.argmax(q)),top2_mass=np.sort(q)[-2:].sum(),entropy=-np.sum(q*np.log(q+1e-12))/np.log(6),**{f'q{i}':q[i] for i in range(6)},**{f'mean{i}':means[i] for i in range(6)}))
    joint=None
    if len(distributions)==2:
        a,b=distributions['vv'],distributions['vh']
        if np.isfinite([a,b]).all():
            joint=dict(vv_vh_tv=float(abs(a-b).sum()/2),top_bin_diff=bool(a.argmax()!=b.argmax()),top_bin_separation=int(abs(a.argmax()-b.argmax())),vv_top2=float(np.sort(a)[-2:].sum()),vh_top2=float(np.sort(b)[-2:].sum()),confidence=det['confidence'],length_px=axis['length_px'])
    return out,joint

def tv(a,b):return float(abs(a-b).sum()/2)

def panel(items,name,title,overlays=True):
    fig,axs=plt.subplots(2,len(items),figsize=(3.15*len(items),6.5),layout='constrained',squeeze=False)
    record=[]
    for c,item in enumerate(items):
        row=LOOK[item['sample_id']]
        rec={'sample_id':row['id'],'mmsi':row['mmsi'],'role':row['role'],'scene_id':row['scene_id'],'paths':row['source_paths'],'selection':item.get('selection','metadata'),'port_proxy':item.get('port_proxy'),'ais_length_m':item.get('ais_length_m'),'ais_width_m':item.get('ais_width_m')}
        for p,pol in enumerate(['vv','vh']):
            ax=axs[p,c];im=read(row,pol)
            if im is None:ax.text(.5,.5,'Unavailable',ha='center');ax.set_axis_off();continue
            h,w=im.shape
            ax.imshow(im,cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=(-w/2,w/2,h/2,-h/2))
            if overlays:
                ms=masks(row,im.shape)
                if ms is not None:ax.add_patch(Polygon(np.array(ms[4]['corners'])-[w/2,h/2],fill=False,edgecolor='#67BDA4',lw=.65))
            ax.set_xlim(-80,80);ax.set_ylim(80,-80);ax.set_aspect('equal');ax.set_facecolor('#dddddd');ax.set_xticks([-80,0,80]);ax.set_yticks([-80,0,80]);ax.tick_params(labelsize=7)
            length=item.get('ais_length_m',np.nan);beam=item.get('ais_width_m',np.nan)
            availability='' if row.get(pol) else ' [raw only; model missing]'
            ax.set_title(f"{item.get('port_proxy','reference')} / {pol.upper()}{availability}\nAIS {length:g} x {beam:g} m",fontsize=9)
            ax.set_xlabel(f"{row['id'].split('_')[-1]} / {w}x{h}px\n{item.get('note','')}",fontsize=8)
            # For selected original panels, verify that the actually consumed PNG preserves source pixels.
            v=row.get(pol)
            if v:
                with Image.open(v['image']) as pi:raw_png=np.asarray(pi.convert('L'))
                rec[pol+'_png_exact']=bool(np.array_equal(im,raw_png))
        record.append(rec)
    fig.suptitle(title+'\nNative pixels, fixed 0-255 range and 160-pixel canvas; outline is detected target, not hull truth',fontsize=11)
    fig.savefig(ROOT/f'{name}_local.png',dpi=145);plt.close(fig)
    return {'name':name,'title':title,'cases':record}

def main():
    s=pd.read_csv(ROOT/'source_observations_local.csv',dtype={'mmsi':str})
    s=s[s.valid & ~s.heldout_identity_overlap & (s.port_proxy!='outside_150km')]
    # Choose first chronological observation in each vessel/area, independently of pixels/predictions.
    reps=s.sort_values(['sar_time_utc','sample_id']).drop_duplicates(['class_name','port_proxy','mmsi'])
    metric_path=ROOT/'native_metrics_local.csv'
    if metric_path.exists():
        metrics=pd.read_csv(metric_path,dtype={'mmsi':str});joints=pd.read_csv(ROOT/'polar_profile_local.csv',dtype={'mmsi':str})
    else:
        metrics=[];joints=[]
        for k,r in enumerate(reps.to_dict('records')):
            fs,j=features(LOOK[r['sample_id']]);base={key:r[key] for key in ['sample_id','mmsi','class_name','port_proxy','ais_length_m','ais_width_m','incidence_deg','product_pass','product_relative_orbit','quarter','product']}
            metrics.extend([{**base,**x} for x in fs])
            if j:joints.append({**base,**j})
            if k%100==0:print('Measured',k,'of',len(reps),flush=True)
        metrics=pd.DataFrame(metrics);joints=pd.DataFrame(joints)
        csv('native_metrics_local.csv',metrics);csv('polar_profile_local.csv',joints)
    agg=metrics.groupby(['class_name','port_proxy','pol'],as_index=False).agg(ships=('sample_id','size'),median_length_px=('length_px','median'),q10_length_px=('length_px',lambda x:x.quantile(.1)),q90_length_px=('length_px',lambda x:x.quantile(.9)),median_contrast=('contrast','median'),median_top2_mass=('top2_mass','median'),median_entropy=('entropy','median'),median_min_bin_pixels=('min_bin_pixels','median'))
    csv('native_metrics_summary.csv',agg)
    polsum=[]
    for scope,g in [('all',joints),('confidence_ge_0p25',joints[joints.confidence>=.25])]:
        for cls,q in g.groupby('class_name'):
            polsum.append(dict(scope=scope,class_name=cls,ships=len(q),median_vv_vh_tv=q.vv_vh_tv.median(),top_bin_different_pct=100*q.top_bin_diff.mean(),at_least_two_bins_apart_pct=100*(q.top_bin_separation>=2).mean()))
    csv('polar_profile_summary.csv',pd.DataFrame(polsum))
    pairs=pd.read_csv(ROOT/'selected_pairs.csv');matched=[];galleries=[]
    feat={(r.sample_id,r.pol):r for r in metrics.itertuples()}
    for r in pairs.itertuples():
        g=reps[reps.class_name==r.class_name];a,b=g[g.port_proxy==r.port_low],g[g.port_proxy==r.port_high]
        shared=set(a.mmsi)&set(b.mmsi);a=a[~a.mmsi.isin(shared)];b=b[~b.mmsi.isin(shared)]
        examples=[]
        for pool in [a,b]:
            med=pool.ais_length_m.median();q=pool.assign(delta=(pool.ais_length_m-med).abs()).sort_values(['delta','sample_id']).head(2)
            examples+=q.to_dict('records')
        galleries.append(panel(examples,r.class_name+'_medians',r.class_name+': vessels nearest each area median AIS length'))
        edges=[]
        for x in a.itertuples():
            for y in b.itertuples():
                delta=np.abs(np.log([x.ais_length_m/y.ais_length_m,x.ais_width_m/y.ais_width_m]))
                if delta.max()<=np.log(1.1):edges.append((float(delta.sum()),x.sample_id,y.sample_id))
        used=set();selected=[];md={x['sample_id']:x for x in g.to_dict('records')}
        for dist,x,y in sorted(edges):
            if x in used or y in used:continue
            used|={x,y}
            if len(selected)<4:selected += [md[x],md[y]]
            for p in ['vv','vh']:
                fa,fb=feat.get((x,p)),feat.get((y,p))
                if fa is None or fb is None:continue
                qa=np.array([getattr(fa,f'q{i}') for i in range(6)]);qb=np.array([getattr(fb,f'q{i}') for i in range(6)])
                if not np.isfinite([qa,qb]).all():continue
                matched.append(dict(class_name=r.class_name,port_low=r.port_low,port_high=r.port_high,sample_a=x,sample_b=y,pol=p,size_log_distance=dist,confidence_min=min(fa.confidence,fb.confidence),incidence_delta=abs(fa.incidence_deg-fb.incidence_deg),same_pass_direction=fa.product_pass==fb.product_pass,same_relative_orbit=fa.product_relative_orbit==fb.product_relative_orbit,same_product=fa.product==fb.product,same_quarter=fa.quarter==fb.quarter,contrast_b_minus_a=fb.contrast-fa.contrast,length_px_b_minus_a=fb.length_px-fa.length_px,profile_tv_fixed=tv(qa,qb),profile_tv_reverse_invariant=min(tv(qa,qb),tv(qa,qb[::-1]))))
        if selected:galleries.append(panel(selected,r.class_name+'_size_matched',r.class_name+': different ships, AIS length AND beam matched within 10%'))
    matches=pd.DataFrame(matched);csv('size_matched_pairs_local.csv',matches)
    summaries=[]
    for scope,q in [('all',matches),('high_confidence',matches[matches.confidence_min>=.25]),('incidence3_same_pass',matches[(matches.incidence_delta<=3)&matches.same_pass_direction&(matches.confidence_min>=.25)])]:
        for (cls,p),g in q.groupby(['class_name','pol']):
            summaries.append(dict(scope=scope,class_name=cls,pol=p,pairs=len(g),median_contrast_delta=g.contrast_b_minus_a.median(),median_obb_length_delta=g.length_px_b_minus_a.median(),median_fixed_profile_tv=g.profile_tv_fixed.median(),median_reverse_invariant_tv=g.profile_tv_reverse_invariant.median(),same_orbit_pairs=int(g.same_relative_orbit.sum()),same_quarter_pairs=int(g.same_quarter.sum())))
    csv('size_matched_summary.csv',pd.DataFrame(summaries))
    # Same identity controls: all available cross-area identities in this legal source subset.
    cross=s.groupby('mmsi').port_proxy.nunique();ids=set(cross[cross>1].index)
    same=reps[reps.mmsi.isin(ids)].sort_values(['mmsi','port_proxy'])
    csv('same_vessel_cross_area_local.csv',same[['sample_id','mmsi','port_proxy','class_name','ais_length_m','ais_width_m']])
    if len(same):galleries.append(panel(same.head(6).to_dict('records'),'same_vessel_cross_area','Same AIS identities across geographic areas (first three IDs)'))
    # Cached baseline outputs used only for train-side descriptive cases.
    scores=np.load(WORK/'m1_coverage_20261101/initial_source.npz')['score']
    assert scores.shape==(len(ROWS),4)
    source=s.copy();source['prediction']=[int(scores[int(i)].argmax()) for i in source['index']];source['correct']=source.prediction==source.label
    margin=[]
    for z in source.itertuples():
        v=scores[z.index];margin.append(v[z.label]-max(np.delete(v,z.label)))
    source['true_class_margin']=margin
    csv('cached_prediction_cases_local.csv',source[['sample_id','class_name','port_proxy','label','prediction','correct','true_class_margin','ais_length_m','ais_width_m','mmsi']])
    csv('cached_source_class_performance.csv',source.groupby('class_name',as_index=False).agg(observations=('sample_id','size'),correct=('correct','sum'),accuracy=('correct','mean')))
    ref=pd.read_csv(ROOT/'references_local.csv',dtype={'mmsi':str});refids=set(ref.sample_id)
    # One correct and one wrong per class, closest to the class median size, no novelty filtering.
    candidates=source[~source.sample_id.isin(refids)].sort_values('sample_id').drop_duplicates('mmsi')
    for cls,g in candidates.groupby('class_name'):
        items=[]
        for correct in [True,False]:
            q=g[g.correct==correct].copy()
            if len(q)==0:continue
            q['delta']=(q.ais_length_m-g.ais_length_m.median()).abs();r=q.sort_values(['delta','sample_id']).iloc[0].to_dict()
            r['selection']='baseline_correct' if correct else 'baseline_wrong';r['note']=f"{'correct' if correct else 'wrong'}; true={r['label']} pred={r['prediction']}"
            items.append(r)
        if not items:continue
        query=items[-1]
        for label in sorted({int(query['label']),int(query['prediction'])}):
            pool=ref[ref.class_name==next(x['class_name'] for x in ROWS if x['label']==label)].copy()
            pool=pool[pool.mmsi!=query['mmsi']]
            pool['delta']=np.maximum(abs(np.log(pool.ais_length_m/query['ais_length_m'])),abs(np.log(pool.ais_width_m/query['ais_width_m'])))
            r=pool.sort_values(['delta','sample_id']).iloc[0].to_dict();r['selection']='actual_reference_size_nearest';r['note']=f'actual bank / class={label}';items.append(r)
        galleries.append(panel(items,cls+'_baseline_refs',cls+': cached source baseline correct/wrong and actual bank examples'))
    write('gallery_local.json',galleries)
    write('image_coverage.json',dict(candidate_vessel_area_records=len(reps),measured_pol_records=len(metrics),paired_vessel_area_records=len(joints),measured_unique_ships=metrics.mmsi.nunique(),missing_or_too_small_records=len(reps)-metrics.sample_id.nunique(),size_matched_pairs=matches.groupby('class_name').apply(lambda q:q[['sample_a','sample_b']].drop_duplicates().shape[0]).to_dict(),same_identity_cross_area_ships=len(ids),gallery_count=len(galleries),source_png_checks=sum(k.endswith('_png_exact') for g in galleries for r in g['cases'] for k in r),source_png_mismatches=sum(not v for g in galleries for r in g['cases'] for k,v in r.items() if k.endswith('_png_exact')),notes=['Statistics describe displayed uint8 responses, not calibrated RCS or decision importance.','Six bins have no bow/stern assignment. Cross-ship profile distance minimizes whole-axis reversal only.','Each pair matches dimensions, not all acquisition conditions; conditioned subset explicitly reported.','Actual bank examples selected by size, not claimed to have maximal learned attention.','No new predictions, model changes, training, rescue/harm events. Cached source errors are in-sample descriptive.']))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print(pd.DataFrame(polsum).to_string(index=False),flush=True)

if __name__=='__main__':main()
