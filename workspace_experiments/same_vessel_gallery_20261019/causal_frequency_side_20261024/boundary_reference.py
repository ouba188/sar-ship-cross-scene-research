"""Boundary support and reference-frame diagnostics on frozen paired chips."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='2'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, binary_dilation
import joint_probe as joint

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'round08_boundary_reference'


def axial_delta(a,b):
    return abs(float(np.degrees(np.arctan2(np.sin(2*(b-a)),np.cos(2*(b-a))))))/2


def region_masks(im,p,record,sigma):
    sh=p['ship_axis'];base=joint.masks_for(im,sh,sh['length_px'],sh['width_px'])
    body=base['body'];radius=int(3*sigma);support=np.ones((2*radius+1,2*radius+1),bool)
    core=binary_erosion(body,structure=support)
    bright=(im*255>=record.r3_threshold)&body
    bright_core=binary_erosion(bright,structure=support)
    bright_edge=binary_dilation(bright,structure=support)&~bright_core
    near_sat=binary_dilation(im*255>=254,structure=support)
    yy,xx=np.indices(im.shape)
    distance=np.hypot(xx+.5-sh['center'][0],yy+.5-sh['center'][1])*10
    ring=(distance>=180)&(distance<260)
    valid=(xx>=radius)&(xx<im.shape[1]-radius)&(yy>=radius)&(yy<im.shape[0]-radius)
    clear=~binary_dilation(body,structure=support)
    masks={**base,'obb_core':core,'obb_edge':body&~core,
           'bright_core':bright_core,'ring':ring,
           'ring_no_obb_input':ring&clear,
           'end_no_obb_input':base['end']&clear,'side_no_obb_input':base['side']&clear}
    return masks,valid,near_sat,bright_edge,bright


def measure(oid,pol,data,frame,scale):
    im,p,r=joint.image_data(oid,pol,data,frame);sigma=scale/10
    masks,valid,sat,bright_edge,bright=region_masks(im,p,r,sigma)
    gx,gy=joint.gradients(im,sigma);energy=gx*gx+gy*gy
    tensors,rows={},[]
    total=energy[masks['body']&valid].sum()
    body_n=int((masks['body']&valid).sum())
    for region,m0 in masks.items():
        m=m0&valid;n=int(m.sum());coverage=n/m0.sum() if m0.any() else 0.
        eng=float(energy[m].sum());eligible=n>=12 and coverage>=.9 and eng>1e-14
        if eng>1e-14 and n:
            ten=joint.tensor(gx,gy,m);tensors[region]=ten
            eig=np.linalg.eigvalsh(ten);anisotropy=float((eig[-1]-eig[0])/eig.sum())
        else:anisotropy=np.nan
        rows.append(dict(obs_id=oid,pol=pol,scale_m=scale,region=region,n=n,
            coverage=float(coverage),eligible=bool(eligible),energy_sum=eng,
            body_energy_fraction=eng/total if region in ('body','obb_core','obb_edge','bright_core') else np.nan,
            body_area_fraction=n/body_n if region in ('body','obb_core','obb_edge','bright_core') else np.nan,
            anisotropy=anisotropy,
            saturated_neighborhood_energy_fraction=float(energy[m&sat].sum()/eng) if eng>0 else np.nan,
            saturated_neighborhood_area_fraction=float((m&sat).sum()/n) if n else np.nan,
            bright_boundary_energy_fraction=float(energy[m&bright_edge].sum()/eng) if eng>0 else np.nan,
            bright_boundary_area_fraction=float((m&bright_edge).sum()/n) if n else np.nan,
            image_path=r.image_path))
    inside=joint.gradients(im*masks['body'],sigma)
    outside=joint.gradients(im*(~masks['body']),sigma)
    tests={}
    for name,mask,other in [('core_support',masks['obb_core']&valid,inside),
                            ('ring_support',masks['ring_no_obb_input']&valid,outside),
                            ('end_support',masks['end_no_obb_input']&valid,outside),
                            ('side_support',masks['side_no_obb_input']&valid,outside)]:
        tests[name]=max(float(abs(g[mask]-h[mask]).max()) for g,h in zip((gx,gy),other)) if mask.any() else 0.
    assert max(tests.values())<1e-12,tests
    return rows,tensors,p,tests


def summarize(measurements,pairs):
    # Measurement rows are unique images, rather than repeated paired exposures.
    measurements.groupby(['pol','scale_m','region']).agg(
        observations=('obs_id','size'),eligible_views=('eligible','sum'),
        median_pixels=('n','median'),median_coverage=('coverage','median'),
        median_body_energy_fraction=('body_energy_fraction','median'),
        median_body_area_fraction=('body_area_fraction','median'),
        median_sat_energy=('saturated_neighborhood_energy_fraction','median'),
        median_sat_area=('saturated_neighborhood_area_fraction','median'),
        median_bright_boundary_energy=('bright_boundary_energy_fraction','median'),
        median_bright_boundary_area=('bright_boundary_area_fraction','median')).to_csv(OUT/'boundary_summary.csv')
    records=[]
    for (pol,scale,region),g in pairs[pairs.eligible].groupby(['pol','scale_m','region']):
        for label,h in [('all',g),('radar_turn_ge5',g[g.radar_turn_deg>=5]),('radar_turn_lt5',g[g.radar_turn_deg<5])]:
            if h.empty:continue
            row=dict(pol=pol,scale_m=scale,region=region,stratum=label,n=len(h),ships=h.mmsi.nunique(),
                median_anisotropy=h.min_anisotropy.median(),low_anisotropy_fraction=(h.min_anisotropy<.1).mean())
            for axis in ('image','ship','radar'):
                row['median_'+axis]=h[axis+'_error'].median();row['mean_'+axis]=h[axis+'_error'].mean()
            for a,b in [('ship','image'),('radar','image'),('radar','ship')]:
                difference=h[b+'_error']-h[a+'_error']
                row[a+'_over_'+b+'_mean_gain']=difference.mean()
                row[a+'_over_'+b+'_win_fraction']=(difference>1e-8).mean()
            records.append(row)
    pd.DataFrame(records).to_csv(OUT/'reference_summary.csv',index=False)
    body=pairs[pairs.region.eq('body')][['query_id','history_id','pol','scale_m','image_error','ship_error','radar_error']]
    core=pairs[pairs.region.eq('obb_core')&pairs.eligible]
    common=core.merge(body,on=['query_id','history_id','pol','scale_m'],suffixes=('_core','_body'),validate='one_to_one')
    common.to_csv(OUT/'common_body_core_pairs.csv',index=False)
    by_pass=[]
    for (pol,region),g in pairs[pairs.eligible & pairs.scale_m.eq(10)].groupby(['pol','region']):
        for axis,base in [('radar','image'),('radar','ship'),('ship','image')]:
            z=g.assign(gain=g[base+'_error']-g[axis+'_error'])
            for pass_id,h in z.groupby('query_pass'):
                by_pass.append(dict(pol=pol,region=region,axis=axis,baseline=base,
                    query_pass=pass_id,n=len(h),mean_gain=h.gain.mean()))
    pd.DataFrame(by_pass).to_csv(OUT/'pass_gains.csv',index=False)


def make_panels(data,frame):
    os.environ['MPLCONFIGDIR']=str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    selected=pd.read_csv(ROOT/'round07_joint_coverage/review_selection.csv')
    selected.to_csv(OUT/'review_selection.csv',index=False)
    for rec in selected.itertuples():
        fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
        for row,pol in enumerate(('vv','vh')):
            items=[joint.image_data(i,pol,data,frame) for i in (rec.history_id,rec.query_id)]
            lim=max(max(im.shape)*5 for im,_,_ in items)
            for col,(oid,(im,p,r)) in enumerate(zip((rec.history_id,rec.query_id),items)):
                masks,valid,sat,edge,bright=region_masks(im,p,r,1)
                h,w=im.shape;cx,cy=p['ship_axis']['center'];extent=(-cx*10,(w-cx)*10,(h-cy)*10,-cy*10)
                for j in (col,col+2):
                    axes[row,j].imshow(im,cmap='gray',vmin=0,vmax=1,extent=extent,interpolation='nearest')
                    axes[row,j].set_xlim(-lim,lim);axes[row,j].set_ylim(lim,-lim)
                    axes[row,j].set_xlabel('native metres')
                overlay=np.zeros((*im.shape,4))
                overlay[masks['obb_edge']&valid]=[1,.3,.1,.35]
                overlay[masks['obb_core']&valid]=[0,.8,1,.6]
                overlay[masks['bright_core']&valid]=[1,1,0,.65]
                axes[row,col].imshow(overlay,extent=extent,interpolation='nearest')
                ext=np.zeros((*im.shape,4));ext[masks['ring']&valid]=[1,0,1,.25]
                ext[masks['ring_no_obb_input']&valid]=[0,1,.4,.4]
                axes[row,col+2].imshow(ext,extent=extent,interpolation='nearest')
                corners=(np.array(p['detections'][p['target_index']]['corners'])-[cx,cy])*10
                for j in (col,col+2):
                    axes[row,j].add_patch(Polygon(corners,fill=False,edgecolor='white',linewidth=.7))
                axes[row,col].set_title(f'{oid} {pol.upper()} boundary/core')
                axes[row,col+2].set_title(f'{oid} {pol.upper()} ring/input support')
        fig.suptitle(f'{rec.state}: {rec.history_id}->{rec.query_id}, margin {rec.margin_old:+.3f}->{rec.margin_new:+.3f}\n'
                     'Orange: OBB edge; cyan: deep OBB; yellow: eroded bright mask; green: ring not reading OBB')
        fig.savefig(OUT/f'case_{rec.query_id}.png',dpi=125);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    data,frame=joint.load()
    old=pd.read_csv(ROOT/'round07_joint_coverage/pairs.csv',dtype={'mmsi':str})
    requests=old[old['mode'].eq('own_box')&old.region.eq('body')]
    cache={};measurements=[];rows=[];max_errors={k:0. for k in ('core_support','ring_support','end_support','side_support')}
    for rec in requests.itertuples():
        for scale in (10.,20.):
            parts=[]
            for oid in (rec.history_id,rec.query_id):
                key=(oid,rec.pol,scale)
                if key not in cache:
                    z,t,p,tests=measure(oid,rec.pol,data,frame,scale)
                    cache[key]=(pd.DataFrame(z).set_index('region'),t,p)
                    measurements.extend(z)
                    for k,v in tests.items():max_errors[k]=max(max_errors[k],v)
                parts.append(cache[key])
            (z0,t0,p0),(z1,t1,p1)=parts
            angles={'image':0.,'ship':p1['ship_axis']['angle']-p0['ship_axis']['angle'],
                    'radar':p1['radar_axis']-p0['radar_axis']}
            for region in z0.index:
                eligible=bool(z0.loc[region,'eligible'] and z1.loc[region,'eligible'])
                row=dict(query_id=rec.query_id,history_id=rec.history_id,pol=rec.pol,mmsi=rec.mmsi,
                    scale_m=scale,region=region,eligible=eligible,
                    pixels_old=int(z0.loc[region,'n']),pixels_new=int(z1.loc[region,'n']),
                    radar_turn_deg=axial_delta(p0['radar_axis'],p1['radar_axis']),
                    ship_turn_deg=axial_delta(p0['ship_axis']['angle'],p1['ship_axis']['angle']),
                    min_anisotropy=min(z0.loc[region,'anisotropy'],z1.loc[region,'anisotropy']),
                    state=rec.state,class_id=rec.class_id,port=rec.port,
                    query_pass=rec.query_pass,history_pass=rec.history_pass,margin_delta=rec.margin_delta)
                for axis,angle in angles.items():
                    value=np.nan
                    if region in t0 and region in t1:
                        a,b=t0[region],t1[region];r=joint.rotation(angle)
                        value=float(np.linalg.norm(b/np.trace(b)-r@(a/np.trace(a))@r.T))
                    row[axis+'_error']=value
                rows.append(row)
    m=pd.DataFrame(measurements);pairs=pd.DataFrame(rows)
    main_body=pairs[pairs.region.eq('body')&pairs.scale_m.eq(10)].merge(
        requests[['query_id','pol','tensor_distance','rotation_residual']],on=['query_id','pol'],validate='one_to_one')
    replay=max(abs(main_body.image_error-main_body.tensor_distance).max(),abs(main_body.ship_error-main_body.rotation_residual).max())
    assert replay<1e-12,replay
    m.to_csv(OUT/'measurements.csv',index=False);pairs.to_csv(OUT/'pairs.csv',index=False)
    summarize(m,pairs);make_panels(data,frame)
    manifest=dict(pair_views=len(requests),ships=requests.mmsi.nunique(),
        unique_observation_pairs=len(requests[['history_id','query_id']].drop_duplicates()),
        unique_image_views=len(m[['obs_id','pol']].drop_duplicates()),
        checks={**max_errors,'round07_replay_error':float(replay)},minimum_region_pixels=12,
        scales_m=[10,20],no_classifier_training=True,physical_causal_effect_estimated=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest));print(pd.read_csv(OUT/'reference_summary.csv').query('scale_m==10 and stratum=="all"').to_string(index=False))


if __name__=='__main__':main()
