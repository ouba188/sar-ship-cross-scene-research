"""Expand paired observations into vessel-balanced multi-pass trajectories."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import importlib.util
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
spec = importlib.util.spec_from_file_location('morphometry', ROOT/'condition_morphometry_20261021/run.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def pass_key(product):
    fields = product.removesuffix('.SAFE').split('_')
    assert len(fields) == 9 and len(fields[6]) == 6 and fields[6].isdigit(), product
    return '_'.join((fields[0], fields[6], fields[7]))


def counts(frame):
    return frame.groupby('mmsi').agg(observations=('obs_id','size'),
        passes=('overpass','nunique'), ports=('port','nunique'), classes=('class_id','nunique'),
        first=('time','min'),last=('time','max'))


def compare(measured):
    rows = []
    metrics = ['length_m','width_m','r3_length_m','r3_width_m']
    for (ship, pol), f in measured.groupby(['mmsi','pol']):
        for a, b in itertools.combinations(f.to_dict('records'),2):
            if a['platform'] != b['platform'] or a['pass_name'] != b['pass_name'] or a['port'] != b['port']:
                continue
            if abs(a['incidence']-b['incidence']) > 2:
                continue
            gap = math.degrees(.5*math.acos(np.clip(a['sin2phi']*b['sin2phi']+a['cos2phi']*b['cos2phi'],-1,1)))
            group = 'near' if gap <= 10 else 'far' if gap >= 30 else None
            if group is None:
                continue
            row = dict(mmsi=ship,pol=pol,group=group,obs_a=a['obs_id'],obs_b=b['obs_id'],angle_gap=gap)
            for metric in metrics:
                valid = not metric.startswith('r3') or (a['r3_valid'] == True and b['r3_valid'] == True)
                av,bv = a.get(metric,np.nan),b.get(metric,np.nan)
                row[metric] = abs(math.log(bv/av)) if valid and av>0 and bv>0 else np.nan
            rows.append(row)
    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT/'condition_pairs.csv',index=False)
    result=[]; ship_rows=[]
    rng=np.random.default_rng(20261021)
    for pol, f in pairs.groupby('pol'):
        for metric in metrics:
            wide=f.groupby(['mmsi','group'])[metric].mean().unstack().dropna(subset=['near','far'])
            delta=(wide['far']-wide['near']).to_numpy()
            for ship,r in wide.iterrows():
                ship_rows.append(dict(mmsi=ship,pol=pol,metric=metric,near=r['near'],far=r['far'],delta=r['far']-r['near']))
            boot=np.array([np.mean(rng.choice(delta,len(delta),replace=True)) for _ in range(1000)]) if len(delta) else np.array([np.nan])
            result.append(dict(pol=pol,metric=metric,ships=len(wide),
                near_mean_abs_log=wide['near'].mean(),far_mean_abs_log=wide['far'].mean(),
                far_minus_near=np.mean(delta) if len(delta) else np.nan,
                ci_low=np.nanquantile(boot,.025),ci_high=np.nanquantile(boot,.975),
                far_more_variable_ships=int((delta>0).sum())))
    pd.DataFrame(ship_rows).to_csv(OUT/'condition_comparison_by_ship.csv',index=False)
    pd.DataFrame(result).to_csv(OUT/'condition_comparison.csv',index=False)
    return result


def plots(measured, obs, cohort):
    groups=[]
    for ship,f in cohort.sort_values(['time','obs_id']).groupby('mmsi'):
        c=f.correct.to_numpy(bool)
        kind='always_correct' if c.all() else 'always_wrong' if not c.any() else 'mixed_last_correct' if c[-1] else 'mixed_last_wrong'
        groups.append(dict(mmsi=ship,kind=kind,n=len(f)))
    g=pd.DataFrame(groups)
    g.to_csv(OUT/'prediction_trajectories.csv',index=False)
    selected=g.sort_values(['n','mmsi'],ascending=[False,True]).groupby('kind',sort=False).head(1)
    selected.to_csv(OUT/'cases.csv',index=False)
    for row in selected.itertuples():
        seq=cohort[cohort.mmsi.eq(row.mmsi)].sort_values(['time','obs_id'])
        ids=seq.obs_id.tolist(); columns=min(4,len(ids)); batches=math.ceil(len(ids)/columns)
        fig,axs=plt.subplots(batches*2,columns,figsize=(columns*4,batches*7),squeeze=False,layout='constrained')
        for ax in axs.ravel(): ax.set_visible(False)
        for i,oid in enumerate(ids):
            o=obs[str(oid)]; prediction=seq[seq.obs_id.eq(oid)].iloc[0]
            for j,pol in enumerate(('vv','vh')):
                m=measured[(measured.obs_id.eq(oid)) & measured.pol.eq(pol)].iloc[0]
                v=o['images'][pol];p=v['priors'];ax=axs[2*(i//columns)+j,i%columns];ax.set_visible(True)
                ax.imshow(np.asarray(Image.open(m.image_path).convert('L')),cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=(0,v['w'],v['h'],0))
                box=p['detections'][p['target_index']]
                ax.add_patch(Polygon(box['corners'],fill=False,edgecolor='lime',lw=.8))
                axis=p['ship_axis'];c=np.array(axis['center']);u=np.array([math.cos(axis['angle']),math.sin(axis['angle'])])
                line=np.array([c-axis['length_px']*.5*u,c+axis['length_px']*.5*u]);ax.plot(line[:,0],line[:,1],color='yellow',lw=.8)
                origin=p['radar_center'];a=p['radar_angle'];ax.arrow(*origin,22*math.cos(a),22*math.sin(a),color='cyan',width=.5,head_width=4,length_includes_head=True)
                ax.set_xlim(0,v['w']);ax.set_ylim(v['h'],0)
                ax.set_title(f"{o['time'][:10]} {o['port']} obs{oid} {pol.upper()}\ninc={m.incidence:.1f} rel={m.phi_deg%180:.1f} deg | {m.length_m:.0f}x{m.width_m:.0f}m\ntrue={o['class_id']} pred={prediction.prediction} {'correct' if prediction.correct else 'wrong'}",fontsize=9)
        fig.suptitle(f'MMSI {row.mmsi} | {row.kind} | {len(ids)} distinct passes\nNative pixels / PNG 0-255; green OBB, yellow unsigned axis, cyan to satellite',fontsize=12)
        fig.savefig(OUT/f'case_{row.mmsi}.png',dpi=105);plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained',sharex=True)
        for pol,color in [('vv','#147b75'),('vh','#b44474')]:
            f=measured[measured.mmsi.eq(row.mmsi)&measured.pol.eq(pol)].sort_values(['time','obs_id'])
            x=pd.to_datetime(f.time)
            axes[0].plot(x,f.length_m,'o-',label=f'{pol.upper()} length',color=color)
            axes[0].plot(x,f.width_m,'s--',label=f'{pol.upper()} width',color=color)
            axes[1].plot(x,f.incidence,'o-',label=f'{pol.upper()} incidence',color=color)
            axes[1].plot(x,np.degrees(.5*np.arccos(np.clip(f.cos2phi,-1,1))),'s--',label=f'{pol.upper()} acute relative angle',color=color)
            axes[2].plot(x,f.contrast,'o-',label=f'{pol.upper()} PNG contrast',color=color)
        for ax,label in zip(axes,['Apparent OBB dimensions (m)','Angle (degrees)','Standardized local contrast']):
            ax.set_ylabel(label);ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2)
        fig.suptitle(f'MMSI {row.mmsi}: all eligible passes; not true physical dimensions')
        fig.savefig(OUT/f'trajectory_{row.mmsi}.png',dpi=120);plt.close(fig)


def main():
    data=json.loads((ROOT/'gallery_data.js').read_text(encoding='utf-8').split('=',1)[1].strip().removesuffix(';'))
    obs=data['observations']
    frame=pd.DataFrame([dict(obs_id=o['id'],mmsi=o['mmsi'],class_id=o['class_id'],port=o['port'],time=o['time'],product=o['product'],overpass=pass_key(o['product'])) for o in obs.values()])
    all_counts=counts(frame);all_counts.to_csv(OUT/'gallery_ship_counts.csv')
    prior=pd.read_csv(ROOT/'paired_relation_probe_20261020/data_review/observation_eligibility.csv')
    checked=frame[frame.obs_id.isin(prior.loc[prior.eligible.eq(1),'obs_id'])].copy()
    checked['geometry_both']=checked.obs_id.map(lambda oid: all(obs[str(oid)]['images'][pol]['priors'].get('radar_kind')=='orbit_los' and obs[str(oid)]['images'][pol]['priors'].get('ship_axis') is not None for pol in ('vv','vh')))
    checked.to_csv(OUT/'checked_observations.csv',index=False)
    clean=checked[checked.geometry_both].sort_values(['time','obs_id']).drop_duplicates(['mmsi','overpass'])
    cc=counts(clean);cc.to_csv(OUT/'checked_ship_counts.csv')
    selected=cc.index[(cc.passes>=4)&(cc.classes==1)]
    cohort=clean[clean.mmsi.isin(selected)].copy()
    pred=pd.read_csv(ROOT/'axial_strip_sequence_20261020/predictions.csv')
    pred=pred[pred.arm.eq('frozen_fivepoint')&pred.seed.eq(0)]
    cohort=cohort.merge(pred[['obs_id','prediction','correct']],on='obs_id',validate='one_to_one')
    assert cohort[['mmsi','overpass']].duplicated().sum()==0
    assert cohort.groupby('mmsi').size().min()>=4
    cohort.to_csv(OUT/'cohort.csv',index=False)
    print('Cohort',len(cohort),'observations',len(selected),'ships',flush=True)
    cache=pd.read_csv(ROOT/'condition_morphometry_20261021/measurements.csv',dtype={'mmsi':str}).set_index(['obs_id','pol'])
    orbit=pd.read_csv(ROOT/'radar_geometry/per_view_geometry.csv',low_memory=False).set_index(['obs_id','pol'])
    rows=[]; reused=0
    for n,oid in enumerate(cohort.obs_id):
        for pol in ('vv','vh'):
            if (oid,pol) in cache.index:
                row=cache.loc[(oid,pol)].to_dict();row.update(obs_id=oid,pol=pol);reused+=1
            else:
                row=old.measure(obs[str(oid)],pol,orbit.loc[(oid,pol)])
            rows.append(row)
        if (n+1)%100==0:print('Measured',n+1,flush=True)
    m=pd.DataFrame(rows);m['mmsi']=m.mmsi.astype(str);m.to_csv(OUT/'measurements.csv',index=False)
    coverage=[]
    for (ship,pol),f in m.groupby(['mmsi','pol']):
        dots=np.outer(f.sin2phi,f.sin2phi)+np.outer(f.cos2phi,f.cos2phi)
        coverage.append(dict(mmsi=ship,pol=pol,observations=len(f),ports=f.port.nunique(),class_id=int(f.class_id.iloc[0]),
            days=(pd.to_datetime(f.time).max()-pd.to_datetime(f.time).min()).total_seconds()/86400,
            incidence_range=f.incidence.max()-f.incidence.min(),max_relative_angle_gap=np.degrees(.5*np.arccos(np.clip(dots,-1,1))).max(),
            length_max_min_ratio=f.length_m.max()/f.length_m.min(),width_max_min_ratio=f.width_m.max()/f.width_m.min()))
    pd.DataFrame(coverage).to_csv(OUT/'condition_coverage.csv',index=False)
    comparison=compare(m);plots(m,obs,cohort)
    summary=dict(gallery_observations=len(frame),gallery_ships=len(all_counts),
        gallery_pass_histogram=all_counts.passes.value_counts().sort_index().to_dict(),
        gallery_ge4=int((all_counts.passes>=4).sum()),gallery_ge6=int((all_counts.passes>=6).sum()),
        checked_observations=len(checked),checked_geometry_observations=int(checked.geometry_both.sum()),
        cohort_ships=len(selected),cohort_observations=len(cohort),views=len(m),cache_reused_views=reused,
        cohort_pass_histogram=cohort.groupby('mmsi').size().value_counts().sort_index().to_dict(),
        cohort_cross_port_ships=int((cohort.groupby('mmsi').port.nunique()>1).sum()),
        cohort_classes=cohort.drop_duplicates('mmsi').class_id.value_counts().to_dict(),
        comparison=comparison,classifier_training=False,new_prediction_validation=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
