"""Native same-vessel measurements; all output stays inside this pilot directory."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import json
import math
from pathlib import Path
from urllib.parse import unquote, urlparse
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
GALLERY = OUT.parent
SOURCE = GALLERY/'paired_relation_probe_20261020/data_review'


def polygon_inside(points, vertices):
    edge = np.roll(vertices, -1, axis=0)-vertices
    diff = points[:, None, :]-vertices[None, :, :]
    cross = edge[None, :, 0]*diff[:, :, 1]-edge[None, :, 1]*diff[:, :, 0]
    return (cross >= -1e-7).all(1) | (cross <= 1e-7).all(1)


def edge_distance(points, vertices):
    edge = np.roll(vertices, -1, axis=0)-vertices
    diff = points[:, None, :]-vertices[None, :, :]
    t = np.clip((diff*edge[None]).sum(2)/(edge*edge).sum(1), 0, 1)
    return np.linalg.norm(diff-t[:, :, None]*edge[None], axis=2).min(1)


def weighted_span(position, weights):
    order = np.argsort(position)
    x, w = position[order], weights[order]
    cdf = (np.cumsum(w)-.5*w)/w.sum()
    return float(np.interp(.95, cdf, x)-np.interp(.05, cdf, x))


def describe(x):
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std()),
                median=float(np.median(x)), p95=float(np.percentile(x, 95)),
                saturated=float((x == 255).mean())) if len(x) else dict(n=0)


def measure(obs, pol, orbit):
    view = obs['images'][pol]; pri = view['priors']
    path = unquote(urlparse(view['url']).path).lstrip('/')
    image = np.asarray(Image.open(path).convert('L'), dtype=float)
    assert image.shape == (view['h'], view['w'])
    g = np.asarray(pri['scale']['transform'], float)
    affine = g[[1, 2, 4, 5]].reshape(2, 2)
    area = abs(np.linalg.det(affine))
    yy, xx = np.indices(image.shape)
    pixels = np.column_stack([xx.ravel()+.5, yy.ravel()+.5])
    center = np.asarray(pri['radar_center'])
    points = (pixels-center)@affine.T
    box = pri['detections'][pri['target_index']]
    vertices = (np.asarray(box['corners'])-center)@affine.T
    edges = np.roll(vertices, -1, axis=0)-vertices
    lengths = np.linalg.norm(edges[:2], axis=1)
    long = int(np.argmax(lengths)); ship = edges[long]/lengths[long]
    if math.atan2(ship[1], ship[0]) % (2*math.pi) >= math.pi:
        ship = -ship
    normal = np.array([-ship[1], ship[0]])
    los = affine@np.array([math.cos(pri['radar_angle']), math.sin(pri['radar_angle'])])
    los /= np.linalg.norm(los)
    flight = affine@np.array([float(orbit['flight_image_x']), float(orbit['flight_image_y'])])
    flight /= np.linalg.norm(flight)
    phi = math.atan2(los[0]*ship[1]-los[1]*ship[0], los@ship)
    target = polygon_inside(points, vertices)
    other = np.zeros(len(points), bool)
    other_count = 0
    for i, det in enumerate(pri['detections']):
        if i != pri['target_index'] and det['confidence'] >= .25 and det['shape_usable']:
            other |= polygon_inside(points, (np.asarray(det['corners'])-center)@affine.T)
            other_count += 1
    radius = np.linalg.norm(points, axis=1)
    distance = edge_distance(points, vertices)
    ring = ~target & ~other & (distance >= 20) & (distance < 80)
    core = (radius <= 180) & ~other
    background = (radius >= 220) & (radius < 300) & ~other
    intensity = image.ravel()
    row = dict(obs_id=obs['id'], mmsi=obs['mmsi'], pol=pol, image_path=path,
               class_id=obs['class_id'], port=obs['port'], time=obs['time'], product=obs['product'],
               platform=obs['platform'], pass_name=obs['pass'],
               length_m=float(max(lengths)), width_m=float(min(lengths)),
               aspect=float(max(lengths)/min(lengths)), obb_area_m2=float(np.prod(lengths)),
               pixel_area_m2=area, confidence=box['confidence'], other_boxes=other_count,
               incidence=pri['radar_incidence'], sin2phi=math.sin(2*phi), cos2phi=math.cos(2*phi),
               phi_deg=math.degrees(phi), core_coverage=float(core.sum()*area/(math.pi*180**2)),
               background_coverage=float(background.sum()*area/(math.pi*(300**2-220**2))),
               target_core_fraction=float((target & (radius<=180)).sum()/max(target.sum(), 1)))
    for region, mask in [('target',target),('near',ring),('far',background)]:
        row.update({region+'_'+k:v for k,v in describe(intensity[mask]).items()})
    row['contrast'] = (row['target_mean']-row.get('near_mean',np.nan))/max(row.get('near_sd',0),1.)
    valid = row['core_coverage'] >= .9 and row['background_coverage'] >= .9 and row['target_core_fraction'] >= .95
    if background.sum() >= 64:
        bg = intensity[background]
        median = np.median(bg); mad = max(1., 1.4826*np.median(np.abs(bg-median)))
        for k in (3,5):
            threshold = median+k*mad
            weights = np.maximum(intensity-threshold, 0)*core
            selected = weights>0
            prefix = f'r{k}_'
            row[prefix+'threshold'] = float(threshold)
            row[prefix+'valid'] = bool(valid and selected.sum()>=5)
            if selected.sum()>=5:
                pp, ww = points[selected], weights[selected]
                row[prefix+'area_m2'] = float(selected.sum()*area)
                row[prefix+'offset_m'] = float(np.linalg.norm(np.average(pp,axis=0,weights=ww)))
                for name, direction in [('length',ship),('width',normal),('range',los),('azimuth',flight)]:
                    row[prefix+name+'_m'] = weighted_span(pp@direction,ww)
            else:
                row[prefix+'failure'] = 'fewer_than_5_high_response_pixels'
    else:
        row['r3_valid'] = row['r5_valid'] = False
    return row


METRICS = ['length_m','width_m','aspect','contrast','target_mean','near_mean','near_sd','far_mean','target_saturated']
METRICS += [f'r{k}_{n}' for k in (3,5) for n in ('length_m','width_m','range_m','azimuth_m','area_m2','offset_m')]


def paired_table(pairs, measurements):
    lookup = measurements.set_index(['obs_id','pol'])
    rows=[]
    for p in pairs.to_dict('records'):
        for pol in ('vv','vh'):
            a,b=(lookup.loc[(p['obs_id_'+side],pol)] for side in ('a','b'))
            r=dict(p,pol=pol)
            for name in ('incidence','sin2phi','cos2phi'):
                r['d_'+name]=b[name]-a[name]; r['mean_'+name]=(b[name]+a[name])/2
            r['angle_gap_deg']=math.degrees(.5*math.acos(np.clip(a.sin2phi*b.sin2phi+a.cos2phi*b.cos2phi,-1,1)))
            r['platform_change']=int(a.platform!=b.platform)
            r['pass_change']=int(a.pass_name!=b.pass_name)
            r['log_days']=math.log1p(p['days_apart'])
            r['other_any']=int(a.other_boxes+b.other_boxes>0)
            for metric in METRICS:
                prefix=metric[:3]
                valid=not metric.startswith(('r3_','r5_')) or bool(a.get(prefix+'valid',False) and b.get(prefix+'valid',False))
                av,bv=a.get(metric,np.nan),b.get(metric,np.nan)
                if not valid: av=bv=np.nan
                r[metric+'_a'],r[metric+'_b']=av,bv
                r['delta_'+metric]=bv-av
                if metric in ('length_m','width_m','aspect') or metric.startswith(('r3_','r5_')):
                    r['dlog_'+metric]=math.log(bv/av) if av>0 and bv>0 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def analyze(paired):
    summaries=[]; relations=[]; fits=[]; predictions=[]
    size_metrics=['length_m','width_m','aspect','r3_length_m','r3_width_m','r5_length_m','r5_width_m']
    for pol,full in paired.groupby('pol'):
        groups=[('all',full),('same_port',full[full.same_port.eq(1)]),('cross_port',full[full.same_port.eq(0)])]
        groups += [('class_'+str(c),f) for c,f in full.groupby('class_id')]
        for group,frame in groups:
            for metric in size_metrics:
                f=frame.dropna(subset=['dlog_'+metric]); v=f['dlog_'+metric].to_numpy()
                if not len(v):continue
                summaries.append(dict(pol=pol,group=group,metric=metric,ships=len(f),
                    median_absolute_ratio_change_pct=100*float(np.median(np.expm1(np.abs(v)))),
                    median_signed_log_change=float(np.median(v)),q90_absolute_ratio_change_pct=100*float(np.quantile(np.expm1(np.abs(v)),.9))))
                for cond in ('angle_gap_deg','d_incidence'):
                    x=np.abs(f[cond].to_numpy()); y=np.abs(v)
                    rho=float(spearmanr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 and len(f)>=20 else np.nan
                    relations.append(dict(pol=pol,group=group,metric=metric,condition=cond,ships=len(f),spearman=rho))
        for metric in ('length_m','width_m','r3_length_m','r3_width_m','contrast'):
            target=('delta_' if metric=='contrast' else 'dlog_')+metric
            f=full.dropna(subset=[target]).copy()
            if len(f)<100:continue
            class_x=pd.get_dummies(f.class_id).to_numpy(float)
            columns=['d_incidence','d_sin2phi','d_cos2phi','mean_incidence','mean_sin2phi','mean_cos2phi','platform_change','pass_change','log_days']
            geom=f[columns].to_numpy(float)
            geom=np.column_stack([geom,f.d_incidence*f.mean_sin2phi,f.d_incidence*f.mean_cos2phi,
                                   f.d_sin2phi*f.mean_incidence,f.d_cos2phi*f.mean_incidence])
            y=f[target].to_numpy()
            meta=f[['platform_change','pass_change','log_days']].to_numpy(float)
            designs={'class_reference':class_x,'class_metadata':np.column_stack([class_x,meta]),
                     'class_incidence':np.column_stack([class_x,meta,geom[:,[0,3]]]),
                     'class_direction':np.column_stack([class_x,meta,geom[:,[1,2,4,5]]]),
                     'class_geometry':np.column_stack([class_x,geom])}
            pred={name:np.empty(len(f)) for name in designs}
            for train,test in KFold(n_splits=5,shuffle=True,random_state=20261021).split(f):
                for name,x in designs.items():
                    scaler=StandardScaler().fit(x[train]); model=Ridge(alpha=10).fit(scaler.transform(x[train]),y[train])
                    pred[name][test]=model.predict(scaler.transform(x[test]))
            for reference,candidate in [('class_reference','class_geometry'),('class_metadata','class_incidence'),
                                        ('class_metadata','class_direction'),('class_metadata','class_geometry')]:
                e0=np.abs(pred[reference]-y); e1=np.abs(pred[candidate]-y)
                gain=e0-e1; rng=np.random.default_rng(20261021)
                boot=np.array([gain[rng.integers(0,len(gain),len(gain))].mean() for _ in range(500)])
                fits.append(dict(pol=pol,metric=metric,reference=reference,candidate=candidate,ships=len(f),reference_mae=float(e0.mean()),geometry_mae=float(e1.mean()),
                    mae_improvement_pct=float(100*(e0.mean()-e1.mean())/e0.mean()),
                    paired_gain_ci_low=float(np.quantile(boot,.025)),paired_gain_ci_high=float(np.quantile(boot,.975)),
                    no_change_mae=float(np.abs(y).mean())))
            for i,r in enumerate(f.itertuples()):
                predictions.append(dict(pair_id=r.pair_id,pol=pol,metric=metric,truth=y[i],
                    **{name:value[i] for name,value in pred.items()}))
    for name,rows in [('change_summary',summaries),('associations',relations),('heldout_change_prediction',fits),('heldout_predictions',predictions)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)


def panels(paired, observations, measurements):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Ellipse
    outcomes=pd.read_csv(SOURCE/'main_pairs_seed0.csv',dtype={'mmsi':str})
    choose=paired[paired.pol.eq('vv')].merge(outcomes[['pair_id','outcome','prediction_a','prediction_b','margin_a','margin_b']],on='pair_id')
    selected=[]
    lookup=measurements.set_index(['obs_id','pol'])
    for outcome,frame in choose.groupby('outcome'):
        values=frame.dlog_width_m.abs(); idx=(values-values.median()).abs().idxmin(); row=frame.loc[idx]
        fig,axs=plt.subplots(2,2,figsize=(10,10),layout='constrained')
        for i,side in enumerate(('a','b')):
            o=observations[str(row['obs_id_'+side])]
            for j,pol in enumerate(('vv','vh')):
                view=o['images'][pol]; p=view['priors']; m=lookup.loc[(o['id'],pol)]
                pixels=np.asarray(Image.open(m.image_path).convert('L'))
                ax=axs[i,j];ax.imshow(pixels,cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=(0,view['w'],view['h'],0))
                for k,det in enumerate(p['detections']):
                    if det['shape_usable'] and det['confidence']>=.25:
                        ax.add_patch(Polygon(det['corners'],fill=False,edgecolor='lime' if k==p['target_index'] else 'orange',lw=1))
                g=np.asarray(p['scale']['transform']); sx=np.hypot(g[1],g[4]);sy=np.hypot(g[2],g[5])
                assert abs(g[1]*g[2]+g[4]*g[5])<1e-5, 'Panel requires orthogonal image axes'
                for radius,color in ((180,'cyan'),(220,'yellow'),(300,'yellow')):
                    ax.add_patch(Ellipse(p['radar_center'],2*radius/sx,2*radius/sy,fill=False,edgecolor=color,lw=.7,linestyle='--'))
                ax.set_xlim(0,view['w']);ax.set_ylim(view['h'],0)
                ax.set_title(f"{side.upper()} obs{o['id']} {pol.upper()} {o['port']}\n{o['time']} inc={m.incidence:.1f} pred={row['prediction_'+side]}\nOBB {m.length_m:.1f} x {m.width_m:.1f}m; contrast {m.contrast:.2f}\nresponse3 {m.get('r3_length_m',np.nan):.1f} x {m.get('r3_width_m',np.nan):.1f}m valid={m.get('r3_valid',False)}",fontsize=9)
                ax.set_xlabel('Native pixel x');ax.set_ylabel('Native pixel y')
        fig.suptitle(f"MMSI {row.mmsi} | {row.class_name} | {outcome} | PNG display, NOT calibrated SAR",fontsize=11)
        file=f'case_{outcome}_{row.pair_id}.png';fig.savefig(OUT/file,dpi=130);plt.close(fig)
        selected.append(dict(pair_id=row.pair_id,mmsi=row.mmsi,outcome=outcome,panel=file))
    pd.DataFrame(selected).to_csv(OUT/'cases.csv',index=False)


def main():
    pairs=pd.read_csv(SOURCE/'main_pairs.csv',dtype={'mmsi':str})
    assert pairs.mmsi.is_unique
    data=json.loads((GALLERY/'gallery_data.js').read_text(encoding='utf-8').removeprefix('window.GALLERY_DATA=').strip().removesuffix(';'))
    obs=data['observations']; ids=sorted(set(pairs.obs_id_a)|set(pairs.obs_id_b))
    for pair in pairs.itertuples():
        for oid in (pair.obs_id_a,pair.obs_id_b):
            assert obs[str(oid)]['mmsi']==pair.mmsi and obs[str(oid)]['class_id']==pair.class_id
    orbit=pd.read_csv(GALLERY/'radar_geometry/per_view_geometry.csv',low_memory=False)
    orbit=orbit[orbit.obs_id.isin(ids)].set_index(['obs_id','pol'])
    rows=[]
    for n,oid in enumerate(ids):
        for pol in ('vv','vh'):
            rows.append(measure(obs[str(oid)],pol,orbit.loc[(oid,pol)]))
        if (n+1)%200==0:print(f'Measured {n+1}/{len(ids)} observations',flush=True)
    m=pd.DataFrame(rows);m.to_csv(OUT/'measurements.csv',index=False)
    p=paired_table(pairs,m);p.to_csv(OUT/'paired_changes.csv',index=False)
    analyze(p);panels(p,obs,m)
    summary=dict(ships=len(pairs),observations=len(ids),views=len(m),classes=pairs.class_name.value_counts().to_dict(),
        cross_port_pairs=int(pairs.same_port.eq(0).sum()),measurement_domain='native uint8 PNG, NOT calibrated backscatter',
        calibrated_pixels_used=False,classifier_training=False,source_files_modified=False,
        r3_valid_views=int(m.r3_valid.sum()),r5_valid_views=int(m.r5_valid.sum()),
        pair_selection='existing maximum-date-gap one pair per MMSI, not based on outcomes',
        classification_reference='archived frozen_fivepoint seed0 local probe, not full support classifier')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
