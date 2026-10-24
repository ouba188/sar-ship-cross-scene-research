"""Frozen spatial-frequency checks on additional local acquisition dates."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from PIL import Image, ImageDraw, ImageFont
import raw_pair_probe as raw
import calibrated_relation as cal
import joint_spatial_frequency as joint

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'round19_external_rotation'


def cohort(gallery):
    m = pd.read_csv(raw.WORK/'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv', low_memory=False)
    products = sorted(p.stem for p in raw.SAFE.glob('*.SAFE') if not p.stem.endswith(('_122F', '_CEA9')))
    m = m[m['product'].isin(products)].dropna(subset=['mmsi']).copy()
    m['mmsi'] = m.mmsi.astype('int64').astype(str)
    m = m[~m.duplicated(['mmsi', 'product'], keep=False)]
    m['date'] = pd.to_datetime(m['product'].str[17:32], format='%Y%m%dT%H%M%S')
    old = set(pd.read_csv(ROOT/'round14_raw_pairs/cohort.csv', dtype={'mmsi': str}).mmsi)
    lookup = {o['instance_id']: o for o in gallery.values()}
    rows = []
    for mmsi, g in m.groupby('mmsi'):
        if len(g) < 2 or g.class_name.nunique() != 1:
            continue
        candidates = list(itertools.combinations(list(g.sort_values('date').itertuples()), 2))
        a, b = min(candidates, key=lambda p: (p[1].date-p[0].date, p[0].date))
        rows.append(dict(mmsi=mmsi, class_name_a=a.class_name, instance_id_a=a.instance_id,
                         instance_id_b=b.instance_id, obs_id_a=lookup[a.instance_id]['id'],
                         obs_id_b=lookup[b.instance_id]['id'], product_a=a.product, product_b=b.product,
                         days=(b.date-a.date).total_seconds()/86400, novel_ship=mmsi not in old))
    c = pd.DataFrame(rows)
    views = pd.read_csv(raw.WORK/'chip_quality_20261017/ais_metadata/view_metadata.csv', low_memory=False)
    ids = set(c.instance_id_a) | set(c.instance_id_b)
    views = views[views.instance_id.isin(ids)].copy()
    assert len(views) == 4*len(c) and not views.duplicated(['instance_id', 'pol']).any()
    checks = []
    for instance, v in views.groupby('instance_id'):
        v = v.set_index('pol')
        if v[['epsg','chip_transform_json']].isna().any().any():
            checks.append(dict(instance_id=instance,obs_id=lookup[instance]['id'],distance_m=np.nan,georef_available=False))
            continue
        assert int(v.loc['vv','epsg']) == int(v.loc['vh','epsg'])
        points = []
        for pol in ('vv','vh'):
            gt = json.loads(v.loc[pol, 'chip_transform_json'])
            cx, cy = lookup[instance]['images'][pol]['priors']['anchor']
            points.append(np.array([gt[0]+cx*gt[1]+cy*gt[2], gt[3]+cx*gt[4]+cy*gt[5]]))
        checks.append(dict(instance_id=instance, obs_id=lookup[instance]['id'], distance_m=np.linalg.norm(points[0]-points[1]),georef_available=True))
    check = pd.DataFrame(checks)
    bad = set(check.loc[check.distance_m > 100, 'instance_id'])
    missing = set(check.loc[~check.georef_available, 'instance_id'])
    c['geo_ok'] = ~c.instance_id_a.isin(bad) & ~c.instance_id_b.isin(bad)
    c['raw_ready'] = ~c.instance_id_a.isin(missing) & ~c.instance_id_b.isin(missing)
    c.to_csv(OUT/'cohort.csv', index=False)
    check.to_csv(OUT/'crosspol_locations.csv', index=False)
    print('Cohort', len(c), 'novel', c.novel_ship.sum(), 'geo valid', c.geo_ok.sum(), 'raw ready',c.raw_ready.sum(),flush=True)
    ready=c[c.raw_ready].copy()
    ready.to_csv(OUT/'ready_cohort.csv',index=False)
    ids=set(ready.instance_id_a)|set(ready.instance_id_b)
    return ready, views[views.instance_id.isin(ids)], lookup


def moments(im, center, theta):
    z, power, win = joint.measure(im, center)
    fx, fy = np.meshgrid(np.fft.fftfreq(64,d=10), np.fft.fftfreq(64,d=10))
    f2 = fx*fx+fy*fy
    band = (f2 >= (1/120)**2) & (f2 <= (1/20)**2)
    fs = (power[band]*2*fx[band]*fy[band]/f2[band]).sum()/power[band].sum()
    fm = complex(z['fft_direction'],fs)
    sm = complex(z['spatial_cos2'],z['spatial_sin2'])
    fa, sa = fm*np.exp(-2j*theta), sm*np.exp(-2j*theta)
    return z | dict(fft_sin2=fs, fft_aligned_real=fa.real, fft_aligned_imag=fa.imag,
                    spatial_aligned_real=sa.real, spatial_aligned_imag=sa.imag), power


def extract(c, gallery):
    rows = []
    for p in c.itertuples():
        for oid in (p.obs_id_a,p.obs_id_b):
            for pol in ('vv','vh'):
                pri = gallery[str(oid)]['images'][pol]['priors']
                s = pri.get('ship_axis')
                if not s: continue
                im = np.load(OUT/'calibration/private_windows'/f'{oid}_{pol}.npz')['sigma_noise_amp']
                for name, center in [('anchor',pri['anchor']),('obb_center',s['center'])]:
                    if min(center[0],center[1],im.shape[1]-center[0],im.shape[0]-center[1]) < 6:
                        rows.append(dict(obs_id=oid,mmsi=p.mmsi,pol=pol,center=name,usable=False))
                        continue
                    z,_ = moments(im,center,s['angle'])
                    # Exact 90-degree pixel permutation, with clockwise image-y convention.
                    rc = (center[1],im.shape[1]-center[0])
                    rz,_ = moments(np.rot90(im),rc,s['angle']-np.pi/2)
                    for k in ('fft_aligned_real','fft_aligned_imag','spatial_aligned_real','spatial_aligned_imag'):
                        assert np.isclose(z[k],rz[k],atol=1e-9), (oid,pol,k,z[k],rz[k])
                    rows.append(dict(obs_id=oid,mmsi=p.mmsi,pol=pol,center=name,usable=True,category=p.class_name_a,
                                     novel_ship=p.novel_ship,geo_ok=p.geo_ok,incidence=pri['radar_incidence'],
                                     radar_axis=pri['radar_axis'],angle=s['angle'],axis_cos2=np.cos(2*s['angle']),
                                     axis_sin2=np.sin(2*s['angle']),cos2relative=np.cos(2*(s['angle']-pri['radar_axis'])),**z))
    e = pd.DataFrame(rows);e.to_csv(OUT/'endpoints.csv',index=False)
    diffs = []
    keys=joint.SPATIAL+joint.TARGETS+['spatial_cos2','spatial_sin2','axis_cos2','axis_sin2','incidence','cos2relative']
    for p in c.itertuples():
        for (pol,center),g in e[e.mmsi.eq(p.mmsi)&e.usable.eq(True)].groupby(['pol','center']):
            if len(g)!=2 or not p.geo_ok:continue
            g=g.set_index('obs_id');a,b=g.loc[p.obs_id_a],g.loc[p.obs_id_b]
            fields = {k:b[k]-a[k] for k in keys}
            for prefix, xr,xi in [('fft','fft_direction','fft_sin2'),('spatial','spatial_cos2','spatial_sin2')]:
                fields[f'{prefix}_unaligned_distance']=abs(complex(b[xr]-a[xr],b[xi]-a[xi]))
                fields[f'{prefix}_aligned_distance']=abs(complex(b[f'{prefix}_aligned_real']-a[f'{prefix}_aligned_real'],b[f'{prefix}_aligned_imag']-a[f'{prefix}_aligned_imag']))
            diffs.append(dict(mmsi=p.mmsi,category=p.class_name_a,pol=pol,center=center,novel_ship=p.novel_ship,
                              product_a=p.product_a,product_b=p.product_b,obs_id_a=p.obs_id_a,obs_id_b=p.obs_id_b,days=p.days,**fields))
    d=pd.DataFrame(diffs);d.to_csv(OUT/'paired_changes.csv',index=False)
    return e,d


def design(g,name):
    cols=joint.SPATIAL+['incidence']
    if 'shape' in name:cols+=['spatial_cos2','spatial_sin2']
    if 'angle' in name:cols+=['cos2relative']
    return g[cols].to_numpy()


def analyze(d):
    old=pd.read_csv(ROOT/'round18_joint_spatial_frequency/paired_changes.csv')
    out, preds, rotation = [],[],[]
    for (pol,center),g in d.groupby(['pol','center']):
        train=old[old.pol.eq(pol)&old.center.eq(center)]
        for target in joint.TARGETS:
            pr={'mean':np.full(len(g),train[target].mean())}
            for name in ('space','space_angle','space_shape','space_shape_angle'):
                scaler=StandardScaler().fit(design(train,name))
                model=Ridge(alpha=10).fit(scaler.transform(design(train,name)),train[target])
                pr[name]=model.predict(scaler.transform(design(g,name)))
            for i,r in enumerate(g.itertuples()):
                preds.append(dict(mmsi=r.mmsi,pol=pol,center=center,target=target,novel_ship=r.novel_ship,
                                  actual=getattr(r,target),**{k:v[i] for k,v in pr.items()}))
            for scope,keep in [('all',np.ones(len(g),dtype=bool)),('novel_ship',g.novel_ship.to_numpy())]:
                h=g.loc[keep]
                out.append(dict(pol=pol,center=center,target=target,scope=scope,n=len(h),
                                rho_angle=spearmanr(h.cos2relative,h[target]).statistic,
                                **{f'mse_{k}':np.mean((v[keep]-h[target].to_numpy())**2) for k,v in pr.items()}))
        for scope,h in [('all',g),('novel_ship',g[g.novel_ship])]:
            for prefix in ('fft','spatial'):
                a,b=h[f'{prefix}_unaligned_distance'],h[f'{prefix}_aligned_distance']
                rotation.append(dict(pol=pol,center=center,scope=scope,representation=prefix,n=len(h),
                                     median_unaligned=a.median(),median_aligned=b.median(),
                                     improved=int((b<a).sum()),rho_residual_abs_angle=spearmanr(b,h.cos2relative.abs()).statistic))
    pd.DataFrame(out).to_csv(OUT/'frozen_transfer.csv',index=False)
    pred_table=pd.DataFrame(preds)
    pred_table.to_csv(OUT/'transfer_predictions.csv',index=False)
    case_loss=pred_table[pred_table.novel_ship & pred_table.center.eq('anchor') & pred_table.target.eq('fft_direction')].copy()
    case_loss['shape_angle_loss_gain']=(case_loss.space_shape-case_loss.actual)**2-(case_loss.space_shape_angle-case_loss.actual)**2
    case_loss.to_csv(OUT/'novel_case_loss.csv',index=False)
    pd.DataFrame(rotation).to_csv(OUT/'rotation_summary.csv',index=False)
    print(pd.DataFrame(out).query("scope=='novel_ship' and center=='anchor'").to_string(index=False),flush=True)
    print(pd.DataFrame(rotation).query("scope=='novel_ship' and center=='anchor'").to_string(index=False),flush=True)


def panels(c,gallery):
    selected={}
    for p in c[c.geo_ok].sort_values(['novel_ship','product_a','obs_id_a'],ascending=[False,True,True]).itertuples():
        state=tuple(bool(gallery[str(i)]['predictions'].get('dual',{}).get('correct')) for i in (p.obs_id_a,p.obs_id_b))
        if state not in selected:selected[state]=p
    items=list(selected.items())
    # Additional mechanism examples chosen after results; not prevalence evidence.
    for oid in (25380,36077):
        p=next(r for r in c.itertuples() if r.obs_id_a==oid)
        state=tuple(bool(gallery[str(i)]['predictions'].get('dual',{}).get('correct')) for i in (p.obs_id_a,p.obs_id_b))
        items.append((state,p))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    records=[]
    for state,p in items:
        canvas=Image.new('RGB',(1040,800),'white');draw=ImageDraw.Draw(canvas)
        for row,(oid,pol) in enumerate([(p.obs_id_a,'vv'),(p.obs_id_b,'vv'),(p.obs_id_a,'vh'),(p.obs_id_b,'vh')]):
            pri=gallery[str(oid)]['images'][pol]['priors']; s=pri.get('ship_axis')
            im=np.load(OUT/'calibration/private_windows'/f'{oid}_{pol}.npz')['sigma_noise_amp']
            other=np.load(OUT/'calibration/private_windows'/f'{p.obs_id_b if oid==p.obs_id_a else p.obs_id_a}_{pol}.npz')['sigma_noise_amp']
            tile=Image.fromarray(np.uint8(np.clip(im/max(im.max(),other.max()),0,1)*255)).convert('RGB')
            td=ImageDraw.Draw(tile);cx,cy=pri['anchor'];td.ellipse((cx-6,cy-6,cx+6,cy+6),outline='orange')
            canvas.paste(tile,(10,row*200+30))
            draw.text((10,row*200+3),f'{oid} {pol}; {p.class_name_a}; stored dual {state}; native pixels',fill='black',font=font)
            if s:
                z,power=moments(im,pri['anchor'],s['angle'])
                spec=np.log1p(np.fft.fftshift(power)/power.max()*1000)/np.log(1001)
                canvas.paste(Image.fromarray(np.uint8(spec*255)).convert('RGB'),(330,row*200+30))
                draw.text((500,row*200+35),f'axis {np.rad2deg(s["angle"]):.1f} deg\nFFT raw moment {z["fft_direction"]:+.3f}, {z["fft_sin2"]:+.3f}\nFFT ship-axis moment {z["fft_aligned_real"]:+.3f}, {z["fft_aligned_imag"]:+.3f}\nFFT radial log-ratio {z["fft_ratio"]:+.3f}',fill='black',font=font)
            else:draw.text((500,row*200+35),'axis unavailable',fill='black',font=font)
        name=f'case_{p.obs_id_a}_{p.obs_id_b}.png';canvas.save(OUT/name)
        records.append(dict(obs_id_a=p.obs_id_a,obs_id_b=p.obs_id_b,mmsi=p.mmsi,category=p.class_name_a,
                            novel_ship=p.novel_ship,correct_a=state[0],correct_b=state[1],panel=name))
    pd.DataFrame(records).to_csv(OUT/'case_selection.csv',index=False)


def main():
    OUT.mkdir(exist_ok=True)
    gallery=json.loads((ROOT.parent/'gallery_data.js').read_text(encoding='utf8').split('=',1)[1].strip().removesuffix(';'))['observations']
    c,views,lookup=cohort(gallery)
    raw.OUT=OUT/'raw';raw.OUT.mkdir(exist_ok=True)
    cal.OUT=OUT/'calibration';cal.OUT.mkdir(exist_ok=True)
    if not (cal.OUT/'conditions.csv').exists():
        raw.recover(views,lookup)
        cal.extract(c,gallery)
    e,d=extract(c,gallery)
    analyze(d);panels(c,gallery)
    full=pd.read_csv(OUT/'cohort.csv')
    summary=dict(candidates=len(full),novel_candidates=int(full.novel_ship.sum()),raw_ready=len(c),
                 geographically_usable=int(c.geo_ok.sum()),usable_novel_before_axis=int((c.geo_ok & c.novel_ship).sum()),
                 acquisition_pairs=d[['product_a','product_b']].drop_duplicates().to_dict('records'),
                 radar_degrees_min=float(np.rad2deg(e.radar_axis.min())),radar_degrees_max=float(np.rad2deg(e.radar_axis.max())),
                 axis_counts=d.groupby(['pol','center','novel_ship']).size().rename('n').reset_index().to_dict('records'),
                 pixel_rotation_check='90-degree joint coordinate check passed',classification_training=False)
    (OUT/'coverage_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print('DONE',flush=True)


if __name__=='__main__':main()
