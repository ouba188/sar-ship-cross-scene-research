"""Bounded LUT calibration/noise sensitivity on the recovered paired cohort."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr, rankdata
from osgeo import gdal
from PIL import Image, ImageDraw, ImageFont
import raw_pair_probe as raw

ROOT=raw.ROOT
OUT=ROOT/'round16_calibrated_relation'


def vec(e,key):
    return np.fromstring(e.findtext(key),sep=' ')


def table(root,vector,value):
    es=root.findall('.//'+vector)
    lines=np.array([float(e.findtext('line')) for e in es])
    pixels=vec(es[0],'pixel')
    assert all(np.array_equal(vec(e,'pixel'),pixels) for e in es)
    values=np.array([vec(e,value) for e in es])
    fn=RegularGridInterpolator((lines,pixels),values,bounds_error=True)
    # Detect swapped axes/incorrect XML value extraction using real LUT nodes.
    ix=np.array([0,len(pixels)//2,len(pixels)-1])
    iy=np.array([0,len(lines)//2,len(lines)-1])
    pts=np.array([(lines[y],pixels[x]) for y in iy for x in ix])
    expected=np.array([values[y,x] for y in iy for x in ix])
    assert np.allclose(fn(pts),expected,rtol=0,atol=1e-10)
    return fn,lines


def coords(ds,r,shape):
    h,w=shape;gt=json.loads(r.chip_transform_json)
    yy,xx=np.indices(shape)
    mx=gt[0]+(xx+.5)*gt[1]+(yy+.5)*gt[2]
    my=gt[3]+(xx+.5)*gt[4]+(yy+.5)*gt[5]
    t=gdal.Transformer(ds,None,[f'DST_SRS=EPSG:{int(r.epsg)}'])
    pts,ok=t.TransformPoints(True,np.c_[mx.ravel(),my.ravel(),np.zeros(h*w)].tolist())
    assert all(ok)
    return np.floor(np.asarray(pts)[:,:2]).astype(int)


def region_masks(shape,p):
    yy,xx=np.indices(shape);cx,cy=p['anchor']
    valid=(xx>=6)&(xx<shape[1]-6)&(yy>=6)&(yy<shape[0]-6)
    return {f'disk{r*10}':(((xx+.5-cx)**2+(yy+.5-cy)**2<=r*r)&valid) for r in (4,6,8)}


def extract(cohort,gallery,views_override=None):
    gdal.UseExceptions();gdal.SetConfigOption('GDAL_PAM_ENABLED','NO');gdal.SetCacheMax(64*1024*1024)
    views=(pd.read_csv(raw.WORK/'chip_quality_20261017/ais_metadata/view_metadata.csv',low_memory=False)
           if views_override is None else views_override.copy())
    ids=set(cohort.instance_id_a)|set(cohort.instance_id_b)
    views=views[views.instance_id.isin(ids)]
    lookup={o['instance_id']:o for o in gallery.values()}
    rows,provenance,conditions=[],[],[]
    cache=OUT/'private_windows';cache.mkdir(exist_ok=True)
    for (prod,pol),group in views.groupby(['product','pol']):
        safe=raw.SAFE/f'{prod}.SAFE'
        measurement=next((safe/'measurement').glob(f'*-{pol}-*.tiff'))
        cal=next((safe/'annotation/calibration').glob(f'calibration-*-{pol}-*.xml'))
        noise=next((safe/'annotation/calibration').glob(f'noise-*-{pol}-*.xml'))
        ann=next((safe/'annotation').glob(f'*-{pol}-*.xml'))
        assert ET.parse(ann).findtext('.//thermalNoiseCorrectionPerformed')=='false'
        cf,_=table(ET.parse(cal),'calibrationVector','sigmaNought')
        nf,nlines=table(ET.parse(noise),'noiseRangeVector','noiseRangeLut')
        az=ET.parse(noise).findall('.//noiseAzimuthVector')
        ds=gdal.Open(str(measurement))
        for r in group.itertuples():
            o=lookup[r.instance_id];oid=o['id'];p=o['images'][pol]['priors']
            dn=np.load(raw.OUT/'private_windows'/f'{oid}_{pol}.npz')['dn']
            xy=coords(ds,r,dn.shape);x,y=xy.T
            # Re-read a small source rectangle: prevents applying LUTs to mismapped cached pixels.
            x0,y0,x1,y1=int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)
            src=ds.ReadAsArray(x0,y0,x1-x0,y1-y0)
            assert np.array_equal(src[y-y0,x-x0].reshape(dn.shape),dn)
            lut=cf(np.c_[y,x]).reshape(dn.shape)
            noise_range=nf(np.c_[y,x])
            factor=np.zeros(len(x));coverage=np.zeros(len(x),dtype=int)
            for e in az:
                ylo,yhi,xlo,xhi=[int(e.findtext(k)) for k in ('firstAzimuthLine','lastAzimuthLine','firstRangeSample','lastRangeSample')]
                use=(x>=xlo)&(x<=xhi)&(y>=ylo)&(y<=yhi)
                if not use.any():continue
                # Actual products span every range-vector line in each azimuth block.
                assert ylo<=nlines.min() and yhi>=nlines.max()
                lines,values=vec(e,'line'),vec(e,'noiseAzimuthLut')
                assert y[use].min()>=lines.min() and y[use].max()<=lines.max()
                factor[use]=np.interp(y[use],lines,values);coverage[use]+=1
            assert (coverage==1).all() and (factor>0).all() and (lut>0).all()
            noise_power=(noise_range*factor).reshape(dn.shape)
            power=dn*dn
            sigma_amp=dn/lut
            sigma_noise_amp=np.sqrt(np.maximum(power-noise_power,1e-5))/lut
            reps=dict(dn=dn,sigma_amp=sigma_amp,sigma_noise_amp=sigma_noise_amp,sigma_power=power/lut**2)
            np.savez_compressed(cache/f'{oid}_{pol}.npz',**reps,noise_fraction=noise_power/np.maximum(power,1))
            masks=region_masks(dn.shape,p)
            sh=p.get('ship_axis');angle=sh['angle'] if sh else np.nan
            cross=o['images']['vv']['priors'].get('ship_axis')
            radar=p.get('radar_axis',np.nan)
            for region,m in masks.items():
                for name,im in reps.items():
                    z=raw.measure(im,m)
                    rows.append(dict(obs_id=oid,mmsi=o['mmsi'],category=o['category'],pol=pol,region=region,representation=name,**z))
            m=masks['disk60']
            provenance.append(dict(obs_id=oid,pol=pol,product=prod,calibration_path=str(cal),noise_path=str(noise),
                lut_relative_span=float((lut.max()-lut.min())/np.median(lut)),
                center_noise_to_power_median=float(np.median(noise_power[m]/np.maximum(power[m],1))),
                center_floored_fraction=float(np.mean(power[m]<noise_power[m])),
                all_floored_fraction=float(np.mean(power<noise_power)),cached_dn_exact=True))
            conditions.append(dict(obs_id=oid,pol=pol,mmsi=o['mmsi'],category=o['category'],
                cos2relative=float(np.cos(2*(angle-radar))),
                vv_axis_relative=float(np.cos(2*(cross['angle']-radar))) if cross else np.nan,
                incidence=p.get('radar_incidence',np.nan),radar_axis=radar,
                ship_axis=angle,match_status=p['match_status']))
        ds=None
    for name,data in [('measurements',rows),('radiometry_provenance',provenance),('conditions',conditions)]:
        pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
    return pd.DataFrame(rows),pd.DataFrame(conditions)


def correlation(x,y):
    return float(spearmanr(x,y).statistic)


def partial(x,y,c):
    design=np.c_[np.ones(len(x)),rankdata(c)]
    rx=rankdata(x);ry=rankdata(y)
    rx-=design@np.linalg.lstsq(design,rx,rcond=None)[0]
    ry-=design@np.linalg.lstsq(design,ry,rcond=None)[0]
    return float(np.corrcoef(rx,ry)[0,1])


def analyze(cohort,measures,conditions):
    m=measures.merge(conditions,on=['obs_id','pol','mmsi','category'],validate='many_to_one')
    rows=[]
    for pair in cohort.itertuples():
        for (pol,region,rep),g in m[m.mmsi.eq(pair.mmsi)].groupby(['pol','region','representation']):
            assert len(g)==2
            g=g.set_index('obs_id');a,b=g.loc[pair.obs_id_a],g.loc[pair.obs_id_b]
            rows.append(dict(mmsi=pair.mmsi,category=pair.class_name_a,pol=pol,region=region,representation=rep,
                delta_ratio=b.log_ratio-a.log_ratio,delta_cos=b.cos2relative-a.cos2relative,
                delta_vv_cos=b.vv_axis_relative-a.vv_axis_relative,delta_incidence=b.incidence-a.incidence))
    pairs=pd.DataFrame(rows);pairs.to_csv(OUT/'paired_changes.csv',index=False)
    stats=[]
    for (pol,region,rep),g in pairs.groupby(['pol','region','representation']):
        for group,h in [('all',g),('product_chemical_tanker',g[g.category.eq('product_chemical_tanker')])]:
            for axis,col in [('same_pol_axis','delta_cos'),('same_pol_common','delta_cos'),('vv_axis_common','delta_vv_cos')]:
                required=[col,'delta_ratio','delta_incidence']
                if axis.endswith('common'):required+=['delta_cos','delta_vv_cos']
                h2=h.dropna(subset=required)
                if len(h2)<10:continue
                x,y,c=h2[col].to_numpy(),h2.delta_ratio.to_numpy(),h2.delta_incidence.to_numpy()
                loo=[correlation(np.delete(x,i),np.delete(y,i)) for i in range(len(x))]
                stats.append(dict(pol=pol,region=region,representation=rep,group=group,axis=axis,n=len(h2),
                    rho=correlation(x,y),partial_rank_incidence=partial(x,y,c),loo_min=min(loo),loo_max=max(loo)))
    stats=pd.DataFrame(stats);stats.to_csv(OUT/'association_summary.csv',index=False)
    effects=[]
    for (pol,region),g in pairs.groupby(['pol','region']):
        z=g.pivot(index='mmsi',columns='representation',values='delta_ratio')
        for rep in ('sigma_amp','sigma_noise_amp','sigma_power'):
            effects.append(dict(pol=pol,region=region,representation=rep,n=len(z),
                median_abs_difference=float((z[rep]-z.dn).abs().median()),
                max_abs_difference=float((z[rep]-z.dn).abs().max()),
                same_sign=int((z[rep]*z.dn>0).sum()),rho=correlation(z[rep],z.dn)))
    pd.DataFrame(effects).to_csv(OUT/'radiometry_effects.csv',index=False)
    print(stats[stats.pol.eq('vh')&stats.region.eq('disk60')].to_string(index=False))


def panels(gallery):
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for a,b in [(8368,8369),(25750,25752),(28337,28338),(7324,7326)]:
        panel=Image.new('RGB',(960,720),'white');draw=ImageDraw.Draw(panel)
        for i,(oid,pol) in enumerate([(a,'vv'),(b,'vv'),(a,'vh'),(b,'vh')]):
            z=np.load(OUT/'private_windows'/f'{oid}_{pol}.npz')
            other=np.load(OUT/'private_windows'/f'{b if oid==a else a}_{pol}.npz')
            p=gallery[str(oid)]['images'][pol]['priors']
            for j,name in enumerate(['dn','sigma_amp','sigma_noise_amp','noise_fraction']):
                im=z[name]
                if name=='noise_fraction':im=np.clip(im,0,1)
                else:
                    scale=max(z[name].max(),other[name].max());im=im/scale
                tile=Image.fromarray((np.clip(im,0,1)*255).round().astype('uint8')).convert('RGB')
                td=ImageDraw.Draw(tile);cx,cy=p['anchor'];td.ellipse((cx-6,cy-6,cx+6,cy+6),outline=(255,160,0))
                panel.paste(tile,(j*240+(240-tile.width)//2,i*180+30))
                draw.text((j*240+4,i*180+5),f'{oid} {pol} {name}',fill='black',font=font)
        panel.save(OUT/f'case_{a}_{b}.png')


def main():
    OUT.mkdir(exist_ok=True)
    cohort=pd.read_csv(raw.OUT/'cohort.csv',dtype={'mmsi':str})
    gallery=json.loads((ROOT.parent/'gallery_data.js').read_text(encoding='utf8').split('=',1)[1].strip().removesuffix(';'))['observations']
    measures,conditions=extract(cohort,gallery)
    analyze(cohort,measures,conditions);panels(gallery)


if __name__=='__main__':main()
