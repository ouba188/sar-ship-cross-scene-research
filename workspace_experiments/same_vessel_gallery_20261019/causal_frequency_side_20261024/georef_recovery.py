"""Recover exact chip windows from live scene TIFFs; never modify source metadata."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import sys
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from osgeo import gdal, osr
from pyproj import Transformer
from scipy.stats import spearmanr
import raw_pair_probe as raw
import calibrated_relation as cal
import external_rotation as ext

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'round20_georef_recovery'
sys.path.insert(0,str(raw.WORK/'scene_aware_classification_20261018/f0921_priority/formal_queue_bridge/orbit_geometry_readout_20261019'))
import recover_orbit as orbit


def recover_views(c,gallery):
    v=pd.read_csv(raw.WORK/'chip_quality_20261017/ais_metadata/view_metadata.csv',low_memory=False)
    ids=set(c.instance_id_a)|set(c.instance_id_b)
    v=v[v.instance_id.isin(ids)].copy()
    lookup={o['instance_id']:o for o in gallery.values()}
    h=json.loads((raw.WORK/'visual_reliable_baseline/e48j_24port_prep/scene_headers.json').read_text(encoding='utf8'))['headers']
    receipts=[]
    gdal.UseExceptions();gdal.SetConfigOption('GDAL_PAM_ENABLED','NO');gdal.SetCacheMax(64*1024*1024)
    for index,r in v.iterrows():
        if pd.notna(r.epsg) and pd.notna(r.chip_transform_json):
            continue
        key=f'{r.port}|{r["product"]}|{r.pol.upper()}'
        info=h.get(key)
        base=dict(instance_id=r.instance_id,obs_id=lookup[r.instance_id]['id'],pol=r.pol,source_tiff=r.source_tiff)
        if not info or not Path(info['path']).is_file():
            receipts.append(base|dict(status='missing_live_scene'));continue
        ds=gdal.Open(info['path']);gt=list(ds.GetGeoTransform())
        assert [ds.RasterXSize,ds.RasterYSize]==info['size'] and np.allclose(gt,info['gt'],atol=.01,rtol=0)
        srs=osr.SpatialReference(wkt=ds.GetProjection());srs.AutoIdentifyEPSG();epsg=int(srs.GetAuthorityCode(None))
        match=re.search(r'__r(\d+)_c(\d+)\.tif$',r.source_tiff)
        assert match and 'int8_grid_4x4_o256' in r.source_tiff
        row,col=map(int,match.groups());assert 1<=row<=4 and 1<=col<=4
        tilex=max(0,int(ds.RasterXSize*(col-1)/4)-128)
        tiley=max(0,int(ds.RasterYSize*(row-1)/4)-128)
        assert float(r.crop_left).is_integer() and float(r.crop_top).is_integer()
        x,y=tilex+int(r.crop_left),tiley+int(r.crop_top)
        png=np.asarray(Image.open(r.image_path).convert('L'))
        ht,w=png.shape
        assert 0<=x and 0<=y and x+w<=ds.RasterXSize and y+ht<=ds.RasterYSize
        patch=ds.ReadAsArray(x,y,w,ht)
        exact=bool(np.array_equal(patch,png));error=np.abs(patch.astype(float)-png.astype(float))
        rank=spearmanr(patch.ravel(),png.ravel()).statistic
        neighbours=[]
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            shifted=ds.ReadAsArray(x+dx,y+dy,w,ht)
            neighbours.append(spearmanr(shifted.ravel(),png.ravel()).statistic)
        yy,xx=np.indices(png.shape);cx,cy=lookup[r.instance_id]['images'][r.pol]['priors']['anchor']
        center=(xx+.5-cx)**2+(yy+.5-cy)**2<=36
        local_rank=spearmanr(patch[center],png[center]).statistic
        supported=bool(rank>=.9 and local_rank>=.85 and rank-max(neighbours)>=.02)
        receipt=base|dict(status='exact_pixels' if exact else ('spatially_supported_not_exact' if supported else 'unresolved_pixel_mismatch'),live_scene=info['path'],epsg=epsg,
                          x=x,y=y,width=w,height=ht,mae=float(error.mean()),max_error=float(error.max()))
        receipt.update(rank_rho=rank,center_rank=local_rank,best_shift_rank=max(neighbours),rank_gap=rank-max(neighbours))
        receipts.append(receipt)
        if exact or supported:
            cg=[gt[0]+x*gt[1]+y*gt[2],gt[1],gt[2],gt[3]+x*gt[4]+y*gt[5],gt[4],gt[5]]
            v.loc[index,'epsg']=epsg;v.loc[index,'chip_transform_json']=json.dumps(cg)
            v.loc[index,'georef_source']=receipt['status']
            v.loc[index,'georef_header_path']=info['path']
        ds=None
    pd.DataFrame(receipts).to_csv(OUT/'recovery_receipts.csv',index=False)
    v.to_csv(OUT/'recovered_views.csv',index=False)
    print(pd.DataFrame(receipts).groupby('status').size().to_string(),flush=True)
    return v,lookup


def geometry(c,v,gallery):
    records=[];annotations={};locs=[]
    lookup={o['instance_id']:o for o in gallery.values()}
    for r in v.itertuples():
        if pd.isna(r.epsg) or pd.isna(r.chip_transform_json):continue
        gt=json.loads(r.chip_transform_json);o=lookup[r.instance_id];pri=o['images'][r.pol]['priors']
        cx,cy=pri['anchor'];mx,my=gt[0]+cx*gt[1]+cy*gt[2],gt[3]+cx*gt[4]+cy*gt[5]
        togeo=Transformer.from_crs(int(r.epsg),4326,always_xy=True)
        lon,lat=togeo.transform(mx,my)
        tomap=Transformer.from_crs(4326,int(r.epsg),always_xy=True)
        if r.product not in annotations:
            path=next((raw.SAFE/f'{r.product}.SAFE'/'annotation').glob('*-vv-*.xml'))
            annotations[r.product]=orbit.load_annotation(path,r.product,path)
        a=annotations[r.product];points=a['points']
        xy=np.column_stack(tomap.transform(points[:,0],points[:,1]))
        t,values=orbit.interpolate_grid(xy,a['times'],np.array([mx,my]),points[:,2:])
        pos,vel=orbit.orbit_at(a['vectors'],t,a['start'],a['stop'])
        los,flight,inc=orbit.look_geometry(pos,vel,lon,lat)
        jac=orbit.enu_to_map_jacobian(lon,lat,tomap)
        _,_,angle=orbit.image_direction(los[:2],jac,np.array([[gt[1],gt[2]],[gt[4],gt[5]]]))
        old=pri.get('radar_axis')
        records.append(dict(obs_id=o['id'],pol=r.pol,lon=lon,lat=lat,angle=angle%np.pi,incidence=inc,
                            grid_incidence=values[2],annotation_path=a['annotation_path'],old_angle=old,
                            difference_deg=orbit.axis_difference(angle,old) if old is not None else np.nan,
                            old_incidence=pri.get('radar_incidence')))
        # Keep original known geometry to preserve the previous frozen readout protocol.
        if old is None:
            pri.update(radar_axis=angle%np.pi,radar_incidence=float(values[2]),radar_source=a['annotation_path'])
        locs.append(dict(instance_id=r.instance_id,pol=r.pol,lon=lon,lat=lat))
    pd.DataFrame(records).to_csv(OUT/'recovered_geometry.csv',index=False)
    loc=pd.DataFrame(locs);checks=[]
    for instance,g in loc.groupby('instance_id'):
        if len(g)!=2:continue
        a,b=g.iloc[0],g.iloc[1]
        distance=orbit.GEOD.inv(a.lon,a.lat,b.lon,b.lat)[2]
        checks.append(dict(instance_id=instance,distance_m=distance))
    checks=pd.DataFrame(checks);checks.to_csv(OUT/'crosspol_locations.csv',index=False)
    valid=set(checks[checks.distance_m.le(100)].instance_id)
    available=set(checks.instance_id)
    c=c.copy();c['raw_ready']=c.instance_id_a.isin(available)&c.instance_id_b.isin(available)
    c['geo_ok']=c.instance_id_a.isin(valid)&c.instance_id_b.isin(valid)
    c.to_csv(OUT/'cohort.csv',index=False)
    return c


def panels(c,gallery):
    chosen={}
    for p in c[c.geo_ok].sort_values(['product_a','obs_id_a']).itertuples():
        state=tuple(gallery[str(i)]['predictions'].get('dual',{}).get('correct') for i in (p.obs_id_a,p.obs_id_b))
        if state not in chosen:chosen[state]=p
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14);cases=[]
    for state,p in chosen.items():
        canvas=Image.new('RGB',(1040,800),'white');draw=ImageDraw.Draw(canvas)
        for row,(oid,pol) in enumerate([(p.obs_id_a,'vv'),(p.obs_id_b,'vv'),(p.obs_id_a,'vh'),(p.obs_id_b,'vh')]):
            pri=gallery[str(oid)]['images'][pol]['priors'];s=pri.get('ship_axis')
            im=np.load(OUT/'calibration/private_windows'/f'{oid}_{pol}.npz')['sigma_noise_amp']
            other=np.load(OUT/'calibration/private_windows'/f'{p.obs_id_b if oid==p.obs_id_a else p.obs_id_a}_{pol}.npz')['sigma_noise_amp']
            tile=Image.fromarray(np.uint8(np.clip(im/max(im.max(),other.max()),0,1)*255)).convert('RGB')
            td=ImageDraw.Draw(tile);cx,cy=pri['anchor'];td.ellipse((cx-6,cy-6,cx+6,cy+6),outline='orange')
            canvas.paste(tile,(10,row*200+30))
            draw.text((10,row*200+4),f'{oid} {pol}; {p.class_name_a}; archived dual {state}',fill='black',font=font)
            if s:
                z,power=ext.moments(im,pri['anchor'],s['angle'])
                sp=np.log1p(np.fft.fftshift(power)/power.max()*1000)/np.log(1001)
                canvas.paste(Image.fromarray(np.uint8(sp*255)).convert('RGB'),(330,row*200+30))
                draw.text((500,row*200+35),f'axis {np.rad2deg(s["angle"]):.1f} deg\nradar {np.rad2deg(pri["radar_axis"]):.1f} deg\nFFT moment {z["fft_direction"]:+.3f}, {z["fft_sin2"]:+.3f}\nship-axis {z["fft_aligned_real"]:+.3f}, {z["fft_aligned_imag"]:+.3f}\nradial log ratio {z["fft_ratio"]:+.3f}',fill='black',font=font)
        name=f'case_{p.obs_id_a}_{p.obs_id_b}.png';canvas.save(OUT/name)
        cases.append(dict(obs_id_a=p.obs_id_a,obs_id_b=p.obs_id_b,category=p.class_name_a,correct_a=state[0],correct_b=state[1],panel=name))
    pd.DataFrame(cases).to_csv(OUT/'case_selection.csv',index=False)


def main():
    OUT.mkdir(exist_ok=True)
    gallery=json.loads((ROOT.parent/'gallery_data.js').read_text(encoding='utf8').split('=',1)[1].strip().removesuffix(';'))['observations']
    c=pd.read_csv(ROOT/'round19_external_rotation/cohort.csv',dtype={'mmsi':str})
    c=c[~c.raw_ready].copy()
    v,lookup=recover_views(c,gallery)
    c=geometry(c,v,gallery)
    c=c[c.raw_ready].copy();ids=set(c.instance_id_a)|set(c.instance_id_b)
    v=v[v.instance_id.isin(ids)]
    print('Recovered pairs',len(c),'geo usable',c.geo_ok.sum(),flush=True)
    if not len(c):
        print('No usable raw pairs; retained coverage records only.',flush=True)
        return
    raw.OUT=OUT/'raw';raw.OUT.mkdir(exist_ok=True)
    cal.OUT=OUT/'calibration';cal.OUT.mkdir(exist_ok=True)
    expected=[cal.OUT/'private_windows'/f'{lookup[r.instance_id]["id"]}_{r.pol}.npz' for r in v.itertuples()]
    if not all(p.is_file() for p in expected):
        raw.recover(v,lookup)
    cal.extract(c,gallery,views_override=v)
    ext.OUT=OUT
    e,d=ext.extract(c,gallery);ext.analyze(d);panels(c,gallery)
    summarize()
    print('DONE',flush=True)


def summarize():
    c=pd.read_csv(OUT/'cohort.csv')
    a=pd.read_csv(OUT/'recovery_receipts.csv')
    d=pd.read_csv(OUT/'paired_changes.csv')
    old=pd.read_csv(ROOT/'round19_external_rotation/paired_changes.csv')
    combined=pd.concat([old[old.novel_ship],d],ignore_index=True)
    combined.to_csv(OUT/'combined_novel_pairs.csv',index=False)
    g=pd.read_csv(OUT/'recovered_geometry.csv')
    report=dict(missing_pairs_requested=len(c),new_view_attempts=len(a),
                recovery_status=a.status.value_counts().to_dict(),both_pol_both_dates_recovered=int(c.raw_ready.sum()),
                usable_new_pairs=int((c.raw_ready&c.geo_ok).sum()),remaining_missing_pairs=int((~c.raw_ready).sum()),
                isolated_crosspol_pairs=int((c.raw_ready&~c.geo_ok).sum()),
                known_geometry_max_axis_difference_deg=float(g.difference_deg.max()),
                known_geometry_max_grid_incidence_difference=float((g.grid_incidence-g.old_incidence).abs().max()),
                new_product_pairs=d[['product_a','product_b']].drop_duplicates().to_dict('records'),
                cumulative_novel_ships=int(combined.mmsi.nunique()),
                cumulative_product_pairs=len(combined[['product_a','product_b']].drop_duplicates()),
                cumulative_products=sorted(set(combined.product_a)|set(combined.product_b)),
                raw_pixel_replay_exact=False,classification_training=False)
    (OUT/'coverage_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':main()
