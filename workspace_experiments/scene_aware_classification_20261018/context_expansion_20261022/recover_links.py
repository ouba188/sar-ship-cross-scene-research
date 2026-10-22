"""Recover chip positions using recorded grids plus pixel correspondence, no labels guessed."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from osgeo import gdal,osr
from probe_recovery import ROOT,HERE,AUDIT,corr

def locate(row,ds):
    gt=ds.GetGeoTransform()
    if pd.notna(row.chip_transform_json):
        cg=json.loads(row.chip_transform_json);x,y=cg[0],cg[3]
        src=osr.SpatialReference();src.ImportFromEPSG(int(row.epsg));src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst=ds.GetSpatialRef();dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not src.IsSame(dst):x,y,_=osr.CoordinateTransformation(src,dst).TransformPoint(x,y)
        x,y=gdal.ApplyGeoTransform(gdal.InvGeoTransform(gt),x,y)
        return round(x),round(y),'existing_georef'
    match=re.search(r'__r(\d+)_c(\d+)\.tif$',str(row.source_tiff))
    if match and 'int8_grid_4x4_o256' in str(row.source_tiff):
        rr,cc=map(int,match.groups())
        if 1<=rr<=4 and 1<=cc<=4:
            x=max(0,int(ds.RasterXSize*(cc-1)/4)-128)+row.crop_left
            y=max(0,int(ds.RasterYSize*(rr-1)/4)-128)+row.crop_top
            return round(x),round(y),'documented_grid_pixel_verified'
    return None

def main():
    gdal.UseExceptions();gdal.SetConfigOption('GDAL_PAM_ENABLED','NO');gdal.SetCacheMax(64*1024**2)
    m=pd.read_csv(ROOT/'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv',low_memory=False)
    v=pd.read_csv(ROOT/'chip_quality_20261017/ais_metadata/view_metadata.csv',low_memory=False)
    a=pd.read_csv(AUDIT/'product_asset_availability.csv');r=pd.read_csv(AUDIT/'local_rasters.csv')
    products=set(a[a.full&a.full_candidates]['product'])
    v=v[v['product'].isin(products)&v.mmsi.notna()&~v.instance_id.eq('pair_0031237')].copy()
    rasters=r[~r.tile].groupby(['product','pol']).path.apply(list).to_dict()
    rows=[];excluded=[]
    for prod,part in v.groupby('product',sort=True):
        ds={};paths={}
        for pol in ['vv','vh']:
            candidates=sorted(rasters.get((prod,pol),[]),key=lambda p:('tif_local_done' not in p,p))
            paths[pol]=next((p for p in candidates if Path(p).exists()),None)
            if paths[pol]:ds[pol]=gdal.Open(paths[pol])
        if len(ds)!=2:continue
        if ds['vv'].GetGeoTransform()!=ds['vh'].GetGeoTransform() or not ds['vv'].GetSpatialRef().IsSame(ds['vh'].GetSpatialRef()):continue
        for row in part.itertuples():
            d=ds[row.pol];loc=locate(row,d)
            if loc is None:
                excluded.append(dict(instance_id=row.instance_id,pol=row.pol,reason='no_documented_grid'));continue
            x,y,method=loc
            im=np.asarray(Image.open(row.image_path).convert('L'));h,w=im.shape
            if min(x,y)<0 or x+w>d.RasterXSize or y+h>d.RasterYSize or max(h,w)>512:
                excluded.append(dict(instance_id=row.instance_id,pol=row.pol,reason='outside_or_oversize'));continue
            patch=d.ReadAsArray(x,y,w,h)
            border=np.ones((h,w),bool);border[h//4:3*h//4,w//4:3*w//4]=False
            if min(np.std(im),np.std(patch),np.std(im[border]),np.std(patch[border]))==0:
                excluded.append(dict(instance_id=row.instance_id,pol=row.pol,reason='constant_patch'));continue
            rho=float(spearmanr(im.ravel(),patch.ravel()).statistic)
            bg=float(spearmanr(im[border],patch[border]).statistic)
            # Same-scene speckle/texture correspondence matters, not just a centered bright object.
            good=rho>=.85 and bg>=.85
            rows.append(dict(instance_id=row.instance_id,product=prod,pol=row.pol,port=row.port,mmsi=str(int(row.mmsi)),
                x=x,y=y,w=w,h=h,method=method,spearman=rho,border_spearman=bg,pearson=corr(im,patch),
                verified=good,full_path=paths[row.pol],original_chip=row.image_path,
                source_tiff=row.source_tiff,pixel_m=abs(d.GetGeoTransform()[1])))
        ds.clear()
        print(prod[-4:],len(rows),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(HERE/'recovered_views.csv',index=False)
    pd.DataFrame(excluded).to_csv(HERE/'recovery_excluded.csv',index=False)
    good=frame[frame.verified]
    vv=good[good.pol.eq('vv')].set_index('instance_id');vh=good[good.pol.eq('vh')].set_index('instance_id')
    ids=vv.index.intersection(vh.index)
    out=m[m.instance_id.isin(ids)].copy()
    for pol,src in [('vv',vv),('vh',vh)]:
        for col in ['x','y','w','h','full_path','original_chip','spearman','border_spearman','method','pixel_m']:
            out[col+'_'+pol]=out.instance_id.map(src[col])
    out['pass_group']=out['product'].map(lambda p:'_'.join(p.split('_')[6:8]))
    out.to_csv(HERE/'linked_targets.csv',index=False)
    result=dict(recovered_views=len(frame),verified_views=int(frame.verified.sum()),dual_targets=len(out),
        mmsi=out.mmsi.nunique(),products=out['product'].nunique(),passes=out.pass_group.nunique(),ports=out.port.nunique(),
        classes=out.class_name.value_counts().to_dict(),new_georef_views=int((frame.verified&frame.method.ne('existing_georef')).sum()),
        rule='Both full-image and outer-half-area Spearman >=0.85; no re-centering; fixed documented tile grid; no label/identity inference')
    (HERE/'recovery_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
