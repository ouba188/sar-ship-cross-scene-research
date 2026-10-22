"""Check documented tile-grid recovery against actual original chip pixels."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from osgeo import gdal
from scipy.stats import spearmanr

ROOT=Path('E:/临时会话')
HERE=Path(__file__).resolve().parent
AUDIT=ROOT/'scene_aware_classification_20261018/data_image_review/ports24'

def corr(a,b):
    if min(np.std(a),np.std(b))<1e-5:return float('nan')
    return float(np.corrcoef(a.ravel(),b.ravel())[0,1])

def main():
    gdal.UseExceptions();gdal.SetConfigOption('GDAL_PAM_ENABLED','NO');gdal.SetCacheMax(64*1024**2)
    m=pd.read_csv(ROOT/'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv',low_memory=False)
    v=pd.read_csv(ROOT/'chip_quality_20261017/ais_metadata/view_metadata.csv',low_memory=False)
    a=pd.read_csv(AUDIT/'product_asset_availability.csv');r=pd.read_csv(AUDIT/'local_rasters.csv')
    products=set(a[a.full&a.full_candidates]['product'])
    v=v[v['product'].isin(products)&v.pol.eq('vv')&v.mmsi.notna()].copy()
    v['known']=v.chip_transform_json.notna()
    # Cover products before selecting on match quality or downstream labels.
    sample=v.sort_values('instance_id').groupby(['product','known'],sort=True).head(3)
    rasters=r[~r.tile&r.pol.eq('vv')].sort_values('path').groupby('product').path.apply(list).to_dict()
    results=[];panels=[]
    for prod,part in sample.groupby('product',sort=True):
        paths=sorted(rasters.get(prod,[]),key=lambda p:('tif_local_done' not in p,p))
        path=next((p for p in paths if Path(p).exists()),None)
        if path is None:continue
        ds=gdal.Open(path);gt=ds.GetGeoTransform()
        for row in part.itertuples():
            match=re.search(r'__r(\d+)_c(\d+)\.tif$',str(row.source_tiff))
            original=np.asarray(Image.open(row.image_path).convert('L'))
            if row.known:
                cg=json.loads(row.chip_transform_json)
                x,y=gdal.ApplyGeoTransform(gdal.InvGeoTransform(gt),cg[0],cg[3])
                method='existing_georef'
            elif match and 'int8_grid_4x4_o256' in str(row.source_tiff):
                rr,cc=map(int,match.groups())
                if not (1<=rr<=4 and 1<=cc<=4):continue
                x=max(0,int(ds.RasterXSize*(cc-1)/4)-128)+row.crop_left
                y=max(0,int(ds.RasterYSize*(rr-1)/4)-128)+row.crop_top
                method='documented_grid_from_current_full'
            else:
                continue
            x,y=int(round(x)),int(round(y));h,w=original.shape
            if x<0 or y<0 or x+w>ds.RasterXSize or y+h>ds.RasterYSize:continue
            actual=ds.ReadAsArray(x,y,w,h)
            border=np.ones((h,w),bool);border[h//4:3*h//4,w//4:3*w//4]=False
            pear=corr(original,actual);bg=corr(original[border],actual[border])
            rank=float(spearmanr(original.ravel(),actual.ravel()).statistic) if np.std(actual)>0 else float('nan')
            results.append(dict(instance_id=row.instance_id,product=prod,port=row.port,known=row.known,method=method,
                source_tiff=row.source_tiff,full_path=path,original_chip=row.image_path,x=x,y=y,w=w,h=h,
                pearson=pear,border_pearson=bg,spearman=rank,exact=bool(np.array_equal(original,actual))))
            if (not row.known and len(panels)<6):panels.append((row.instance_id,original,actual,pear,bg))
        ds=None
    frame=pd.DataFrame(results);frame.to_csv(HERE/'recovery_probe.csv',index=False)
    canvas=Image.new('RGB',(650,190*len(panels)),'white');draw=ImageDraw.Draw(canvas)
    for i,(iid,old,new,co,bg) in enumerate(panels):
        draw.text((6,i*190+4),f'{iid} | native pixels | full corr={co:.4f}, border={bg:.4f}',fill='black')
        canvas.paste(Image.fromarray(old).convert('RGB'),(8,i*190+35))
        canvas.paste(Image.fromarray(new).convert('RGB'),(210,i*190+35))
        draw.text((8,i*190+20),'Original chip',fill='black');draw.text((210,i*190+20),'Recovered whole-image window',fill='black')
    canvas.save(HERE/'recovery_probe.png')
    print(frame.groupby(['known','method'])[['pearson','border_pearson','spearman']].describe().to_string())
    print('n',len(frame),'matched both >.8',frame.assign(ok=frame.pearson.gt(.8)&frame.border_pearson.gt(.8)).groupby('known').ok.agg(['sum','count']).to_string())

if __name__=='__main__':main()
