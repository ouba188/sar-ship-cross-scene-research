"""Check the two observed image-boundary failures and the deployed no-context path."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from osgeo import gdal

HERE=Path(__file__).resolve().parent/'ports24'


def main():
    gdal.UseExceptions()
    t=pd.read_csv(HERE/'targets.csv',dtype={'mmsi':str})
    v=pd.read_csv('E:/临时会话/chip_quality_20261017/ais_metadata/view_metadata.csv',low_memory=False)
    v=v[v.pol.eq('vv')].set_index('instance_id')
    datasets={p:gdal.Open(p) for p in t.raster_vv.unique()}
    max_corner_error=0.
    for row in t.itertuples():
        source=v.loc[row.instance_id]
        im=Image.open(source.image_path)
        original=json.loads(source.chip_transform_json)
        ds=datasets[row.raster_vv]
        assert int(source.epsg)==int(ds.GetSpatialRef().GetAuthorityCode(None))
        x,y,w,h=json.loads(row.target_window)
        for a,b,c,d in [(0,0,x,y),(im.width,0,x+w,y),(0,im.height,x,y+h),(im.width,im.height,x+w,y+h)]:
            expected=np.asarray(gdal.ApplyGeoTransform(original,a,b))
            actual=np.asarray(gdal.ApplyGeoTransform(ds.GetGeoTransform(),c,d))
            max_corner_error=max(max_corner_error,float(abs(actual-expected).max()))
    assert max_corner_error<1e-6, max_corner_error
    max_overlap=0.
    e=pd.read_csv(HERE/'neighbors.csv')
    for row in e.itertuples():
        x,y,w,h=json.loads(row.read_window)
        window=np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],np.float32)
        target=np.asarray(json.loads(row.target_points),np.float32)
        area,_=cv2.intersectConvexConvex(target,window)
        max_overlap=max(max_overlap,float(area))
    assert max_overlap<1e-3, max_overlap
    p=pd.read_csv(HERE/'predictions.csv')
    zero=p.neighbors.eq(0)
    difference=float(np.abs(p.loc[zero,['s0','s1','s2']].to_numpy()-p.loc[zero,['b0','b1','b2']].to_numpy()).max())
    assert difference==0, difference
    s=pd.read_csv(HERE/'splits.csv').merge(t[['instance_id','mmsi','pass_group']],on='instance_id')
    for fold,part in s.groupby('fold'):
        for a,b in [('train','test'),('validation','test'),('train','validation')]:
            aa,bb=part[part.role.eq(a)],part[part.role.eq(b)]
            assert not(set(aa.mmsi)&set(bb.mmsi))
            assert not(set(aa.pass_group)&set(bb.pass_group))
    result=dict(targets=len(t),edges=len(e),max_original_corner_error_m=max_corner_error,
                max_target_obb_neighbor_window_intersection_px2=max_overlap,
                no_context_score_difference=difference,folds=s.fold.nunique())
    (HERE/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result)


if __name__=='__main__':
    main()
