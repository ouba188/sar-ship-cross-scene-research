"""Recover exact chip-source links and cached GeoTIFF headers, without pixels."""
import json
import sys
from collections import defaultdict
from pathlib import Path,PureWindowsPath

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ART=ROOT/'visual_reliable_baseline/artifacts'
sys.path.insert(0,str(ROOT/'ship_radar_geometry_design_20261016'))
sys.path.insert(0,str(ROOT/'visual_reliable_baseline'))
import geometry_metadata as original
from recover_remote_header_coordinates import transform as parse_header,tile_origin_in_scene


def name(value):
    return PureWindowsPath(str(value)).name


def canonical(value):
    return str(PureWindowsPath(str(value))).lower()


def equivalent(a,b):
    return (a['epsg']==b['epsg'] and a['width']==b['width'] and a['height']==b['height']
            and np.allclose(a['transform'],b['transform'],rtol=0,atol=.01))


def load_headers():
    headers=[]
    for key,h in original.load_georefs().items():
        headers.append(dict(h,path=key,header_source='original_server_georef_cache'))
    f=ART/'real_k_asset_audit_20260917/detection_recovery/georef_headers_private.json'
    for key,h in json.loads(f.read_text(encoding='utf-8')).items():
        if h.get('epsg') and h.get('transform'):
            headers.append(dict(h,path=key,header_source=str(f)))
    f=ART/'remote_exact_coordinates_v1/headers_private.jsonl'
    for line in f.read_text(encoding='utf-8').splitlines():
        h=json.loads(line)
        if h.get('status')!='read':
            continue
        x,y,dx,dy,epsg=parse_header(h)
        headers.append(dict(path=h['path'],width=h['width'],height=h['height'],epsg=epsg,
                            transform=[x,dx,0.,y,0.,-dy],header_source=str(f)))
    return headers


def choose(headers,path,port):
    exact=[h for h in headers if canonical(h['path'])==canonical(path)]
    candidates=exact or [h for h in headers if name(h['path'])==name(path) and port.lower() in canonical(h['path'])]
    if not candidates:
        return None,'no_header'
    if any(not equivalent(candidates[0],h) for h in candidates[1:]):
        return None,'conflicting_headers'
    return dict(candidates[0]),'exact_path_header' if exact else 'unique_product_pol_tile_header'


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(original.MANIFEST,low_memory=False)
    before=pd.read_csv(ROOT/'geometry_targeted_fix_20261016/recovered_geometry.csv',low_memory=False)
    needed=np.flatnonzero(before.ship_recovery_status.notna())
    sources=pd.read_csv(ART/'real_k_asset_audit_20260917/fdrive_sources/original_sources_private.csv',low_memory=False)
    assert not sources.duplicated(['instance_id','polarization']).any()
    sources=sources.set_index(['instance_id','polarization'])
    headers=load_headers()
    by_name=defaultdict(list)
    for h in headers:
        by_name[name(h['path'])].append(h)
    repaired=manifest.to_dict('records')
    links=[]; georefs={}
    for number,index in enumerate(needed):
        row=manifest.iloc[index]
        if number%1000==0:
            print('BRIDGE',number,'/',len(needed),flush=True)
        for pol in ('vv','vh'):
            info=dict(global_row=int(index),instance_id=row.instance_id,port=row.port,product=row['product'],polarization=pol)
            source=row['source_tiff_'+pol]
            center=np.array([row['target_center_x_'+pol],row['target_center_y_'+pol]],float)
            key=(row.instance_id,pol)
            if pd.isna(source) and key in sources.index:
                s=sources.loc[key]
                assert s['product']==row['product']
                assert name(s.context_output_png)==name(row['image_relpath_'+pol])
                np.testing.assert_allclose(center,[s.target_center_x,s.target_center_y],atol=1e-6,rtol=0)
                source=s.source_tiff
                repaired[index]['source_tiff_'+pol]=source
                for out,raw in [('source_json','source_json'),('source_object_index','source_object_index'),
                                ('obj_uid','obj_uid'),('crop_left','crop_left'),('crop_top','crop_top')]:
                    column=out+'_'+pol
                    if column not in repaired[index] or pd.isna(repaired[index][column]):
                        repaired[index][column]=s[raw]
                info['source_link']='exact_original_crop_manifest'
            elif pd.notna(source):
                info['source_link']='preserved_manifest_path'
            else:
                info.update(source_link='unresolved',status='missing_original_source_link')
                links.append(info); continue
            info['source_tiff']=source
            found,status=choose(by_name.get(name(source),[]),source,row.port)
            # Only read a real TIFF header; never decode its raster.
            if found is None and Path(source).is_file():
                import rasterio
                with rasterio.open(source) as ds:
                    if ds.crs and ds.crs.to_epsg():
                        found=dict(path=source,epsg=ds.crs.to_epsg(),width=ds.width,height=ds.height,
                                   transform=list(ds.transform.to_gdal()),header_source='live_local_rasterio_header')
                        status='live_exact_source_header'
            if found is None and 'int8_grid_4x4_o256' in str(source):
                scene_name=row['product']+'_'+pol.upper()+'_UTM_8bit.tif'
                scene,scene_status=choose(by_name.get(scene_name,[]),scene_name,row.port)
                if scene is not None:
                    x0,y0=tile_origin_in_scene(name(source),scene['width'],scene['height'])
                    # This is the original documented 4x4, 256-overlap tiling rule.
                    import re
                    m=re.search(r'__r(\d+)_c(\d+)\.tif$',name(source))
                    tr,tc=map(int,m.groups())
                    x1=min(scene['width'],int(scene['width']*tc/4)+128)
                    y1=min(scene['height'],int(scene['height']*tr/4)+128)
                    gt=scene['transform']
                    found=dict(scene,transform=[gt[0]+x0*gt[1]+y0*gt[2],gt[1],gt[2],gt[3]+x0*gt[4]+y0*gt[5],gt[4],gt[5]],
                               width=x1-x0,height=y1-y0)
                    status='exact_product_scene_derived_original_tile'
            if found is None:
                info['status']=status
                links.append(info); continue
            if not (np.isfinite(center).all() and 0<=center[0]<found['width'] and 0<=center[1]<found['height']):
                info['status']='center_outside_header'
                links.append(info); continue
            found.update(status='ok',remote_source_tiff=source,bridge_method=status)
            if source in georefs:
                assert equivalent(georefs[source],found)
            georefs[source]=found
            info.update(status='recovered_header',header_method=status,header_path=found['path'],
                        header_source=found['header_source'],epsg=found['epsg'])
            links.append(info)
    frame=pd.DataFrame(links)
    frame.to_csv(HERE/'source_links.csv',index=False)
    pd.DataFrame(repaired).to_csv(HERE/'source_manifest_recovered.csv',index=False)
    (HERE/'source_georefs.json').write_text(json.dumps(georefs,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=dict(cohort_rows=len(needed),headers_examined=len(headers),source_link_counts=frame.source_link.value_counts().to_dict(),
                 view_status_counts=frame.status.value_counts().to_dict(),unique_source_headers=len(georefs),
                 rows_with_both_headers=int(frame.groupby('global_row').status.apply(lambda x:(x=='recovered_header').sum()==2).sum()))
    (HERE/'source_bridge_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
