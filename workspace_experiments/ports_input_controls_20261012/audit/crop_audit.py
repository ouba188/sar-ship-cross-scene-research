"""Native geometry audit and small label-free crop smoke test; no models."""
from pathlib import Path
import json
import re
import sys
from lxml import etree as ET

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transforms_fixed import crop_box

OUT = Path(__file__).resolve().parent
WORK = OUT.parent.parent
ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
SOURCE = WORK / 'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv'
RETURN = Path('E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/metadata')
FRACTION = 0.75
PORTS = ('Los Angeles', 'Port Klang', 'Santos')
XML_ISSUES = []


def read(path):
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def save(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def rect(w, h):
    return crop_box((w, h))


def inside(x, y, bounds):
    l, t, r, b = bounds
    return l <= x < r and t <= y < b


def descriptors(dataset):
    path = (WORK / 'ports_public_data_study_20261012/image_diagnostics/oss1.csv'
            if dataset == 'oss1' else ROOT / f'c4_condition_diagnosis_20261009/descriptors/{dataset}.csv')
    return read(path)


def source_rows():
    frame = read(SOURCE)
    desc = descriptors('self24ports')
    assert np.array_equal(frame.instance_id, desc.sample_id)
    dims = desc.to_dict('records')
    lookup = {}
    for path in sorted(RETURN.glob('server*/manifest.csv')):
        for row in read(path).to_dict('records'):
            key = Path(row['context_output_png']).name
            if key in lookup:
                assert lookup[key] == row
            lookup[key] = row
    rows = []
    for i, row in enumerate(frame.to_dict('records')):
        for pol in ('vv', 'vh'):
            missing = row[f'target_chip_center_x_{pol}'] == ''
            origin = 'existing_manifest'
            if missing:
                recovered = lookup[Path(row[f'image_relpath_{pol}']).name]
                for key in ('product', 'mmsi', 'port'):
                    assert str(recovered[key]) == str(row[key])
                assert abs(float(recovered['target_center_x'])-float(row[f'target_center_x_{pol}'])) < 1e-4
                assert abs(float(recovered['target_center_y'])-float(row[f'target_center_y_{pol}'])) < 1e-4
                cx = float(recovered['target_center_x'])-float(recovered['crop_left'])
                cy = float(recovered['target_center_y'])-float(recovered['crop_top'])
                bw, bh = float(recovered['width']), float(recovered['height'])
                side = float(recovered['crop_side'])
                origin = 'recovered_exact_png_basename'
            else:
                cx, cy = (float(row[f'target_chip_center_{a}_{pol}']) for a in ('x', 'y'))
                bw, bh = float(row[f'width_{pol}']), float(row[f'height_{pol}'])
                side = float(row[f'crop_side_{pol}'])
                assert abs(cx-(float(row[f'target_center_x_{pol}'])-float(row[f'crop_left_{pol}']))) < 1e-4
                assert abs(cy-(float(row[f'target_center_y_{pol}'])-float(row[f'crop_top_{pol}']))) < 1e-4
            w, h = (int(dims[i][f'native_{a}_px_{pol}']) for a in ('width', 'height'))
            bounds = rect(w, h)
            l, t, r, b = bounds
            rows.append(dict(dataset='self24ports', row=i, sample_id=row['instance_id'], pol=pol,
                port=row['port'],class_id=row['class_id'],class_name=row['class_name'],
                primary_scope=row['port'] in PORTS,
                path=row[f'image_relpath_{pol}'], origin=origin, native_w=w, native_h=h,
                crop_side=side, center_x=cx, center_y=cy,
                center_offset_px=float(np.hypot(cx-(w-1)/2, cy-(h-1)/2)),
                center_inside_original=inside(cx, cy, (0,0,w,h)),
                center_inside_crop=inside(cx,cy,bounds),
                box_w=bw, box_h=bh, box_inside_original=cx-bw/2>=0 and cy-bh/2>=0 and cx+bw/2<=w and cy+bh/2<=h,
                box_inside_crop=cx-bw/2>=l and cy-bh/2>=t and cx+bw/2<=r and cy+bh/2<=b,
                box_long_side_fraction=max(bw/w,bh/h),
                size_matches_crop_side=w==side and h==side))
    return rows


def public_rows(dataset):
    frame = read(ROOT / f'opensarship{dataset[-1]}_transfer_20261007/target_manifest.csv')
    desc = descriptors(dataset)
    assert np.array_equal(frame.sample_id, desc.sample_id)
    dims = desc.to_dict('records')
    xml_cache = {}
    rows = []
    for i, row in enumerate(frame.to_dict('records')):
        source = Path(row['chip_path_vv'] if dataset=='oss1' else row['source_path_vv'])
        xml = source.parent.parent / 'Ship.xml'
        if xml not in xml_cache:
            by_center = {}
            parser = ET.XMLParser(recover=dataset=='oss2', encoding='utf-8')
            payload = xml.read_bytes()
            if dataset == 'oss2':
                decoded = payload.decode('utf-8', errors='replace')
                if '\ufffd' in decoded:
                    XML_ISSUES.append(dict(dataset=dataset,path=str(xml),line=-1,
                        message=f'Invalid UTF-8 replaced in-memory only: {decoded.count(chr(65533))} bytes/characters'))
                payload = decoded.encode('utf-8')
            tree = ET.fromstring(payload, parser)
            for error in parser.error_log:
                XML_ISSUES.append(dict(dataset=dataset, path=str(xml), line=error.line, message=error.message))
            for ship in tree.findall('ship'):
                s = ship.find('SARShipInformation')
                vals = {e.tag: e.text for e in s}
                vals['xml_mmsi'] = ship.findtext('AISShipInformation/MMSI')
                assert not any('\ufffd' in str(vals.get(k,'')) for k in
                    ('Center_x','Center_y','Head_x','Head_y','Tail_x','Tail_y',
                     'UpperLeft_x','UpperLeft_y','LowerRight_x','LowerRight_y','xml_mmsi'))
                key = tuple(int(vals[f'Center_{a}']) for a in ('x','y'))
                assert key not in by_center
                by_center[key] = vals
            xml_cache[xml] = by_center
        match = re.search(r'_x(-?\d+)_y(-?\d+)_vv\.tif$',source.name)
        key = (int(match[1]),int(match[2]))
        v = xml_cache[xml][key]
        assert str(v['xml_mmsi']).strip() == str(row['mmsi_or_ship_id']).strip()
        corners = [[float(v[f'{name}_{a}']) for a in ('x','y')] for name in ('UpperLeft','LowerRight')]
        lo, hi = np.min(corners,axis=0), np.max(corners,axis=0)
        cx, cy = np.asarray(key)-lo
        hx, hy = np.array([float(v[f'Head_{a}']) for a in ('x','y')])-lo
        tx, ty = np.array([float(v[f'Tail_{a}']) for a in ('x','y')])-lo
        for pol in ('vv','vh'):
            w,h = (int(dims[i][f'native_{a}_px_{pol}']) for a in ('width','height'))
            bounds = rect(w,h)
            rows.append(dict(dataset=dataset,row=i,sample_id=row['sample_id'],pol=pol,
                port='',class_id=row['class_id'],class_name=row['class_name'],primary_scope=True,
                path=row[f'chip_path_{pol}'],source_path=str(source.with_name(source.name[:-6]+pol+'.tif')),
                xml_path=str(xml),native_w=w,native_h=h,origin='xml_min_corner_translation',
                center_x=cx,center_y=cy,center_offset_px=float(np.hypot(cx-(w-1)/2,cy-(h-1)/2)),
                center_midpoint_error_px=float(np.linalg.norm(np.asarray(key)-(lo+hi)/2)),
                center_inside_original=inside(cx,cy,(0,0,w,h)),center_inside_crop=inside(cx,cy,bounds),
                xml_w=hi[0]-lo[0]+1,xml_h=hi[1]-lo[1]+1,
                xml_size_matches_header=w==hi[0]-lo[0]+1 and h==hi[1]-lo[1]+1,
                head_x=hx,head_y=hy,tail_x=tx,tail_y=ty,
                endpoints_inside_original=inside(hx,hy,(0,0,w,h)) and inside(tx,ty,(0,0,w,h)),
                endpoints_inside_crop=inside(hx,hy,bounds) and inside(tx,ty,bounds),
                visible_length_px=float(np.hypot(hx-tx,hy-ty)),
                endpoints_long_side_fraction=max(abs(hx-tx)/w,abs(hy-ty)/h)))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    all_rows = source_rows()
    print('source geometry',len(all_rows),flush=True)
    for name in ('oss1','oss2'):
        data = public_rows(name)
        all_rows.extend(data)
        print(name,'geometry',len(data),flush=True)
    data = pd.DataFrame(all_rows)
    save('geometry.csv',data)
    save('xml_recovery.csv',XML_ISSUES)
    summaries=[]
    for (dataset,pol),g in data.groupby(['dataset','pol']):
        rec=dict(dataset=dataset,pol=pol,rows=len(g),recovered=int(g.origin.eq('recovered_exact_png_basename').sum()),
            center_offset_median=g.center_offset_px.median(),center_offset_p99=g.center_offset_px.quantile(.99),
            center_offset_max=g.center_offset_px.max())
        for col in ('center_inside_original','center_inside_crop','box_inside_original','box_inside_crop',
                    'size_matches_crop_side','xml_size_matches_header','endpoints_inside_original','endpoints_inside_crop'):
            valid=g[col].dropna()
            rec[col+'_known']=len(valid)
            rec[col+'_yes']=int(valid.astype(bool).sum())
        summaries.append(rec)
    save('geometry_summary.csv',summaries)
    coverage=[]
    for (dataset,port,cls),g in data.groupby(['dataset','port','class_name'],dropna=False):
        if dataset=='self24ports' and port not in PORTS:
            continue
        by_sample=g.groupby('sample_id')
        if dataset=='self24ports':
            original=g.origin.eq('existing_manifest').groupby(g.sample_id).all()
            valid=g.center_inside_original.astype(bool).groupby(g.sample_id).all()
            preserved=g.box_inside_crop.astype(bool).groupby(g.sample_id).all()
            lostcenter=(~g.center_inside_crop.astype(bool)).groupby(g.sample_id).any()
            kind='detector_box'
        else:
            original=pd.Series(False,index=by_sample.size().index)
            valid=(g.xml_size_matches_header.astype(bool)&g.endpoints_inside_original.astype(bool)).groupby(g.sample_id).all()
            preserved=g.endpoints_inside_crop.astype(bool).groupby(g.sample_id).all()
            lostcenter=pd.Series(False,index=by_sample.size().index)
            kind='head_tail_endpoints_only'
        coverage.append(dict(dataset=dataset,port=port,class_name=cls,n=len(valid),
            source_original_geometry=int(original.sum()),metadata_joined=len(valid),
            usable_truncation_proxy=int(valid.sum()),usable_rate=float(valid.mean()),
            indeterminate=int((~valid).sum()),new_proxy_crop_crossing=int((valid&~preserved).sum()),
            target_center_crop_crossing=int(lostcenter.sum()),proxy_type=kind))
    save('class_crop_coverage.csv',coverage)
    anomalies=data[data.primary_scope & (
        (data.dataset.eq('self24ports') & data.box_inside_crop.eq(False)) |
        (~data.dataset.eq('self24ports') & (data.xml_size_matches_header.eq(False) |
          data.endpoints_inside_original.eq(False) | data.endpoints_inside_crop.eq(False))))]
    save('primary_geometry_flags.csv',anomalies)
    tests=[]
    for dataset,g in data[data.pol.eq('vv')].groupby('dataset',sort=False):
        # Fixed row-spaced samples plus worst center offset, with no class/outcome access.
        flagged=set(anomalies.loc[anomalies.dataset.eq(dataset),'sample_id'])
        indices=sorted(set(np.linspace(0,len(g)-1,8,dtype=int).tolist()+
            [int(np.argmax(g.center_offset_px.to_numpy()))]+np.flatnonzero(g.sample_id.isin(flagged)).tolist()))
        for pos in indices:
            vv=g.iloc[pos]
            pair=data[(data.dataset==dataset)&(data.row==vv.row)]
            for row in pair.itertuples():
                with Image.open(row.path) as image:
                    header_ok=image.size==(row.native_w,row.native_h)
                    assert header_ok
                    pixels=np.asarray(image)
                    bounds=rect(*image.size)
                    cropped=np.asarray(image.crop(bounds))
                    l,t,r,b=bounds
                    np.testing.assert_array_equal(cropped,pixels[t:b,l:r])
                    image.crop(bounds).resize((128,128),Image.Resampling.BILINEAR)
                    # Pixel proxy is intentionally not called a vessel truncation detector.
                    gray=np.asarray(image.convert('L'),dtype=float)
                    bright=gray>np.quantile(gray,.95)
                    mask=np.zeros(gray.shape,bool); mask[t:b,l:r]=True
                    n=int(bright.sum())
                    retained=float((bright&mask).sum()/n) if n else np.nan
                    exact_native=np.nan
                    if dataset=='oss2':
                        with Image.open(row.source_path) as original:
                            exact_native=np.array_equal(pixels,np.asarray(original))
                            assert exact_native
                    tests.append(dict(dataset=dataset,row=row.row,sample_id=row.sample_id,pol=row.pol,
                        path=row.path,width=row.native_w,height=row.native_h,
                        center_x=row.center_x,center_y=row.center_y,
                        left=l,top=t,right=r,bottom=b,retained_area=(r-l)*(b-t)/(row.native_w*row.native_h),
                        bright_q95_retained_fraction=retained,bright_count=n,
                        header_matches_cache=header_ok,crop_matches_array_slice=True,oss2_png_equals_native_tif=exact_native))
    save('sample_crop_tests.csv',tests)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(status='complete',geometry_rows=len(data),sample_views=len(tests),
        crop_fraction=FRACTION,labels_used_for_transform=False,models_run=False,
        original_images_modified=False,scope='native geometry and limited input-array test only'),indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    main()
