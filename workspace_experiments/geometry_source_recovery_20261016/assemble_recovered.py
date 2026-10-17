"""Integrate recovered sources/annotations, retaining every previously valid field."""
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'geometry_targeted_fix_20261016'))
import recover_geometry as prior
from recover_source_bridge import name


@lru_cache(None)
def to_wgs84(epsg):
    return Transformer.from_crs(epsg,4326,always_xy=True)


def coordinate(geo,x,y):
    gt=geo['transform']
    projected=(gt[0]+x*gt[1]+y*gt[2],gt[3]+x*gt[4]+y*gt[5])
    return to_wgs84(int(geo['epsg'])).transform(*projected)


def fill_crop_centers(records,before,needed):
    requested=[int(i) for i in needed if before.iloc[i].ship_recovery_status=='not_computed_missing_target_center']
    metadata=Path('E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/metadata')
    sources=defaultdict(list)
    wanted={name(records[i]['image_relpath_'+pol]) for i in requested for pol in ('vv','vh')}
    for server in ('server1','server2'):
        path=metadata/server/'manifest.csv'
        f=pd.read_csv(path,low_memory=False)
        for s in f.loc[f.context_output_png.map(name).isin(wanted)].to_dict('records'):
            sources[name(s['context_output_png'])].append((s,str(path)))
    evidence=[]
    for i in requested:
        r=records[i]
        for pol in ('vv','vh'):
            candidates=sources[name(r['image_relpath_'+pol])]
            if len(candidates)!=1:
                continue
            s,path=candidates[0]
            assert s['product']==r['product'] and str(s['polarization']).lower()==pol
            assert name(s['source_tiff'])==name(r['source_tiff_'+pol])
            np.testing.assert_allclose([r['target_center_x_'+pol],r['target_center_y_'+pol]],
                                       [s['target_center_x'],s['target_center_y']],atol=1e-6,rtol=0)
            with Image.open(r['image_relpath_'+pol]) as image:
                assert image.size==(s['crop_right']-s['crop_left'],s['crop_bottom']-s['crop_top'])
            for axis,side in [('x','left'),('y','top')]:
                r['target_chip_center_'+axis+'_'+pol]=s['target_center_'+axis]-s['crop_'+side]
                r['crop_'+side+'_'+pol]=s['crop_'+side]
            evidence.append(dict(global_row=i,instance_id=r['instance_id'],polarization=pol,
                                 center_x=r['target_chip_center_x_'+pol],center_y=r['target_chip_center_y_'+pol],source=path))
    pd.DataFrame(evidence).to_csv(HERE/'crop_center_recovery.csv',index=False)


def load_grids():
    paths={}
    for path in (prior.old.GEOMETRY/'radar_grid').glob('*.json'):
        paths[path.stem]=path
    f=HERE/'product_annotations/manifest.csv'
    if f.exists():
        table=pd.read_csv(f)
        for r in table.itertuples():
            if isinstance(r.local_grid_path,str) and Path(r.local_grid_path).is_file():
                paths[r.product]=Path(r.local_grid_path)
    grids={}
    for product,path in paths.items():
        grid=json.loads(path.read_text(encoding='utf-8'))
        if grid.get('status')=='ok' and grid.get('product')==product:
            grids[product]=(grid,str(path))
    return grids


def main():
    before=pd.read_csv(ROOT/'geometry_targeted_fix_20261016/recovered_geometry.csv',low_memory=False)
    manifest=pd.read_csv(HERE/'source_manifest_recovered.csv',low_memory=False)
    np.testing.assert_array_equal(before.instance_id,manifest.instance_id)
    needed=np.flatnonzero(before.ship_recovery_status.notna())
    georefs=json.loads((HERE/'source_georefs.json').read_text(encoding='utf-8'))
    inputs=manifest.to_dict('records')
    fill_crop_centers(inputs,before,needed)
    pd.DataFrame(inputs).to_csv(HERE/'geometry_input_manifest.csv',index=False)
    records=before.to_dict('records')
    incidence=pd.read_csv(prior.old.INCIDENCE).set_index('instance_id')
    grids=load_grids()
    groups=defaultdict(list)
    changed=[]
    for number,index in enumerate(needed):
        row=inputs[index]; record=records[index]
        geo=georefs.get(row['source_tiff_vv'])
        coord=None; provenance='missing_exact_center'
        if geo and np.isfinite([row['target_center_x_vv'],row['target_center_y_vv']]).all():
            coord=coordinate(geo,row['target_center_x_vv'],row['target_center_y_vv'])
            provenance=geo['bridge_method']
        elif row['instance_id'] in incidence.index:
            inc=incidence.loc[row['instance_id']]
            if inc.coordinate_kind in prior.KINDS and np.isfinite([inc.longitude,inc.latitude]).all():
                coord=(float(inc.longitude),float(inc.latitude)); provenance='preserved_exact_incidence_coordinate'
        if coord:
            record.update(recovery_longitude=coord[0],recovery_latitude=coord[1],recovery_coordinate_method=provenance)
        if not record['ship_valid'] and str(record['ship_recovery_status']).startswith('not_computed'):
            missing=[pol for pol in ('vv','vh') if row['source_tiff_'+pol] not in georefs]
            if missing:
                record['ship_recovery_status']='missing_'+'_'.join(missing)+'_source_georef'
            else:
                updates,status=prior.measure_missing_pair(row,georefs)
                record.update(updates); record['ship_recovery_status']=status
                if updates:
                    record['ship_provenance']='recovered_exact_crop_manifest_and_headers_same_estimator'
        if not record['radar_valid']:
            status='missing_vv_source_georef' if geo is None else 'missing_exact_coordinate'
            if geo is not None and coord is not None:
                status='missing_exact_product_grid'
                if row['product'] in grids:
                    grid,path=grids[row['product']]
                    with Image.open(row['image_relpath_vv']) as image:
                        updates,status=prior.radar_recovery(grid,*coord,geo,*image.size)
                    record.update(updates)
                    if updates:
                        record['radar_provenance']=path
                        record['radar_status']='recovered_exact_source_header_and_product_grid'
            record['radar_recovery_status']=status
        if not record['incidence_valid']:
            record['incidence_recovery_status']='missing_exact_coordinate' if coord is None else 'missing_exact_product_grid'
            if coord is not None:
                groups[row['product']].append((int(index),coord))
        if number%1000==0:
            print('GEOMETRY',number,'/',len(needed),flush=True)
    for product,items in groups.items():
        path=prior.TRACK/'s1_grid'/(product+'.json')
        if product in grids:
            grid,source=grids[product]
            points=[[p['latitude'],p['longitude'],p['incidenceAngle']] for p in grid['points']]
        elif path.is_file():
            points=json.loads(path.read_text(encoding='utf-8')); source=str(path)
        else:
            continue
        values=prior.interpolate(points,[[c[1],c[0]] for _,c in items])
        for (i,_),value in zip(items,values):
            r=records[i]
            if value is None:
                r['incidence_recovery_status']='outside_exact_grid_hull'
                continue
            r.update(incidence_rad=np.deg2rad(value['incidence_deg']),incidence_valid=1,
                     incidence_status='recovered_exact_source_center_grid_interpolation',
                     incidence_recovery_status='recovered',incidence_provenance=source)
    after=pd.DataFrame(records)
    prior.verify_preservation(before,after,needed)
    np.testing.assert_array_equal(before.global_row,after.global_row)
    after.to_csv(HERE/'recovered_geometry.csv',index=False)
    role,task=prior.coverage(before,after)
    role.to_csv(HERE/'role_coverage.csv',index=False); task.to_csv(HERE/'task_coverage.csv',index=False)
    for field in ('ship','radar','incidence'):
        indices=needed[(after.iloc[needed][field+'_valid']>before.iloc[needed][field+'_valid']).to_numpy()]
        for i in indices:
            changed.append(dict(global_row=int(i),instance_id=inputs[i]['instance_id'],port=inputs[i]['port'],product=inputs[i]['product'],
                                field=field,provenance=after.loc[i,field+'_provenance']))
    pd.DataFrame(changed).to_csv(HERE/'newly_valid_fields.csv',index=False)
    summary=dict(cohort_rows=len(needed),baseline='geometry_targeted_fix_20261016/recovered_geometry.csv',
                 before={k:int(before.iloc[needed][k+'_valid'].sum()) for k in ('ship','radar','incidence')},
                 after={k:int(after.iloc[needed][k+'_valid'].sum()) for k in ('ship','radar','incidence')},
                 new={k:int((after.iloc[needed][k+'_valid']-before.iloc[needed][k+'_valid']).sum()) for k in ('ship','radar','incidence')},
                 statuses={k:after.iloc[needed][k+'_recovery_status'].value_counts().to_dict() for k in ('ship','radar','incidence')},
                 old_valid_values_preserved=True,outside_cohort_unchanged=True,old_results_untouched=True,
                 ship_slot_cache_requires_refresh=bool((after.ship_valid>before.ship_valid).any()),training_started=False)
    (HERE/'recovery_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
