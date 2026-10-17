"""Check observation joins, retained conflicts, and output provenance."""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
import pandas as pd
import build_sidecar as b


def join_cases():
    product='S1A_IW_GRDH_1SDV_20250115T173330_20250115T173355_057458_07130D_51CB'
    row=dict(instance_id='test',mmsi='205396000',product=product,port='test',class_name='test',data_origin='test')
    view=dict(global_row=0,instance_id='test',pol='vv',mmsi=row['mmsi'],product=product,
              object_status='exact_uid_product_mmsi',object_length_m=200,object_width_m=30,
              object_attribute_source='raw_ais_review_only',object_local_json='source.json',
              obj_uid='target',json_object_index=2,ais_epoch=100.,stored_coordinates_json='[]')
    rec=dict(product=product,mmsi=row['mmsi'],epoch=100.,lon=1.,lat=2.,raw_motion=('0','0','0'),
             length_m=200.,width_m=30.,imo='',source_txt='raw.txt',source_line=2)
    def run(exact, views=None, ps=None, vs=None):
        views=list(views or [view])
        if not any(v['pol']=='vh' for v in views):
            views.append(dict(view,pol='vh',object_status='no_exact_uid_json',object_length_m=None,object_width_m=None,ais_epoch=None))
        return b.assemble([row],views,(exact,ps or defaultdict(dict),vs or defaultdict(dict),
                                              defaultdict(set),defaultdict(set)))[0]
    assert run({(product,row['mmsi'],100.):[rec]})['ais_size_scope']=='exact_observation_record'
    wrong=run({('OTHER_PRODUCT',row['mmsi'],100.):[rec]})
    assert wrong['ais_size_scope']=='exact_uid_object_metadata' and wrong['ais_record_match_quality']=='no_exact_timestamp_record'
    wrong=run({(product,row['mmsi'],101.):[rec]})
    assert wrong['ais_size_scope']=='exact_uid_object_metadata' and wrong['ais_record_match_quality']=='no_exact_timestamp_record'
    conflict=run({(product,row['mmsi'],100.):[rec,dict(rec,length_m=250.)]})
    assert not conflict['ais_size_valid'] and conflict['ais_size_scope']=='conflicting_exact_evidence'
    conflict=run({},[view,dict(view,pol='vh',ais_epoch=101.)])
    assert not conflict['ais_size_valid'] and conflict['object_view_time_conflict']
    chosen,status=b.resolve_raw([rec,dict(rec,lon=3.)],[])
    assert not chosen and status=='conflicting_exact_position_or_motion'
    chosen,status=b.resolve_raw([rec,dict(rec,lon=3.)],[(1.,2.)])
    assert chosen==[rec] and status.endswith('_coordinate')
    missing=dict(view,object_status='no_exact_uid_json',object_length_m=None,object_width_m=None,ais_epoch=None)
    ps={(product,row['mmsi']):{(200.,30.):rec,(250.,30.):dict(rec,length_m=250.)}}
    conflict=run({},[missing],ps)
    assert not conflict['ais_size_valid']
    return 8


def main():
    cases=join_cases()
    f=pd.read_csv(b.OUT/'ais_metadata_sidecar.csv',dtype={'mmsi':str},low_memory=False)
    m=pd.read_csv(b.MANIFEST,dtype={'mmsi':str},low_memory=False)
    assert len(f)==38091 and f.instance_id.is_unique
    assert f.global_row.tolist()==list(range(38091))
    for col in ('instance_id','mmsi','product','port'):
        assert f[col].tolist()==m[col].tolist(),col
    assert f.loc[f.ais_size_valid,['ais_length_m','ais_width_m']].notna().all().all()
    assert (f.loc[f.ais_size_valid,'ais_length_m']>=f.loc[f.ais_size_valid,'ais_width_m']).all()
    assert (f.loc[f.ais_size_valid,'ais_width_m']>0).all()
    assert f.loc[~f.ais_size_valid,['ais_length_m','ais_width_m']].isna().all().all()
    exact=f[f.ais_size_scope.eq('exact_observation_record')]
    samples=exact.groupby('port',group_keys=False).head(3)
    requests=defaultdict(list)
    for r in samples.to_dict('records'):
        requests[r['ais_size_source']].append(r)
    checked=0
    for path, group in requests.items():
        wanted={int(r['ais_source_line']):r for r in group}
        with Path(path).open(encoding='utf-8-sig',newline='') as stream:
            for line,fields in enumerate(csv.reader(stream),1):
                if line not in wanted:
                    continue
                r=wanted[line]
                assert r['product'] in path
                assert b.mmsi(fields[1])==r['mmsi'] and float(fields[0])==r['ais_epoch']
                assert float(fields[14])==r['ais_length_m'] and float(fields[15])==r['ais_width_m']
                assert float(fields[6])==r['ais_raw_lon'] and float(fields[7])==r['ais_raw_lat']
                checked+=1
    assert checked==len(samples)
    for pol in ('vv','vh'):
        valid=f['pixel_x_m_'+pol].notna()
        assert f.loc[valid,'ground_scale_status_'+pol].isin(['valid_native_png_original_crop_grid','valid_native_png_scale_position_missing']).all()
        assert valid.equals(f['scale_valid_'+pol])
        assert f['chip_transform_json_'+pol].notna().equals(f['chip_position_valid_'+pol])
        assert f.loc[~valid,'pixel_y_m_'+pol].isna().all()
        for r in f.loc[valid].to_dict('records'):
            g=json.loads(r['source_transform_json_'+pol])
            assert all(r[k+'_'+pol]==g[j] for k,j in [('affine_xx',1),('affine_xy',2),('affine_yx',4),('affine_yy',5)])
            assert math.isclose(r['pixel_x_m_'+pol],math.hypot(g[1],g[4]))
            assert math.isclose(r['pixel_y_m_'+pol],math.hypot(g[2],g[5]))
    assert f[['ais_A_m','ais_B_m','ais_C_m','ais_D_m']].isna().all().all()
    assert f.loc[~f.mmsi_valid,'ais_size_valid'].eq(False).all()
    assert f.mmsi.notna().equals(f.mmsi_valid)
    assert int(f.loc[f.mmsi_valid,'mmsi'].nunique())==18766
    result=dict(status='passed',observation_join_and_conflict_cases=cases,manifest_rows_checked=len(f),
                raw_source_rows_reopened=checked,raw_source_ports=samples.port.nunique(),
                vv_linear_transform_rows=int(f.pixel_x_m_vv.notna().sum()),
                vh_linear_transform_rows=int(f.pixel_x_m_vh.notna().sum()),
                dimension_components_left_unknown=True)
    (b.OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
