"""Add explicit final interface flags without repeating source recovery."""
import json
from pathlib import Path
import pandas as pd

OUT=Path(__file__).resolve().parent


def main():
    f=pd.read_csv(OUT/'ais_metadata_sidecar.csv',dtype={'mmsi':str},keep_default_na=False,low_memory=False)
    f['mmsi_valid']=f.mmsi.str.fullmatch(r'[0-9]{9}')
    f.loc[~f.mmsi_valid,'identity_link_quality']='missing_mmsi'
    f['transform_json_order']='GDAL_x0_xx_xy_y0_yx_yy'
    f['ais_position_match_verified']=f.ais_record_match_quality.str.endswith('_timestamp_coordinate')
    for pol in ('vv','vh'):
        f['scale_valid_'+pol]=f['pixel_x_m_'+pol].ne('')
        f['chip_position_valid_'+pol]=f['chip_transform_json_'+pol].ne('')
    # Finish the new file before replacing the current interface read by the parent task.
    temp=OUT/'ais_metadata_sidecar.final.csv'
    f.to_csv(temp,index=False,encoding='utf-8-sig')
    temp.replace(OUT/'ais_metadata_sidecar.csv')
    conflict=f.vessel_static_conflict.astype(str).str.lower().eq('true')|f.ais_size_scope.eq('conflicting_exact_evidence')
    f[conflict].to_csv(OUT/'dimension_conflicts.csv',index=False,encoding='utf-8-sig')
    s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    s.update(vessels=int(f.loc[f.mmsi_valid,'mmsi'].nunique()),missing_mmsi_rows=int((~f.mmsi_valid).sum()),
             nonempty_mmsi_distinct=int(f.loc[f.mmsi.ne(''),'mmsi'].nunique()),
             ais_position_match_verified_rows=int(f.ais_position_match_verified.sum()),
             same_product_imo_conflict_rows=int(f.same_product_imo_conflict.astype(str).str.lower().eq('true').sum()),
             transform_json_order='GDAL [x0,xx,xy,y0,yx,yy]',
             chip_position_valid_vv=int(f.chip_position_valid_vv.sum()),
             chip_position_valid_vh=int(f.chip_position_valid_vh.sum()))
    reads=pd.read_csv(OUT/'read_ais_sources.csv')
    s.update(raw_products_read=int(reads.status.eq('read').sum()),raw_bytes_read=int(reads.bytes.sum()))
    (OUT/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    print(json.dumps(s,indent=2))


if __name__=='__main__':
    main()
