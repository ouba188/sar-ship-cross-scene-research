"""Describe size-matched and same-vessel image comparisons without causal claims."""
import json
import numpy as np
import pandas as pd
from analyze import ROOT, WORK, STORE, csv, write
from images import PAIRS, native, panel


def main():
    rows=json.loads((STORE/'manifest.json').read_text(encoding='utf-8'))
    df=pd.read_csv(ROOT/'source_observations_local.csv',dtype={'mmsi':str})
    df=df[df.valid & df.plausible].set_index('index',drop=False).rename_axis('row_index')
    pairs=pd.read_csv(ROOT/'ais_matched_pairs_local.csv')
    metrics=[]
    for r in pairs.itertuples():
        for pol in ('vv','vh'):
            a,b=native(rows[r.index_a],pol),native(rows[r.index_b],pol)
            if a is None or b is None:continue
            da,db=df.loc[r.index_a],df.loc[r.index_b]
            rec=dict(class_name=r.class_name,pol=pol,pair_rank=r.pair_rank,
                     index_a=r.index_a,index_b=r.index_b,port_a=r.port_a,port_b=r.port_b,
                     same_quarter=da.quarter==db.quarter,
                     verified_same_10m=bool(da[f'scale_valid_{pol}'] and db[f'scale_valid_{pol}'] and
                         all(np.isclose([da[f'pixel_x_m_{pol}'],da[f'pixel_y_m_{pol}'],
                                         db[f'pixel_x_m_{pol}'],db[f'pixel_y_m_{pol}']],10))))
            for key in ('obb_length_px','contrast','outside_mean','saturation'):
                rec[key+'_delta']=b['metrics'][key]-a['metrics'][key]
            # Axis sign ambiguity is not interpreted as bow/stern evidence.
            aa=np.array(a['profile']);bb=np.array(b['profile'])
            aa=(aa+aa[::-1])/2;bb=(bb+bb[::-1])/2
            rec['symmetric_profile_l1']=float(np.nanmean(abs(aa-bb)))
            metrics.append(rec)
    m=pd.DataFrame(metrics)
    csv('size_matched_image_pairs_local.csv',m)
    result=[]
    for scope,part in [('all',m),('same_quarter',m[m.same_quarter]),
                       ('verified_same_10m',m[m.verified_same_10m])]:
        for keys,s in part.groupby(['class_name','pol','port_a','port_b']):
            result.append(dict(scope=scope,class_name=keys[0],pol=keys[1],port_a=keys[2],port_b=keys[3],
                pairs=len(s),median_obb_length_delta=s.obb_length_px_delta.median(),
                median_contrast_delta=s.contrast_delta.median(),
                median_abs_contrast_delta=s.contrast_delta.abs().median(),
                median_background_delta=s.outside_mean_delta.median(),
                median_symmetric_profile_l1=s.symmetric_profile_l1.median()))
    csv('size_matched_image_summary.csv',pd.DataFrame(result))
    # Same-MMSI cross-port references isolate population membership from appearance,
    # subject to the existing identity/association caveat; no new identity claims.
    records=[]; examples=[]
    for name,a,b in PAIRS:
        sub=df[(df.class_name==name)&df.port.isin([a,b])].sort_values(['product','index'])
        common=set(sub[sub.port==a].mmsi)&set(sub[sub.port==b].mmsi)
        choices=[]
        for mmsi in sorted(common):
            items=[sub[(sub.port==port)&(sub.mmsi==mmsi)].iloc[0] for port in (a,b)]
            va,vb=[native(rows[int(r['index'])],'vv') for r in items]
            if va is None or vb is None:continue
            difference=abs(va['metrics']['contrast']-vb['metrics']['contrast'])
            choices.append((difference,items))
        if choices:
            choices.sort(key=lambda x:x[0])
            difference,items=choices[len(choices)//2]
            examples.extend([x.to_dict() for x in items])
            records.append(dict(class_name=name,eligible_pairs=len(choices),median_selected_delta=difference,
                indices=[int(x['index']) for x in items],mmsi=items[0].mmsi))
    if examples:
        cases=panel(examples,rows,'same_vessel_cross_port',
            'Same AIS identity, cross-port: median VV contrast-change pair within each class')
        write('same_vessel_controls_local.json',dict(selection=records,cases=cases))
    groups=json.loads((STORE/'groups.json').read_text(encoding='utf-8'))
    refs=df[df['index'].isin(groups['references']) & (df.class_name=='chemical_tanker')]
    pool=df[(df.class_name=='chemical_tanker') & (df.port=='Houston') & df.current_fit]
    query=pool.loc[(pool.ais_length_m-183).abs().sort_values(kind='stable').index[0]]
    cases=panel([query.to_dict()]+refs.to_dict('records'),rows,'chemical_actual_references',
        'Chemical/product tanker: a source-fit query and its TWO fixed same-class references')
    write('reference_cases_local.json',cases)
    print(pd.DataFrame(result).query("scope=='all'").to_string(index=False))
    print('Same vessel eligible pairs:',[(x['class_name'],x['eligible_pairs']) for x in records])


if __name__=='__main__':main()
