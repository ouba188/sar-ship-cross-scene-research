"""Additional local GRDH metadata, excluding known validation/evaluation vessels."""
from lxml import etree
import numpy as np
import pandas as pd
import json
from analyze import ROOT,OLD,META,MANIFEST,csv,write,vessels,pair_values

def main():
    observations=pd.read_csv(OLD/'observations_grd.csv',dtype={'mmsi':str})
    selected=observations[observations.class_name.isin(['bulk_carrier','fishing_vessel','general_cargo','product_chemical_tanker','container_ship','crude_oil_tanker','tug_towing','offshore_supply'])].copy()
    rows=json.loads(MANIFEST.read_text(encoding='utf-8'))
    held={r['mmsi'] for r in rows if r['role'] in ['evaluation','fit_validation']}
    selected=selected[~selected.mmsi.isin(held)].copy()
    records=[]
    for path,q in selected.groupby('ship_xml'):
        raw=open(path,'rb').read().decode('utf-8',errors='replace')
        xml=etree.fromstring(raw.encode(),etree.XMLParser(recover=True))
        ships=xml.findall('.//ship')
        for r in q.itertuples():
            node=ships[r.xml_index]
            def value(field):return (node.findtext(field) or '').strip()
            assert value('AISShipInformation/MMSI')==r.mmsi
            assert int(value('SARShipInformation/Center_x'))==r.center_x and int(value('SARShipInformation/Center_y'))==r.center_y
            def number(field):
                try:return float(value(field))
                except ValueError:return np.nan
            records.append(dict(sample_id=f'{r.scene_id}:{r.xml_index}',class_name=r.class_name,port_proxy=r.port_proxy,mmsi=r.mmsi,product=r.scene_id,ais_length_m=number('AISShipInformation/Length'),ais_width_m=number('AISShipInformation/Width'),conservative_assignment=r.conservative_assignment,polarizations=r.available_polarizations))
    d=pd.DataFrame(records)
    conflicts=d.groupby('mmsi').agg(classes=('class_name','nunique'),lengths=('ais_length_m','nunique'),widths=('ais_width_m','nunique'))
    bad=set(conflicts[(conflicts.classes>1)|(conflicts.lengths>1)|(conflicts.widths>1)].index)
    d['valid']=d.ais_length_m.gt(0)&d.ais_width_m.gt(0)&~d.mmsi.isin(bad)&d.mmsi.str.fullmatch(r'\d{9}')&(d.port_proxy!='outside_150km')
    csv('extended_observations_local.csv',d)
    out=[];pair=[]
    for scope,q in [('extended_GRDH',d[d.valid]),('extended_conservative',d[d.valid & d.conservative_assignment]),('extended_dual_VV_VH',d[d.valid & (d.polarizations=='vh|vv')])]:
        v=vessels(q)
        for (cls,p),g in v.groupby(['class_name','port_proxy']):
            out.append(dict(scope=scope,class_name=cls,port_proxy=p,ships=len(g),length_median=g.length.median(),q10=g.length.quantile(.1),q90=g.length.quantile(.9)))
        for r in pd.read_csv(ROOT/'selected_pairs.csv').itertuples():
            a,b,common=pair_values(v[v.class_name==r.class_name],r.port_low,r.port_high)
            pair.append(dict(scope=scope,class_name=r.class_name,port_low=r.port_low,port_high=r.port_high,n_low=len(a),n_high=len(b),shared_removed=common,median_low=a.length.median(),median_high=b.length.median(),gap_m=b.length.median()-a.length.median()))
    csv('extended_port_class_sizes.csv',pd.DataFrame(out));csv('extended_selected_pair_sensitivity.csv',pd.DataFrame(pair))
    write('extended_coverage.json',dict(all_local_grdh_observations=len(observations),research8_before_exclusion=int(observations.class_name.isin(d.class_name.unique()).sum()),excluded_known_heldout_identities=len(held),remaining_observations=len(d),conflicted_ships=len(bad),valid_observations=int(d.valid.sum()),valid_ships=d.loc[d.valid,'mmsi'].nunique(),classes=d.loc[d.valid,'class_name'].nunique(),notes=['Descriptive expanded metadata, not additional actual training or an independent confirmation set.','Known validation/evaluation vessels excluded before dimensions were extracted.','Reuses eight-class mapping and geographic proxies from prior GRDH census.','No image statistics or model performance inferred for these extra rows.']))
    print(pd.DataFrame(pair).to_string(index=False));print((ROOT/'extended_coverage.json').read_text())

if __name__=='__main__':main()
