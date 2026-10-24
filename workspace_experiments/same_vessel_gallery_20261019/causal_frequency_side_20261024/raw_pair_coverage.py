"""Intersect this cohort with previously traced, still-readable raw products."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'round13_natural_amplitude'


def main():
    prior=ROOT.parent/'crude_input_controls_20261021/radiometry_trace/raw_window_mappings.json'
    records=json.loads(prior.read_text(encoding='utf8'))
    products={r['product'] for r in records if Path(r['path']).is_file()}
    cohort=pd.read_csv(ROOT.parent/'conditional_evidence_pilot/cohort.csv').set_index('obs_id')
    p=pd.read_csv(ROOT/'round08_boundary_reference/pairs.csv',dtype={'mmsi':str})
    p=p[p.region.eq('body')&p.scale_m.eq(10)].copy()
    p['history_product']=p.history_id.map(cohort['product']);p['query_product']=p.query_id.map(cohort['product'])
    p['history_verified_raw_product']=p.history_product.isin(products)
    p['query_verified_raw_product']=p.query_product.isin(products)
    cols=['history_id','query_id','pol','mmsi','history_product','query_product','history_verified_raw_product','query_verified_raw_product']
    p[cols].to_csv(OUT/'raw_pair_readiness.csv',index=False)
    a,b=p.history_verified_raw_product,p.query_verified_raw_product
    p[a^b][cols].to_csv(OUT/'one_endpoint_recovery_targets.csv',index=False)
    result=dict(scope='Only previously traced measurement paths, not an exhaustive filesystem/online inventory',
        verified_products=sorted(products),both_endpoints_pair_views=int((a&b).sum()),
        one_endpoint_pair_views=int((a^b).sum()),neither_endpoint_pair_views=int((~a&~b).sum()),
        missing_companion_products=sorted(set(p.loc[a&~b,'query_product'])|set(p.loc[b&~a,'history_product'])),
        downloaded=False,calibrated=False)
    (OUT/'raw_pair_readiness.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
