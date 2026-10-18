"""Fixed source-lineage cohort, vessel-isolated held-date enrollment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
DATA=Path('E:/Hermes/paper_tracking/f0921_chips_clean_npy')
SEED=20261018


def main():
    links=pd.read_csv(HERE/'pixel_links.csv',dtype={'mmsi':str,'obs_time':str})
    index=pd.read_csv(DATA/'index.csv',dtype={'mmsi':str})
    classes=json.loads((HERE.parent/'region_relation_probe/PROTOCOL.json').read_text())['classes']
    a=links[links.lineage_unique & links.crop_source.eq('det') & links.class_raw.isin(classes)].copy()
    matched=index.set_index('row').loc[a.clean_row]
    for left,right in [('mmsi','mmsi'),('class_raw','class_raw'),('source_fid','row_in_shp'),('scene','scene')]:
        np.testing.assert_array_equal(a[left].to_numpy(),matched[right].to_numpy())
    conflicts=a.groupby('mmsi').class_raw.nunique()
    if (conflicts>1).any():
        raise ValueError('Conflicting identity labels need explicit handling before enrollment')
    a=a.sort_values(['obs_time','sensor','code','mmsi','clean_row']).reset_index(drop=True)
    a['label']=a.class_raw.map({c:i for i,c in enumerate(classes)})
    a['date']=a.obs_time.str[:8]
    a['observation_key']=a.sensor+'|'+a.obs_time+'|'+a.code
    product_lookup={}
    for line in Path('E:/Hermes/paper_tracking/f0921_products.txt').read_text().splitlines():
        parts=line.strip().split('_')
        if len(parts)==9:
            product_lookup.setdefault('|'.join([parts[0],parts[4],parts[-1]]),set()).add(line.strip())
    for line in a['product'].dropna().unique():
        parts=line.split('_')
        if len(parts)==9:product_lookup.setdefault('|'.join([parts[0],parts[4],parts[-1]]),set()).add(line)
    a['resolved_product']=a.observation_key.map(lambda k:next(iter(product_lookup[k])) if len(product_lookup.get(k,set()))==1 else '')
    a['acquisition']=a.resolved_product.map(lambda p:'|'.join([p.split('_')[i] for i in [0,6,7]]) if p else '')
    a['row']=np.arange(len(a))
    a.to_csv(HERE/'cohort.csv',index=False)
    pd.crosstab(a.class_raw,a.date).to_csv(HERE/'class_by_date.csv')
    checkpoint=PROJECT/'scene_cross_dataset_redesign_20261014/cross_dataset'
    source=pd.read_csv(PROJECT/'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv',
                       usecols=['mmsi','product'],dtype=str)
    splits=np.load(checkpoint/'source_split.npz')
    keys=source['product'].str.split('_').map(lambda s:'|'.join([s[0],s[4],s[-1]]))
    source_acq=source['product'].str.split('_').map(lambda s:'|'.join([s[0],s[6],s[7]]))
    audit={}
    for part in ['ssl_train','source_validation']:
        r=splits[part]
        audit[part]=dict(rows=len(r),identity_overlap=len(set(a.mmsi)&set(source.iloc[r].mmsi)),
                        product_key_overlap=len(set(a.observation_key)&set(keys.iloc[r])),
                        acquisition_overlap=len(set(a.acquisition)-{''}&set(source_acq.iloc[r])))
        assert audit[part]['identity_overlap']==audit[part]['product_key_overlap']==audit[part]['acquisition_overlap']==0
    audit['unresolved_acquisitions']=int(a.acquisition.eq('').sum())
    train_or_selection=source.iloc[np.r_[splits['ssl_train'],splits['source_validation']]].drop_duplicates('product')
    timing=[]
    for sensor,time in a[['sensor','obs_time']].drop_duplicates().itertuples(index=False,name=None):
        candidates=train_or_selection[train_or_selection['product'].str.startswith(sensor+'_')]
        times=pd.to_datetime(candidates['product'].str.split('_').str[4],format='%Y%m%dT%H%M%S')
        delta=(times-pd.to_datetime(time,format='%Y%m%dT%H%M%S')).abs().dt.total_seconds()/60
        timing.append(dict(sensor=sensor,obs_time=time,nearest_pretrain_minutes=float(delta.min()),
                           nearest_pretrain_product=candidates.loc[delta.idxmin(),'product']))
    pd.DataFrame(timing).to_csv(HERE/'pretraining_time_separation.csv',index=False)
    audit['minimum_time_separation_minutes']=min(t['nearest_pretrain_minutes'] for t in timing)
    audit['limit']='Some full datatake IDs remain unresolved; retain exact product-key and nearest same-sensor time separation evidence, without inventing datatake IDs.'
    (HERE/'pretraining_audit.json').write_text(json.dumps(audit,indent=2))
    episodes=[]; capacity=[]
    for di,(date,g) in enumerate(a.groupby('date',sort=True)):
        # Duplicate observations of one ship never become independent support/query units.
        pool=g.drop_duplicates('mmsi').row.to_numpy()
        nquery=max(8,int(np.ceil(.3*len(pool))))
        shuffled=np.random.default_rng(SEED+di).permutation(pool)
        q=shuffled[:nquery];bank=shuffled[nquery:]
        source=a[(a.date!=date)&~a.mmsi.isin(g.mmsi)].drop_duplicates('mmsi').row.to_numpy()
        valid=len(q)>=8 and len(bank)>=16 and a.iloc[source].label.nunique()==len(classes)
        capacity.append(dict(date=date,observations=len(g),ships=len(pool),queries=len(q),support_pool=len(bank),
                             source_ships=len(source),source_classes=a.iloc[source].label.nunique(),eligible=valid))
        if not valid:continue
        draws=[np.random.default_rng(SEED+1000+di*10+draw).permutation(bank).tolist() for draw in range(5)]
        for budget in [16,32,64]:
            if len(bank)<budget:continue
            episodes.append(dict(fold=str(date),budget=budget,source=source.tolist(),query=q.tolist(),
                                 supports=[s[:budget] for s in draws]))
    pd.DataFrame(capacity).to_csv(HERE/'capacity.csv',index=False)
    (HERE/'episodes.json').write_text(json.dumps(episodes,indent=2))
    config=dict(classes=classes,rows=len(a),ships=a.mmsi.nunique(),dates=a.date.nunique(),
                observation_keys=a.observation_key.nunique(),episodes=len(episodes),source_index=str(DATA/'index.csv'),
                frozen_before_outcomes=True,all_query_support_ship_disjoint=True,
                status='Ready for re-extraction; no reuse of old feature indices')
    (HERE/'enrollment.json').write_text(json.dumps(config,indent=2))
    print(pd.DataFrame(capacity).to_string(index=False));print(json.dumps(config,indent=2))


if __name__=='__main__':main()
