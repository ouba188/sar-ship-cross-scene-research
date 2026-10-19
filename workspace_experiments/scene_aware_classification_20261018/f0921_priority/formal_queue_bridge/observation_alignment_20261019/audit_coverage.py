"""Coverage audit only: no model fits, relabeling, or new task selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
RUN=PARENT/'orbit_geometry_readout_20261019'


def pass_key(product):
    parts=product.split('_')
    return '|'.join([parts[0],parts[6],parts[7]])


def axial_distance(a,b):
    return np.abs((a-b+np.pi/2)%np.pi-np.pi/2)*180/np.pi


def main():
    meta=pd.read_csv(PARENT/'cohort.csv',dtype={'date':str,'mmsi':str})
    geo=pd.read_csv(RUN/'orbit_geometry.csv')
    axes=pd.read_csv(RUN/'axes.csv')
    np.testing.assert_array_equal(meta.row,geo.row)
    np.testing.assert_array_equal(meta.row,axes.row)
    assert meta.row.tolist()==list(range(len(meta)))
    meta['pass_key']=geo.resolved_product.map(pass_key)
    meta['readout_valid']=axes.readout_valid.to_numpy(bool)
    meta['orbit_valid']=geo.orbit_valid.to_numpy(bool)
    meta['los_deg']=np.degrees(geo.los_angle_image_rad)
    meta['incidence_deg']=geo.orbit_incidence_deg
    meta['relative_axis_rad']=(axes.ship_axis_rad-geo.los_angle_image_rad)%np.pi
    rows,budgets=[],[]
    tasks=json.loads((PARENT/'episodes.json').read_text())
    for t in tasks:
        q=np.array(t['query'])
        budgets.append(dict(fold=t['fold'],budget=t['budget'],queries=len(q),
                            valid_queries=int(meta.iloc[q].readout_valid.sum()),query_rows=json.dumps(q.tolist())))
        for draw,support in enumerate(t['supports']):
            assert not set(meta.iloc[q].mmsi)&set(meta.iloc[support].mmsi)
            for i in q:
                for j in support:
                    a,b=meta.iloc[i],meta.iloc[j]
                    valid=bool(a.readout_valid and b.readout_valid)
                    orbit=bool(a.orbit_valid and b.orbit_valid)
                    rows.append(dict(fold=t['fold'],budget=t['budget'],draw=draw,query=int(i),support=int(j),
                        query_mmsi=a.mmsi,support_mmsi=b.mmsi,query_class=int(a.label),support_class=int(b.label),
                        same_class=bool(a.label==b.label),query_pass=a.pass_key,support_pass=b.pass_key,
                        cross_pass=bool(a.pass_key!=b.pass_key),readout_valid=valid,orbit_valid=orbit,
                        relative_angle_gap_deg=axial_distance(a.relative_axis_rad,b.relative_axis_rad) if valid else np.nan,
                        incidence_gap_deg=abs(a.incidence_deg-b.incidence_deg) if valid else np.nan))
    edges=pd.DataFrame(rows)
    # Event rows include repeated support draws. Distinct directed pairs are separate.
    summaries=[]
    for gate in ['orbit_valid','readout_valid']:
        subset=edges[edges[gate]]
        for budget,frame in [('ALL',subset)]+list(subset.groupby('budget')):
            for name,g in [('events',frame),('unique_pairs',frame.drop_duplicates(['query','support']))]:
                same=g[g.same_class]
                summaries.append(dict(gate=gate,unit=name,budget=budget,pairs=len(g),cross_pass=int(g.cross_pass.sum()),
                    cross_pass_fraction=float(g.cross_pass.mean()),same_class_pairs=len(same),
                    same_class_cross_pass=int(same.cross_pass.sum()),
                    distinct_query_rows=g['query'].nunique(),distinct_query_mmsi=g.query_mmsi.nunique(),
                    query_with_cross_pass_same_class=same.loc[same.cross_pass,'query'].nunique()))
    pd.DataFrame(summaries).to_csv(HERE/'coverage_summary.csv',index=False)
    edges[edges.readout_valid].to_csv(HERE/'valid_pair_events.csv',index=False)
    unique=edges[edges.readout_valid].drop_duplicates(['query','support']).copy()
    unique['angle_bin']=pd.cut(unique.relative_angle_gap_deg,[0,15,30,60,90],include_lowest=True)
    unique['incidence_bin']=pd.cut(unique.incidence_gap_deg,[0,2,5,10,90],include_lowest=True)
    unique.groupby(['same_class','cross_pass','angle_bin','incidence_bin'],observed=True).size().rename('distinct_pairs').to_csv(HERE/'condition_pair_coverage.csv')
    pd.DataFrame(budgets).to_csv(HERE/'budget_queries.csv',index=False)
    valid=meta[meta.readout_valid]
    valid.groupby(['pass_key','label']).agg(observations=('row','size'),ships=('mmsi','nunique')).to_csv(HERE/'pass_class_inventory.csv')
    valid[['row','mmsi','date','pass_key','label','los_deg','incidence_deg','relative_axis_rad']].to_csv(HERE/'valid_geometry_inventory.csv',index=False)
    # Potential cross-pass partners are an inventory, not a deployed support rule.
    candidates=[]
    for item in valid.itertuples():
        pool=valid[(valid.pass_key!=item.pass_key)&(valid.mmsi!=item.mmsi)]
        same=pool[pool.label==item.label]
        candidates.append(dict(row=item.row,label=item.label,pass_key=item.pass_key,
            other_pass_pool_rows=len(pool),other_pass_pool_ships=pool.mmsi.nunique(),
            other_pass_same_class_rows=len(same),other_pass_same_class_ships=same.mmsi.nunique(),
            other_pass_same_class_passes=same.pass_key.nunique()))
    pd.DataFrame(candidates).to_csv(HERE/'potential_cross_pass_partners.csv',index=False)
    feasibility=[]
    for query_pass,qgroup in valid.groupby('pass_key'):
        for support_pass,sgroup in valid.groupby('pass_key'):
            if query_pass==support_pass:
                continue
            sp=sgroup[~sgroup.mmsi.isin(qgroup.mmsi)]
            held_dates=set(qgroup.date)|set(sgroup.date)
            held_ships=set(meta.loc[meta.date.isin(held_dates),'mmsi'])
            source=meta[~meta.date.isin(held_dates)&~meta.mmsi.isin(held_ships)]
            source_geometry=source[source.readout_valid]
            missing=sorted(set(range(7))-set(source.label))
            feasibility.append(dict(query_pass=query_pass,support_pass=support_pass,
                query_rows=len(qgroup),query_ships=qgroup.mmsi.nunique(),support_ships=sp.mmsi.nunique(),
                source_rows=len(source),source_ships=source.mmsi.nunique(),
                source_geometry_rows=len(source_geometry),source_geometry_ships=source_geometry.mmsi.nunique(),
                source_geometry_classes=json.dumps(sorted(source_geometry.label.unique().tolist())),
                source_missing_classes=json.dumps(missing),source_all_seven=not missing,
                held_dates=json.dumps(sorted(held_dates))))
    pd.DataFrame(feasibility).to_csv(HERE/'prospective_pass_pair_capacity.csv',index=False)
    budget_sets={b:set(np.concatenate([np.array(t['query']) for t in tasks if t['budget']==b])) for b in [16,32,64]}
    common=set.intersection(*budget_sets.values())
    report=dict(observations=len(meta),readout_valid=len(valid),passes=valid.pass_key.nunique(),
        readout_los_169_to_171=int(valid.los_deg.between(169,171).sum()),
        readout_los_168_5_to_171_5=int(valid.los_deg.between(168.5,171.5).sum()),
        query_rows_by_budget={str(b):len(s) for b,s in budget_sets.items()},common_queries=len(common),
        common_query_dates=sorted(meta.iloc[sorted(common)].date.unique().tolist()),
        cross_pass_same_class_available_rows=sum(x['other_pass_same_class_rows']>0 for x in candidates),
        cross_pass_same_class_unavailable_rows=sum(x['other_pass_same_class_rows']==0 for x in candidates),
        pass_pairs_with_query8_support16_source7=sum(x['query_ships']>=8 and x['support_ships']>=16 and x['source_all_seven'] for x in feasibility),
        same_scene_is_not_same_geometry=True,training=False,predictions_changed=False,
        note='Potential partner counts use development labels for coverage diagnosis only, not query-time support selection.')
    (HERE/'audit_receipt.json').write_text(json.dumps(report,indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
