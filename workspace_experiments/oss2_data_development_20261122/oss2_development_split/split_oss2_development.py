"""Preserve OSS2 historical boundaries; audit metadata only, never train/read pixels.

Run with Python. Local IDs/paths in outputs are private working artifacts.
The selected protocol is immutable role mapping, not a performance-based search.
"""
from pathlib import Path
from itertools import combinations
import json
import re
import pandas as pd

OUT = Path(__file__).resolve().parent
OLD = Path('E:/临时会话/m1_coverage_20261101')
META = Path('E:/临时会话/all_factor_probe_20261017/metadata/oss2_factor_metadata.csv')
SOURCE = Path('E:/临时会话/oss2_port_prototypes_20261121/source_observations_local.csv')
ROLE = {'fit_train':'train','utility_train':'train','fit_validation':'val',
        'evaluation':'test','selection':'quarantine_historical_selection'}
Q_VIEWED = {r['id'] for r in json.loads((OUT.parent/'q_review_selection_local.json').read_text(encoding='utf8'))}
_water = pd.read_csv(OUT.parent/'q_review_water_context_local.csv')
Q_WATER_CONTEXT = set(_water.loc[_water.coast_status.eq('offshore'), 'sample_id']) & Q_VIEWED
KEYS = ('mmsi','canonical_scene','pass_group','absolute_orbit_key')


def component_ids(frame):
    parent = {i:i for i in frame.index}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for key in KEYS:
        for _, group in frame.groupby(key):
            first = group.index[0]
            for i in group.index: parent[find(i)] = find(first)
    roots = {i:find(i) for i in frame.index}
    labels = {r:min(frame.loc[i,'sample_id'] for i in roots if roots[i] == r) for r in set(roots.values())}
    return pd.Series({i:labels[r] for i,r in roots.items()})


def overlap_records(frame, group):
    result = []
    for a,b in combinations(sorted(frame[group].unique()),2):
        x,y = frame[frame[group] == a],frame[frame[group] == b]
        for key in ('sample_id',*KEYS):
            common = sorted(set(x[key]) & set(y[key]))
            result.append(dict(grouping=group,left=a,right=b,key=key,overlap_count=len(common),values=';'.join(common)))
    return result


def main():
    rows = json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    data = []
    for index,r in enumerate(rows):
        scene = re.sub(r'\(\d+\)$','',r['scene_id'])
        orbit = re.search(r'^(S1[AB])_.*?_(\d{6})_[0-9A-F]{6}_',scene)
        assert orbit, scene
        entry = dict(sample_id=r['id'],manifest_index=index,label=r['label'],class_name=r['class_name'],
                     mmsi=str(r['mmsi']),historical_role=r['role'],historical_split=r['split'],
                     scene_id=r['scene_id'],canonical_scene=scene,pass_group=r['pass_group'],
                     absolute_orbit_key='_'.join(orbit.groups()),development_split=ROLE[r['role']],
                     historical_training_exposed=r['role'] in ('fit_train','utility_train','selection'),
                     historical_validation_exposed=r['role']=='fit_validation',
                     historical_evaluation_exposed=r['role']=='evaluation',
                     genuinely_unseen_test=False,viewed_for_Q_20261122=r['id'] in Q_VIEWED,
                     Q_water_context_review_passed=r['id'] in Q_WATER_CONTEXT,
                     Q_donor_training_split_eligible=r['id'] in Q_WATER_CONTEXT and ROLE[r['role']]=='train',
                     source_alignment_audit_scope=r['role'] in ('fit_train','utility_train'),
                     alignment_qualification='source_audit_available' if r['role'] in ('fit_train','utility_train') else 'pending_not_inherited_from_source',
                     formal_training_ready=False)
        for pol in ('vv','vh'):
            view = r.get(pol)
            entry[pol+'_manifest_view_available'] = bool(view)
            entry[pol+'_manifest_png_exists'] = bool(view and Path(view['image']).is_file())
            entry[pol+'_native_uint8_exists'] = Path(r['source_paths'][pol]).is_file()
        entry['both_manifest_views_available'] = entry['vv_manifest_view_available'] and entry['vh_manifest_view_available']
        data.append(entry)
    d = pd.DataFrame(data)
    assert len(d) == 1519 and d.sample_id.is_unique
    source_ids = set(pd.read_csv(SOURCE,usecols=['sample_id']).sample_id)
    assert set(d.loc[d.development_split.eq('train'),'sample_id']) == source_ids and len(source_ids)==719
    assert set(d.loc[d.viewed_for_Q_20261122,'development_split']) == {'train'}
    meta = pd.read_csv(META,usecols=['sample_id','mmsi','scene_id','identity_match_valid','product_metadata_available'],dtype={'mmsi':str})
    d = d.merge(meta.rename(columns={'mmsi':'metadata_mmsi','scene_id':'metadata_scene'}),on='sample_id',how='left',validate='one_to_one')
    assert d.mmsi.eq(d.metadata_mmsi).all() and d.scene_id.eq(d.metadata_scene).all()
    d['joint_component_full'] = component_ids(d)
    source = d[d.development_split.eq('train')].copy()
    source['joint_component_source_only'] = component_ids(source)
    d['joint_component_source_only'] = source.joint_component_source_only.reindex(d.index).fillna('')
    primary = d[d.development_split.isin(['train','val','test'])]
    overlaps = overlap_records(primary,'development_split') + overlap_records(d,'historical_role')
    assert all(r['overlap_count']==0 for r in overlaps if r['grouping']=='development_split')
    classes = pd.crosstab(d.development_split,d.class_name)
    counts = d.groupby('development_split').agg(observations=('sample_id','size'),ships=('mmsi','nunique'),
                scenes=('canonical_scene','nunique'),passes=('pass_group','nunique'),absolute_orbits=('absolute_orbit_key','nunique'),
                both_manifest_views=('both_manifest_views_available','sum'))
    components = source.groupby('joint_component_source_only').agg(observations=('sample_id','size'),
                  scenes=('canonical_scene','nunique'),q_viewed=('viewed_for_Q_20261122','sum'))
    # Feasibility example only: no new assignment file or activated protocol.
    candidate = source.copy()
    candidate['hypothetical'] = candidate.scene_id.map(lambda s:'val' if '20160515T' in s else 'test' if '20160608T' in s else 'train')
    cross = set(candidate.groupby('mmsi').hypothetical.nunique().loc[lambda x:x>1].index)
    retained = candidate[~candidate.mmsi.isin(cross)]
    assert all(r['overlap_count']==0 for r in overlap_records(retained,'hypothetical'))
    # The public source paths exist even where the old detector omitted a view.
    d.to_csv(OUT/'split_assignment_local.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(overlaps).to_csv(OUT/'overlap_audit_local.csv',index=False,encoding='utf-8-sig')
    counts.to_csv(OUT/'split_counts.csv',encoding='utf-8-sig')
    classes.to_csv(OUT/'class_counts.csv',encoding='utf-8-sig')
    components.to_csv(OUT/'source_components_local.csv',encoding='utf-8-sig')
    summary = dict(protocol='preserve_legacy_719_228_378_v1',manifest_rows=len(d),metadata_rows=len(meta),
                   metadata_joined_rows=int(d.metadata_mmsi.notna().sum()),identity_match_valid=int(d.identity_match_valid.sum()),
                   product_metadata_available=int(d.product_metadata_available.sum()),
                   split_counts=counts.to_dict('index'),class_counts=classes.to_dict('index'),
                   source_joint_component_sizes=sorted(components.observations.tolist(),reverse=True),
                   source_components_locked_train_by_Q=int(components.loc[components.q_viewed>0,'observations'].sum()),
                   primary_all_identity_scene_pass_orbit_overlaps_zero=True,
                   full_joint_components=d.joint_component_full.nunique(),
                   historical_role_orbit_overlaps=[r for r in overlaps if r['grouping']=='historical_role' and r['overlap_count']],
                   native_uint8_exists={p:int(d[p+'_native_uint8_exists'].sum()) for p in ('vv','vh')},
                   manifest_view_available={p:int(d[p+'_manifest_view_available'].sum()) for p in ('vv','vh')},
                   new_unseen_test_rows=0,current_alignment_audit_scope_rows=719,
                   val_test_alignment_qualification_pending=606,training_performed=False,image_pixels_read=False,
                   hypothetical_source_only_scene_split=dict(activated=False,excluded_cross_group_mmsis=len(cross),
                       excluded_observations=len(candidate)-len(retained),retained_counts=retained.hypothetical.value_counts().to_dict(),
                       retained_class_counts=pd.crosstab(retained.hypothetical,retained.class_name).to_dict('index')),
                   exposure_evidence=[str(OLD/'experiment.py'),str(OLD/'small_class_s17/evaluation.npz'),
                       str(OLD/'expanded_class_s17/evaluation.npz')])
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
