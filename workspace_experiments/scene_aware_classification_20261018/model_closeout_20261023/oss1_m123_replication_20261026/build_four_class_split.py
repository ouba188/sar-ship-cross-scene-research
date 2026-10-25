"""Freeze existing OSS1 pilot roles using metadata only; no model imports."""
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEGACY = ROOT / 'scene_cross_dataset_redesign_20261014'
CLASS_MAP = {0: 0, 2: 1, 3: 2, 4: 3}
INPUTS = {
    'source': LEGACY / 'strict_training/oss1/source_train.csv',
    'val': LEGACY / 'strict_training/oss1/source_validation.csv',
    'test': LEGACY / 'public_split/oss1_query.csv',
}


def main():
    inventory = pd.read_csv(HERE / 'data_inventory.csv', dtype={'mmsi': str, 'mmsi_or_ship_id': str})
    assert inventory.sample_id.is_unique
    inventory = inventory.rename(columns={'class_id': 'original_class_id'})
    # round23 fold assignments are unrelated to this legacy pilot protocol.
    inventory = inventory.drop(columns=['outer_fold'])
    views = pd.read_csv(HERE / 'view_inventory.csv')
    parts, rejected = [], []
    for role, path in INPUTS.items():
        legacy = pd.read_csv(path, dtype={'mmsi': str})
        assert legacy.sample_id.is_unique
        frame = legacy[['sample_id', 'mmsi', 'acquisition']].merge(
            inventory, on='sample_id', validate='one_to_one', suffixes=('_legacy', ''))
        assert len(frame) == len(legacy)
        assert frame.mmsi.eq(frame.mmsi_legacy).all()
        assert frame.acquisition.eq(frame.overpass).all()
        frame = frame.drop(columns=['mmsi_legacy', 'acquisition'])
        frame.insert(0, 'role', role)
        frame.insert(0, 'split_id', 'oss1_legacy_pilot_four_class_20261026')
        frame['class_id'] = frame.original_class_id.map(CLASS_MAP).astype('Int64')
        frame['exclusion_reason'] = ''
        frame.loc[~frame.original_class_id.isin(CLASS_MAP), 'exclusion_reason'] = 'outside_four_class_scope'
        frame.loc[frame.original_class_id.isin(CLASS_MAP) & ~frame.any_obb, 'exclusion_reason'] = 'no_obb_either_polarization_conf025'
        rejected.append(frame[frame.exclusion_reason.ne('')].copy())
        frame = frame[frame.exclusion_reason.eq('')].drop(columns='exclusion_reason').copy()
        frame['class_id'] = frame.class_id.astype(int)
        frame['pass_group'] = frame.overpass
        frame['valid_vv'] = frame.obb_vv
        frame['valid_vh'] = frame.obb_vh
        frame['input_scale'] = 1.0
        frame['canvas_size'] = 160
        frame['reference_candidate'] = role == 'source'
        frame = frame.sort_values('obs_id').reset_index(drop=True)
        assert set(frame.class_id) == set(range(4)), (role, 'missing class')
        parts.append(frame)
    checks = []
    for a, b in combinations(parts, 2):
        check = dict(role_a=a.role.iloc[0], role_b=b.role.iloc[0],
                     sample_overlap=len(set(a.sample_id) & set(b.sample_id)),
                     mmsi_overlap=len(set(a.mmsi) & set(b.mmsi)),
                     pass_overlap=len(set(a.pass_group) & set(b.pass_group)))
        assert all(check[k] == 0 for k in ('sample_overlap', 'mmsi_overlap', 'pass_overlap')), check
        checks.append(check)
    table = pd.concat(parts, ignore_index=True)
    table.insert(0, 'table_index', range(len(table)))
    source = table[table.role.eq('source')]
    candidates = source.drop_duplicates('mmsi', keep='first').copy()
    assert candidates.groupby('class_id').size().min() >= 8
    held = pd.read_csv(LEGACY / 'public_split/oss1_support_candidates.csv', dtype={'mmsi': str})
    assert not set(source.mmsi) & set(held.mmsi)
    assert not set(source.pass_group) & set(held.acquisition)
    selected_views = views[views.obs_id.isin(table.obs_id)].copy()
    assert len(selected_views) == len(table) * 2
    assert selected_views['exists'].all()
    table.to_csv(HERE / 'four_class_roles.csv', index=False)
    for role in INPUTS:
        table[table.role.eq(role)].to_csv(HERE / f'four_class_{role}.csv', index=False)
    pd.concat(rejected, ignore_index=True).to_csv(HERE / 'four_class_exclusions.csv', index=False)
    selected_views.to_csv(HERE / 'four_class_views.csv', index=False)
    candidates.to_csv(HERE / 'four_class_reference_candidates.csv', index=False)
    coverage = table.groupby(['role', 'class_id', 'original_class_id', 'class_name'], sort=False).agg(
        observations=('obs_id', 'size'), vessels=('mmsi', 'nunique'), passes=('pass_group', 'nunique'),
        valid_vv=('valid_vv', 'sum'), valid_vh=('valid_vh', 'sum'), both_obb=('both_obb', 'sum'))
    coverage.to_csv(HERE / 'four_class_coverage.csv')
    table.groupby(['role', 'pass_group', 'class_id']).size().rename('observations').to_csv(HERE / 'four_class_pass_coverage.csv')
    receipt = dict(
        status='fixed_roles_ready_no_training', split_id=table.split_id.iloc[0],
        choice='earliest existing strict OSS1 pilot; no target outcomes or new fold search',
        inputs={role: str(path) for role, path in INPUTS.items()}, class_map=CLASS_MAP,
        roles={role: dict(n=len(g), vessels=g.mmsi.nunique(), passes=sorted(g.pass_group.unique()))
               for role, g in table.groupby('role', sort=False)},
        isolation=checks, reference_vessels_by_class={int(c): len(g) for c, g in candidates.groupby('class_id')},
        references_per_class=8, total_references=32, references_per_query_class=2,
        bank_selection='existing source-only KMeans(n_clusters=8, random_state=17, n_init=10), after feature extraction',
        bank_selected=False, repeated_reference_padding_allowed=False,
        legacy_target_support_pool_used=False, source_val_refit_allowed=False,
        test_scope='one historically used acquisition; not blind or all OSS1',
        original_datasets_modified=False, gpu_used=False, training_run=False, git_used=False)
    (HERE / 'FOUR_CLASS_SPLIT.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    print(coverage.to_string())


if __name__ == '__main__':
    main()
