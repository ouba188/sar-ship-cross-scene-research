"""Materialize one fixed, source-only metadata split for the OSS1 stack."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROLES = ['fit_train', 'fit_validation', 'utility_train', 'selection', 'evaluation']


def main():
    old = pd.read_csv(HERE / 'four_class_roles.csv', dtype={'mmsi': str, 'mmsi_or_ship_id': str})
    old = old.rename(columns={'role': 'legacy_role', 'table_index': 'legacy_table_index'})
    source = old[old.legacy_role.eq('source')].copy()
    groups = np.random.default_rng(17).permutation(sorted(source.pass_group.unique()))
    assert len(groups) == 16
    cuts = [0, 9, 12, 16]
    assignments = {str(p): ROLES[j] for j in range(3) for p in groups[cuts[j]:cuts[j + 1]]}
    source['role'] = source.pass_group.map(assignments)
    kept, excluded, held_by = {}, [], {}
    for role in reversed(ROLES[:3]):
        part = source[source.role.eq(role)].copy()
        drop = part.mmsi.isin(held_by)
        rejected = part[drop].copy()
        rejected['exclusion_reason'] = 'mmsi_reserved_for_' + rejected.mmsi.map(held_by).astype('string')
        excluded.append(rejected)
        kept[role] = part[~drop].copy()
        held_by.update({mmsi: role for mmsi in kept[role].mmsi})
    for legacy_role, role in [('val', 'selection'), ('test', 'evaluation')]:
        kept[role] = old[old.legacy_role.eq(legacy_role)].copy()
        kept[role]['role'] = role
    table = pd.concat([kept[r].sort_values('obs_id') for r in ROLES], ignore_index=True)
    table['split_id'] = 'oss1_legacy_pilot_four_class_stack_20261026'
    table.insert(0, 'table_index', np.arange(len(table)))
    table['row_index'] = table.table_index
    table['up_index'] = table.table_index
    table['upstream_role'] = np.where(table.role.isin(ROLES[:2]), table.role, 'external')
    table['reference_candidate'] = table.role.eq('fit_train')
    checks = []
    for a, b in combinations(ROLES, 2):
        left, right = table[table.role.eq(a)], table[table.role.eq(b)]
        check = dict(role_a=a, role_b=b)
        for key in ('sample_id', 'mmsi', 'pass_group'):
            check[key + '_overlap'] = len(set(left[key]) & set(right[key]))
            assert check[key + '_overlap'] == 0, check
        checks.append(check)
    for role in ROLES:
        assert set(table[table.role.eq(role)].class_id) == set(range(4))
    bank = table[table.role.eq('fit_train')].drop_duplicates('mmsi').copy()
    assert bank.groupby('class_id').size().min() >= 8
    table.to_csv(HERE / 'stack_table.csv', index=False)
    for role in ROLES:
        table[table.role.eq(role)].to_csv(HERE / f'stack_{role}.csv', index=False)
    np.savez_compressed(HERE / 'stack_indices.npz', **{
        role: np.flatnonzero(table.role.eq(role)).astype(np.int64) for role in ROLES})
    pd.concat(excluded, ignore_index=True).to_csv(HERE / 'stack_mmsi_exclusions.csv', index=False)
    bank.to_csv(HERE / 'stack_reference_candidates.csv', index=False)
    table.groupby(['role', 'class_id', 'original_class_id', 'class_name'], sort=False).agg(
        observations=('obs_id', 'size'), vessels=('mmsi', 'nunique'), passes=('pass_group', 'nunique'),
        valid_vv=('valid_vv', 'sum'), valid_vh=('valid_vh', 'sum'), both_obb=('both_obb', 'sum')
    ).to_csv(HERE / 'stack_coverage.csv')
    table.groupby(['role', 'pass_group', 'class_id']).size().rename('observations').to_csv(HERE / 'stack_pass_coverage.csv')
    receipt = dict(status='ready_metadata_only', split_id=table.split_id.iloc[0],
        parent_split='oss1_legacy_pilot_four_class_20261026', partition_seed=17,
        source_pass_counts=[9, 3, 4], source_pass_assignments=assignments,
        identity_priority=['utility_train', 'fit_validation', 'fit_train'],
        roles={r: dict(observations=len(table[table.role.eq(r)]),
                       vessels=table[table.role.eq(r)].mmsi.nunique(),
                       passes=sorted(table[table.role.eq(r)].pass_group.unique())) for r in ROLES},
        excluded_source_observations=sum(len(x) for x in excluded), isolation=checks,
        references_per_class=8, total_references=32,
        reference_source_role='fit_train',
        reference_candidate_vessels={int(c): len(g) for c, g in bank.groupby('class_id')},
        bank_selected=False, padding_repeated_references=False,
        bank_selection='source-only original KMeans(8, random_state=17, n_init=10) after extraction',
        source_fit_selection_shared=False, whole_stack_nested_oof=False,
        seeds=[17, 29, 43], outer_folds=1, target_outcomes_used_for_split=False,
        training_run=False, gpu_used=False, git_used=False)
    (HERE / 'STACK_SPLIT.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({r: receipt['roles'][r]['observations'] for r in ROLES}))
    print('Excluded source observations:', receipt['excluded_source_observations'])
    print('Reference candidate vessels:', receipt['reference_candidate_vessels'])


if __name__ == '__main__':
    main()
