"""Reuse OSS2 fold0 and fixed pass bags; isolate all stack roles by vessel."""
import json
from itertools import combinations
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

from build_inventory import CLASS_MAP, LEGACY, HERE, save_json, view_spec

OSS1 = HERE.parent / 'oss1_m123_replication_20261026'
ROLES = ['fit_train', 'fit_validation', 'utility_train', 'selection', 'evaluation']
BAG_ROLE = {0: 'evaluation', 1: 'selection', 2: 'fit_train', 3: 'fit_train',
            4: 'fit_validation', 5: 'utility_train'}


def load(path):
    return pd.read_csv(path, dtype={'mmsi': str, 'mmsi_or_ship_id': str, 'sample_id': str})


def canonical(table):
    table = table.copy().reset_index(drop=True)
    table['table_index'] = np.arange(len(table))
    table['row_index'] = table.table_index
    table['up_index'] = table.table_index
    table['pass_id'] = table.pass_group
    table['overpass'] = table.pass_group
    table['y'] = table.class_id.astype(int)
    table['reference_candidate'] = table.role.eq('fit_train')
    table['upstream_role'] = np.where(table.role.isin(ROLES[:2]), table.role, 'external')
    return table


def coverage(table):
    return table.groupby(['role', 'class_id', 'original_class_id', 'class_name'], sort=False).agg(
        observations=('obs_id', 'size'), vessels=('mmsi', 'nunique'), passes=('pass_group', 'nunique'),
        valid_vv=('valid_vv', 'sum'), valid_vh=('valid_vh', 'sum'), both_obb=('both_obb', 'sum'))


def cross_dataset(oss2, stack):
    oss1_all = load(OSS1 / 'data_inventory.csv')
    oss1_all['pass_group'] = oss1_all.overpass
    oss1_all = oss1_all.rename(columns={'class_id': 'original_class_id'})
    oss1_all['class_id'] = oss1_all.original_class_id.map(CLASS_MAP).astype('Int64')
    oss1 = oss1_all[oss1_all.original_class_id.isin(CLASS_MAP) & oss1_all.any_obb].copy()
    oss1['class_id'] = oss1.class_id.astype(int)
    oss1['valid_vv'], oss1['valid_vh'] = oss1.obb_vv, oss1.obb_vh
    oss1['input_scale'], oss1['canvas_size'] = 1., 160
    oss1_stack = load(OSS1 / 'stack_table.csv')
    summaries, overlaps, rejected = [], [], []
    common_mmsi = sorted(set(oss1_all.mmsi) & set(oss2.mmsi))
    common_passes = sorted(set(oss1_all.pass_group) & set(oss2.pass_group))
    for key, values in [('mmsi', common_mmsi), ('pass_group', common_passes)]:
        for value in values:
            overlaps.append(dict(key=key, value=value, oss1_observations=int(oss1_all[key].eq(value).sum()),
                                 oss2_observations=int(oss2[key].eq(value).sum())))
    pd.DataFrame(overlaps, columns=['key', 'value', 'oss1_observations', 'oss2_observations']).to_csv(
        HERE / 'cross_dataset_overlap.csv', index=False)
    for direction, source, candidates in [('oss1_to_oss2', oss1_stack, oss2[oss2.runnable]),
                                           ('oss2_to_oss1', stack, oss1)]:
        learned = source[~source.role.eq('evaluation')]
        candidates = candidates.copy()
        candidates['shares_source_mmsi'] = candidates.mmsi.isin(learned.mmsi)
        candidates['shares_source_pass'] = candidates.pass_group.isin(learned.pass_group)
        drop = candidates.shares_source_mmsi | candidates.shares_source_pass
        ex = candidates[drop].copy()
        ex['direction'] = direction
        ex['exclusion_reason'] = np.select(
            [ex.shares_source_mmsi & ex.shares_source_pass, ex.shares_source_mmsi],
            ['source_mmsi_and_pass', 'source_mmsi'], default='source_pass')
        rejected.append(ex)
        target = candidates[~drop].copy()
        target['role'] = 'evaluation'
        target['split_id'] = 'cross_' + direction + '_all_legal_20261026'
        target = canonical(target)
        assert not set(target.mmsi) & set(learned.mmsi)
        assert not set(target.pass_group) & set(learned.pass_group)
        assert set(target.class_id) == set(range(4))
        target.to_csv(HERE / f'cross_{direction}_table.csv', index=False)
        coverage(target).to_csv(HERE / f'cross_{direction}_coverage.csv')
        summaries.append(dict(direction=direction, source_roles=ROLES[:4],
            source_observations=len(learned), source_vessels=learned.mmsi.nunique(),
            candidate_observations=len(candidates), kept_observations=len(target),
            kept_vessels=target.mmsi.nunique(), kept_passes=target.pass_group.nunique(),
            excluded_observations=int(drop.sum()), mmsi_overlap_observations=int(candidates.shares_source_mmsi.sum()),
            pass_overlap_observations=int(candidates.shares_source_pass.sum()),
            target_original_roles='all legal inventory rows, including historical training/validation roles',
            historical_development_exposure=True, new_training=False))
    pd.concat(rejected, ignore_index=True).to_csv(HERE / 'cross_dataset_source_exclusions.csv', index=False)
    save_json('CROSS_DATASET_PROTOCOL.json', dict(
        status='target_queues_ready', overlap_scope='OSS1 full 8-class inventory versus OSS2 selected 4-class GRDH inventory',
        shared_mmsi=len(common_mmsi), shared_passes=common_passes, directions=summaries,
        rule='Freeze each existing source model; exclude target rows sharing MMSI OR pass with ANY source fit/validation/utility/selection row.',
        target_label_tuning_allowed=False, target_support_fit_allowed=False,
        previously_exposed_public_data=True, blind_test=False))
    return summaries


def main():
    inventory = load(HERE / 'data_inventory.csv')
    if not inventory.detection_complete.all():
        raise RuntimeError('Complete real detection coverage before freezing the runnable stack.')
    blockers = inventory[~inventory.all_views_prepare_ok]
    blockers.to_csv(HERE / 'prepare_protocol_blockers.csv', index=False)
    if len(blockers):
        save_json('STACK_SPLIT.json', dict(status='blocked_by_native160_protocol',
            sample_ids=blockers.sample_id.tolist(), rule='Do not resize, replace boxes, or silently remove these rows.'))
        raise RuntimeError('Native160 preparation failed; see prepare_protocol_blockers.csv. User protocol decision required.')
    membership = load(LEGACY / 'membership.csv')
    membership = membership[membership.fold.eq(0)].copy()
    membership['assigned_role'] = membership.pass_bag.map(BAG_ROLE)
    base = inventory.merge(membership[['sample_id', 'role', 'assigned_role']], on='sample_id', validate='one_to_one')
    base = base.rename(columns={'role': 'legacy_role'})
    assert len(base) == len(inventory)
    base['role'] = base.assigned_role
    base['split_exclusion_reason'] = ''
    base.loc[base.legacy_role.str.startswith('excluded'), 'split_exclusion_reason'] = base.legacy_role
    # Preserve the old outer source exclusion, then reserve entire held candidate bags.
    source_members = membership[membership.role.eq('train')]
    reserved = {}
    for role in ['utility_train', 'fit_validation', 'fit_train']:
        members = source_members[source_members.assigned_role.eq(role)]
        rows = base.role.eq(role) & base.split_exclusion_reason.eq('')
        blocked = rows & base.mmsi.isin(reserved)
        base.loc[blocked, 'split_exclusion_reason'] = 'mmsi_reserved_for_' + base.loc[blocked, 'mmsi'].map(reserved).astype('string')
        for mmsi in members.mmsi:
            reserved.setdefault(mmsi, role)
    base.loc[base.split_exclusion_reason.eq('') & ~base.runnable, 'split_exclusion_reason'] = base.exclusion_reason
    base.to_csv(HERE / 'stack_membership.csv', index=False)
    excluded = base[base.split_exclusion_reason.ne('')].copy()
    excluded.to_csv(HERE / 'stack_exclusions.csv', index=False)
    excluded[excluded.split_exclusion_reason.str.contains('mmsi')].to_csv(HERE / 'stack_mmsi_exclusions.csv', index=False)
    selected = base[base.split_exclusion_reason.eq('')].copy()
    table = pd.concat([selected[selected.role.eq(r)].sort_values('obs_id') for r in ROLES], ignore_index=True)
    table['split_id'] = 'oss2_legacy_fold0_four_class_stack_20261026'
    table = canonical(table)
    checks = []
    for a, b in combinations(ROLES, 2):
        left, right = table[table.role.eq(a)], table[table.role.eq(b)]
        check = dict(role_a=a, role_b=b)
        for key in ['sample_id', 'mmsi', 'pass_group']:
            check[key + '_overlap'] = len(set(left[key]) & set(right[key]))
            assert check[key + '_overlap'] == 0, check
        checks.append(check)
    for role in ROLES:
        assert set(table[table.role.eq(role)].class_id) == set(range(4)), role
    bank = table[table.role.eq('fit_train')].drop_duplicates('mmsi').copy()
    assert bank.groupby('class_id').size().min() >= 8
    table.to_csv(HERE / 'stack_table.csv', index=False)
    for role in ROLES:
        table[table.role.eq(role)].to_csv(HERE / f'stack_{role}.csv', index=False)
    save_json('stack_indices.json', {r: np.flatnonzero(table.role.eq(r)).tolist() for r in ROLES})
    bank.to_csv(HERE / 'stack_reference_candidates.csv', index=False)
    coverage(table).to_csv(HERE / 'stack_coverage.csv')
    table.groupby(['role', 'pass_group', 'class_id']).size().rename('observations').to_csv(HERE / 'stack_pass_coverage.csv')
    assignments = pd.read_csv(LEGACY / 'pass_assignment.csv')
    assignments['role'] = assignments.bag.map(BAG_ROLE)
    assignments.to_csv(HERE / 'stack_pass_assignment.csv', index=False)
    receipt = dict(status='ready_metadata_only', split_id=table.split_id.iloc[0], outer_fold=0,
        parent_split=str(LEGACY / 'membership.csv'), partition_seed=20261023, pass_bag_role=BAG_ROLE,
        new_partition_draws=0, target_outcomes_used_for_split=False, class_map=CLASS_MAP,
        roles={r: dict(observations=len(table[table.role.eq(r)]), vessels=table[table.role.eq(r)].mmsi.nunique(),
                       passes=sorted(table[table.role.eq(r)].pass_group.unique())) for r in ROLES},
        isolation=checks, exclusions=excluded.split_exclusion_reason.value_counts().to_dict(),
        references_per_class=8, total_references=32, reference_source_role='fit_train',
        reference_candidate_vessels={int(c): len(g) for c, g in bank.groupby('class_id')},
        reference_selection='same source-only OSS1 KMeans(8, random_state=17, n_init=10), after extraction',
        bank_selected=False, padding_repeated_references=False, seeds=[17, 29, 43],
        native_canvas=160, resized=0, radar_axis_valid=0, training_run=False, git_used=False)
    save_json('STACK_SPLIT.json', receipt)
    print('STACK READY', json.dumps({r: receipt['roles'][r]['observations'] for r in ROLES}), flush=True)
    print('BANK', receipt['reference_candidate_vessels'], flush=True)
    summaries = cross_dataset(inventory, table)
    print('CROSS', json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
