"""Small CPU contract checks; no training or complete transfer run."""
import sys
sys.dont_write_bytecode = True
import argparse
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import transfer as t


def identity_checks(out):
    source = pd.DataFrame(dict(obs_id=range(5), mmsi=['10', '20', '30', '40', '50'],
        role=[*t.SUPERVISED, 'evaluation'],
        pass_group=[f'S1A_{i:06d}_00AC44' for i in range(1, 6)]))
    target = pd.DataFrame(dict(obs_id=range(8), mmsi=['10', '20', '30', '40', '60', '50', '70', '80'],
        pass_group=['OSS2_S1A_000099_00AC44']*4+['OSS2_S1A_000004_00AC44',
                     'OSS2_S1A_000005_00AC44', 'OSS2_S1A_000100_00AC44', 'OSS2_S1A_000101_00AC44']))
    views = pd.DataFrame(dict(obs_id=range(8), exists=True, detected=True, axis_valid=True, confidence=.9))
    views.loc[7, 'exists'] = False
    views.loc[6, 'axis_valid'] = False
    selected, reason = t.select_target(source, target, views)
    np.testing.assert_array_equal(selected.obs_id, [5, 6])
    assert reason.source_mmsi_overlap.iloc[:4].all()
    assert reason.source_pass_overlap.iloc[4]
    merged = t.mapped_tables(source, selected, Path('OSS1'), Path('OSS2'))
    assert merged.obs_id.is_unique and len(merged) == 7
    np.testing.assert_array_equal(merged.original_obs_id.iloc[5:], [5, 6])
    # Class columns are not even materialized in the inference metadata frame.
    table = target.assign(class_id=np.arange(8), original_class_id=99, label_value='hidden', ais_type=80,
                          y=np.arange(8), ais_ship_type=80)
    path = out / 'label_isolation_fixture.csv'
    table.to_csv(path, index=False)
    metadata = t.read_table(path)
    assert not set(metadata) & {'class_id', 'original_class_id', 'label_value', 'ais_type', 'y', 'ais_ship_type'}
    try:
        t.evaluate_saved(out, Path('unreadable_labels.csv'), selected, [17])
    except RuntimeError:
        pass
    else:
        raise AssertionError('Label loading was not blocked before prediction save.')
    return dict(all_four_supervised_roles_excluded=True, physical_pass_prefix_normalized=True,
                evaluation_role_not_treated_as_supervision=True, no_usable_view_excluded=True,
                colliding_obs_ids_mapped=True, target_labels_absent=True, metrics_save_barrier=True)


def frozen_tau_check(out):
    with TemporaryDirectory(dir=out) as temporary:
        root = Path(temporary)
        models, cache, dest = root/'source'/'models', root/'cache', root/'summaries'
        models.mkdir(parents=True)
        cache.mkdir()
        tau = {'layer1': [2., 3.], 'layer3': [4., 5.]}
        t.write_json(models.parent/'scales.json', {'tau': tau, 'fit_obs_id': [101]})
        for layer, dim in (('layer1', 256), ('layer3', 1024)):
            np.savez(models/f'reader_{layer}.npz', W=np.zeros((4, 2*dim)))
        np.savez(cache/'frontend_raw.npz', obs_id=[1], footprint=np.zeros((1, 2, 160, 160)),
                 obb_geometry=np.zeros((1, 2, 6)), valid=np.zeros((1, 2), bool),
                 quality=np.zeros((1, 2, 3, 4)), geometry=np.zeros((1, 2, 4)),
                 region_valid=np.zeros((1, 2, 3), bool), global3=np.zeros((1, 2, 1024)))
        table = pd.DataFrame({'obs_id': [1]})
        with patch.object(t.extract, 'backbone', return_value=lambda value: None), \
             patch.object(t.extract, 'prepare_inventory', return_value=[{}]), \
             patch.object(t.extract, 'prepare', return_value=None), \
             patch.object(np, 'median', side_effect=AssertionError('Target tau fit attempted')):
            sums = t.extract.extract_summaries(table, np.array([], int), cache, models, dest,
                                               device='cpu', frozen_tau=tau)
        scales = t.read_json(dest/'scales.json')
        assert scales['tau'] == tau and scales['fit_obs_id'] == [] and scales['frozen']
        assert not (dest/'scale_sigma.npz').exists()
        assert not sums['mean3'].any()
    return dict(source_tau_preserved=True, empty_target_fit_indices=True,
                median_forbidden_during_frozen_extraction=True, source_provenance_saved=True)


def cache_merge_check(out):
    with TemporaryDirectory(dir=out) as temporary:
        root = Path(temporary)
        cache = root/'existing'
        cache.mkdir()
        table = pd.DataFrame(dict(obs_id=[3, 1, 2], sample_id=['c', 'a', 'b']))
        for name in ('frontend_raw.npz', 'm3_raw.npz'):
            np.savez(cache/name, obs_id=[1, 3], sample_id=['a', 'c'], value=[10, 30])
        def extract_missing(rows, dest, device, weights):
            np.testing.assert_array_equal(rows.obs_id, [2])
            dest.mkdir(parents=True)
            for name in ('frontend_raw.npz', 'm3_raw.npz'):
                np.savez(dest/name, obs_id=[2], sample_id=['b'], value=[20])
        with patch.object(t.extract, 'extract_raw', side_effect=extract_missing) as call:
            result = t.ensure_raw_cache(table, cache, root/'output', 'cpu', Path('unused'))
            assert call.call_count == 1
        for name in ('frontend_raw.npz', 'm3_raw.npz'):
            z = t.load_npz(result/name)
            np.testing.assert_array_equal(z['obs_id'], [3, 1, 2])
            np.testing.assert_array_equal(z['value'], [30, 10, 20])
    return dict(only_missing_raw_rows_extracted=True, requested_order_preserved=True)


def cohort_checks(oss1, oss2):
    """Explicitly requested cohort audit, not inference or use of labels for tuning."""
    report = {}
    for name, source_root, target_root in (('oss1_to_oss2', oss1, oss2), ('oss2_to_oss1', oss2, oss1)):
        path = oss2/f'cross_{name}_table.csv'
        source, target = t.read_table(source_root/'stack_table.csv'), t.read_table(path)
        views = pd.read_csv(target_root/'view_inventory.csv')
        selected, reasons = t.select_target(source, target, views)
        np.testing.assert_array_equal(selected.obs_id, target.obs_id)
        classes = pd.read_csv(path, usecols=['class_id', 'original_class_id'])
        expected = classes.original_class_id.map({0: 0, 2: 1, 3: 2, 4: 3})
        np.testing.assert_array_equal(classes.class_id, expected)
        assert set(classes.class_id) == set(range(4))
        report[name] = dict(n=len(target), vessels=target.mmsi.nunique(),
            physical_passes=len(np.unique(t.canonical_pass(target))),
            class_mapping_correct=True, all_rows_usable_and_nonoverlapping=True,
            excluded_by_independent_check=int((~reasons.included).sum()),
            source_supervised_rows=int(source.role.isin(t.SUPERVISED).sum()))
    return report


def checkpoint_checks(root, m3root):
    source = t.read_table(m3root/'table.csv', labels=True)
    # Cover roles and both-present / missing-polarization conditions, when available.
    mraw = t.aligned(root/'frontend_cache'/'m3_raw.npz', source)
    ids = list(source.groupby('role', sort=False).head(1).index)
    available = mraw['mass'].sum(-1) > 0
    missing = np.flatnonzero(~available.all(1))
    if len(missing):
        ids.append(int(missing[0]))
    ids = np.unique(ids)
    subset = source.iloc[ids].reset_index(drop=True)
    raw = t.aligned(root/'frontend_cache'/'frontend_raw.npz', subset)
    summaries = t.aligned(root/'frontend'/'summaries.npz', subset)
    query = t.project_raw({k: v[ids] for k, v in mraw.items()}, t.load_npz(m3root/'projection.npz'))
    bank, bank_rows, covariance = t.source_bank(m3root, source)
    refs = t.retrieve(query, bank, subset.mmsi, source.iloc[bank_rows].mmsi)
    expected_refs = t.load_npz(m3root/'references.npz')['refs'][ids]
    np.testing.assert_array_equal(bank_rows[refs], expected_refs)
    report = dict(obs_id=subset.obs_id.tolist(), n=len(subset), refs_exact=True, seeds={})
    # Any accidental invocation of imported training helpers must fail this check.
    with ExitStack() as stack:
        for module, names in ((t.front, ('scaled', 'fit_reader', 'fit_base', 'fit_heads', 'train_head')),
                              (t.m3, ('scale_fit', 'prepare_local', 'fit_head', 'fit_metric', 'fit_gate', 'choose_alpha'))):
            for name in names:
                stack.enter_context(patch.object(module, name, side_effect=AssertionError(f'Fit called: {name}')))
        for seed in (17, 29, 43):
            front = t.frontend_predict(raw, summaries, root/'frontend'/'models', seed)
            expected_front = t.aligned(root/'frontend'/f'scores_seed{seed}.npz', subset)
            expected_front['frontend_base'] = expected_front['base']
            checks = t.compare_replay(front, expected_front, ('frontend_base', 'm1', 's12'))
            actual = t.m3_predict(query, bank, refs, covariance, front['s12'].astype(np.float32),
                                  m3root/f'seed{seed}')
            expected = {k: v[ids] for k, v in t.load_npz(m3root/f'seed{seed}'/'predictions.npz').items()}
            checks.update(t.compare_replay(actual, expected,
                (*t.ARMS, 'base', 'delta_vv', 'delta_vh', 'local', 'gate_raw', 'available',
                 'weight_B_fixed', 'weight_spatial', 'weight_frequency')))
            # Missing VH must keep the source B anchor and zero its unavailable weight.
            for arm in ('spatial', 'frequency'):
                single = ~actual['available'].astype(bool).all(1)
                np.testing.assert_array_equal(actual[f'weight_{arm}'][single], actual['weight_B_fixed'][single])
            report['seeds'][str(seed)] = checks
    report['fit_helpers_forbidden'] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=t.CODE)
    parser.add_argument('--source-m3', type=Path, default=t.CODE/'m3_results_grid01')
    parser.add_argument('--check-cohorts', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = t.HERE/'core_validation'
    out.mkdir(exist_ok=True)
    checks = dict(identity=identity_checks(out), frozen_tau=frozen_tau_check(out),
                  cache_merge=cache_merge_check(out),
                  checkpoints=checkpoint_checks(args.source_root, args.source_m3))
    if args.check_cohorts:
        checks['cohorts'] = cohort_checks(t.CODE, t.HERE.parent/'oss2_m123_replication_20261026')
    t.write_json(out/'checks.json', dict(passed=True, scope='bounded CPU source-cache contract checks',
                 full_raw_source_replay_run=False, cross_dataset_run=False, checks=checks))
    print('PASS: identity, target-label barrier, frozen tau, bounded source checkpoint replay (3 seeds).')


if __name__ == '__main__':
    main()
