"""Read-only OSS1 preparation. Writes metadata only beside this script; no torch."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SIDE = ROOT / 'same_vessel_gallery_20261019/causal_frequency_side_20261024'
TRANSFER = Path('E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007')


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    frame = pd.read_csv(TRANSFER / 'target_manifest.csv', dtype={'mmsi_or_ship_id': str})
    cohort = pd.read_csv(SIDE / 'round23_public_oss1/cohort.csv', dtype={'mmsi': str})
    geometry = pd.read_csv(ROOT / 'oss1_joint_geometry_20261017/geometry.csv')
    assert frame.sample_id.equals(cohort.sample_id)
    assert frame.sample_id.equals(geometry.sample_id)
    assert frame.sample_id.is_unique
    table = frame.copy()
    table['obs_id'] = cohort.obs_id
    table['mmsi'] = cohort.mmsi
    table['overpass'] = cohort.overpass
    table['outer_fold'] = cohort.fold
    table['round23_region_eligible'] = cohort.intervention_eligible
    r26 = pd.read_csv(SIDE / 'round26_polarization_coverage/cohort.csv')
    assert table.sample_id.equals(r26.sample_id)
    table['round26_region_eligible'] = r26.eligible
    detections = pd.DataFrame(json.loads((ROOT / 'oss1_joint_geometry_20261017/detection_views.json').read_text(encoding='utf-8')))
    views = []
    for pol in ('vv', 'vh'):
        d = detections[detections.pol.eq(pol)].set_index('global_row').loc[table.obs_id]
        table[f'obb_{pol}'] = (d.detected & d.confidence.ge(.25)).to_numpy()
        table[f'axis_{pol}'] = (d.axis_valid & d.confidence.ge(.25)).to_numpy()
        for row, det in zip(table.to_dict('records'), d.to_dict('records')):
            path = Path(row['chip_path_' + pol])
            record = dict(obs_id=row['obs_id'], sample_id=row['sample_id'],
                          path=str(path), exists=path.is_file(), **det)
            if record['exists']:
                with Image.open(path) as im:
                    record.update(actual_w=im.width, actual_h=im.height, mode=im.mode)
                assert (record['actual_h'], record['actual_w']) == (det['native_h'], det['native_w'])
            views.append(record)
    for key in ('raw_patch_path', 'calibrated_patch_path', 'ship_xml_path', 'metadata_xml_path'):
        table[key] = geometry[key]
        table[key + '_exists_now'] = geometry[key].map(lambda x: Path(x).is_file())
    table['any_obb'] = table.obb_vv | table.obb_vh
    table['both_obb'] = table.obb_vv & table.obb_vh
    table.to_csv(OUT / 'data_inventory.csv', index=False)
    pd.DataFrame(views).to_csv(OUT / 'view_inventory.csv', index=False)
    counts = table.groupby(['class_id', 'class_name']).agg(
        observations=('obs_id', 'size'), vessels=('mmsi', 'nunique'), passes=('overpass', 'nunique'),
        any_obb=('any_obb', 'sum'), both_obb=('both_obb', 'sum'),
        round23_regions=('round23_region_eligible', 'sum'), round26_regions=('round26_region_eligible', 'sum'))
    counts.to_csv(OUT / 'class_coverage.csv')
    split = pd.read_csv(SIDE / 'round23_public_oss1/splits.csv')
    roles, stats, candidates = [], [], []
    for fold in range(3):
        part = split[split.fold.eq(fold)]
        source = table[table.obs_id.isin(part[part.partition.eq('train')].obs_id)]
        test = table[table.obs_id.isin(part[part.partition.eq('test')].obs_id)]
        assert not set(source.mmsi) & set(test.mmsi)
        assert not set(source.overpass) & set(test.overpass)
        assert set(source.class_id) == set(range(8))
        for role, group in [('source_pool', source), ('test', test)]:
            records = group[['obs_id', 'sample_id', 'mmsi', 'overpass', 'class_id']].copy()
            records.insert(0, 'role', role)
            records.insert(0, 'fold', fold)
            roles.append(records)
            for cid, g in group.groupby('class_id'):
                stats.append(dict(fold=fold, role=role, class_id=int(cid), n=len(g),
                                  vessels=g.mmsi.nunique(), passes=g.overpass.nunique(),
                                  obb_vessels=g[g.any_obb].mmsi.nunique()))
        # These are feasibility candidates, not selected validation folds.
        for pas in sorted(source.overpass.unique()):
            val = source[source.overpass.eq(pas)]
            fit = source[~source.overpass.eq(pas) & ~source.mmsi.isin(val.mmsi)]
            candidates.append(dict(fold=fold, validation_pass=pas, fit_n=len(fit), validation_n=len(val),
                                   fit_classes=','.join(map(str, sorted(fit.class_id.unique()))),
                                   validation_classes=','.join(map(str, sorted(val.class_id.unique()))),
                                   fit_all8=set(fit.class_id) == set(range(8)),
                                   validation_all8=set(val.class_id) == set(range(8))))
    pd.concat(roles, ignore_index=True).to_csv(OUT / 'outer_roles.csv', index=False)
    pd.DataFrame(stats).to_csv(OUT / 'fold_class_coverage.csv', index=False)
    pd.DataFrame(candidates).to_csv(OUT / 'inner_pass_candidates.csv', index=False)
    cache_paths = [SIDE / 'round23_public_oss1/native_features.npy',
                   SIDE / 'round26_polarization_coverage/private_feature_changes.npy',
                   TRANSFER / 'features.npz']
    cache_paths += sorted((ROOT / 'oss1_joint_geometry_20261017').glob('*/local_features.npz'))
    caches = []
    for path in cache_paths:
        item = dict(path=str(path), bytes=path.stat().st_size)
        if path.suffix == '.npy':
            arr = np.load(path, mmap_mode='r', allow_pickle=False)
            item['shape'], item['dtype'] = list(arr.shape), str(arr.dtype)
        else:
            with np.load(path, allow_pickle=False) as archive:
                item['arrays'] = {key: list(archive[key].shape) for key in archive.files}
        caches.append(item)
    save_json('cache_inventory.json', caches)
    legacy, legacy_counts = [], []
    strict = ROOT / 'scene_cross_dataset_redesign_20261014'
    for path in sorted(strict.glob('**/source_train.csv')):
        if 'oss1' not in path.parts:
            continue
        val_path = path.with_name('source_validation.csv')
        if not val_path.is_file():
            continue
        train = pd.read_csv(path, dtype={'mmsi': str})
        val = pd.read_csv(val_path, dtype={'mmsi': str})
        pilot = path.parent.name == 'oss1' and path.parent.parent.name == 'strict_training'
        query_path = strict / 'public_split/oss1_query.csv' if pilot else path.with_name('query.csv')
        support_path = strict / 'public_split/oss1_support_candidates.csv' if pilot else path.with_name('support_candidates.csv')
        held_paths = [p for p in (query_path, support_path) if p.is_file()]
        held = pd.concat([pd.read_csv(p, dtype={'mmsi': str}) for p in held_paths]) if held_paths else None
        source = pd.concat([train, val])
        identity_overlap = len(set(source.mmsi) & set(held.mmsi)) if held is not None else None
        pass_overlap = len(set(source.acquisition) & set(held.acquisition)) if held is not None else None
        assert identity_overlap in (None, 0) and pass_overlap in (None, 0)
        for role, group in [('fit_train', train), ('fit_validation', val)]:
            joined = group[['sample_id']].merge(table, on='sample_id', validate='one_to_one')
            assert len(joined) == len(group)
            for cid, g in joined.groupby('class_id'):
                legacy_counts.append(dict(train_path=str(path), role=role, class_id=int(cid), n=len(g),
                                          vessels=g.mmsi.nunique(), obb_vessels=g[g.any_obb].mmsi.nunique()))
        legacy.append(dict(train_path=str(path), validation_path=str(val_path), train_n=len(train),
                           validation_n=len(val), train_classes=sorted(map(int, train.class_id.unique())),
                           validation_classes=sorted(map(int, val.class_id.unique())),
                           identity_overlap=len(set(train.mmsi) & set(val.mmsi)),
                           pass_overlap=len(set(train.acquisition) & set(val.acquisition)),
                           held_paths=[str(p) for p in held_paths], held_identity_overlap=identity_overlap,
                           held_pass_overlap=pass_overlap, held_paths_complete=len(held_paths) == 2))
    save_json('legacy_source_validation.json', legacy)
    pd.DataFrame(legacy_counts).to_csv(OUT / 'legacy_fold_class_coverage.csv', index=False)
    receipt = dict(status='metadata_preparation_only', observations=len(table), vessels=table.mmsi.nunique(),
                   passes=table.overpass.nunique(), any_obb=int(table.any_obb.sum()),
                   both_obb=int(table.both_obb.sum()), missing_chip_files=sum(not r['exists'] for r in views),
                   existing_raw=int(table.raw_patch_path_exists_now.sum()),
                   existing_calibrated=int(table.calibrated_patch_path_exists_now.sum()),
                   existing_ship_xml=int(table.ship_xml_path_exists_now.sum()),
                   existing_metadata_xml=int(table.metadata_xml_path_exists_now.sum()),
                   training_run=False, model_inference_run=False, gpu_used=False,
                   git_used=False, inner_split_frozen=False, exact_eight_class_m3_bank_feasible=False)
    save_json('PREPARATION_RECEIPT.json', receipt)
    print(json.dumps(receipt, indent=2))
    print(counts.to_string())


if __name__ == '__main__':
    main()
