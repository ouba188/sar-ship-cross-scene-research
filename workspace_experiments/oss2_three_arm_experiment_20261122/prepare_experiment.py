"""Build the fixed OSS2 development experiment from independently audited records.

Private manifests, masks, and individual review ledgers stay local. This does
not certify sensor nodata masks or whole-background causal independence.
"""
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
CODE = Path('E:/临时会话/port_prototype_model_20261121')
sys.path.insert(0, str(CODE))
from sar_port.two_stage.data import InputConfig, _axis
from sar_port.two_stage.geometry import GeometryConfig
from sar_port.two_stage.cache import prepare_record_cached
from sar_port.two_stage.cli import protocol, SupplementalInterventions


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def csvrows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8', newline='\n')


def main():
    torch.set_num_threads(4)
    os.environ['SAR_PORT_PREPARE_CACHE'] = str(ROOT/'prepared_cache_local')
    config = read(CODE/'configs/two_stage_native.json')
    input_config = InputConfig(**config['input'])
    geometry_config = GeometryConfig(**config['geometry'])
    audit = WORK/'data_qualification_audit_20261122'
    rows = read(audit/'oss2_alignment/manifest_alignment_restored_local.json')
    rows += read(ROOT/'data_eval_qualification/val_alignment_restored_local.json')
    rows += read(ROOT/'data_eval_qualification/test_alignment_restored_local.json')
    assignment = {r['sample_id']: r for r in csvrows(audit/'oss2_development_split/split_assignment_local.csv')}
    source = {r['sample_id']: r for r in csvrows(Path('E:/临时会话/oss2_port_prototypes_20261121/source_observations_local.csv'))}
    candidates = csvrows(ROOT/'q_expansion/selected_candidates_local.csv')
    reviewed = [r for r in candidates if r['coast_status'] == 'offshore']
    assert len(reviewed) == 20
    by_class = defaultdict(list)
    for r in reviewed:
        by_class[r['class_name']].append(r)
    classes = sorted(by_class)
    donors = [by_class[c][k]['sample_id'] for k in range(2) for c in classes]
    queries = [by_class[c][k]['sample_id'] for k in (2, 3) for c in classes]
    reviewed = {r['sample_id']: r for r in reviewed}
    masks = ROOT/'masks_local'
    masks.mkdir(exist_ok=True)
    ledger = []
    for row in rows:
        sid = row['id']; split = assignment[sid]
        row.update(label=str(row['label']), vessel_id=str(split['mmsi']),
                   group_id=split['absolute_orbit_key'], development_split=split['development_split'],
                   port=source.get(sid, {}).get('port_proxy') or 'unknown',
                   leakage_controlled=False, leakage_regularization_eligible=sid in queries,
                   training_qualified=False, empirical_experiment_eligible=False,
                   sensor_valid_mask_status='unknown',
                   availability_mask_semantics='stored_native_pixel_footprint_not_sensor_nodata',
                   historical_test_exposure=True)
        if sid in queries:
            row['background_donor_ids'] = donors
        h, w = row['native_size']
        footprint = masks/f'footprint_{h}_{w}.png'
        if not footprint.exists():
            Image.fromarray(np.full((h, w), 255, np.uint8)).save(footprint)
        for p in ('vv', 'vh'):
            assert row[p]['alignment_valid']
            if not row[p].get('ship_axis') and row[p].get('target_index') is None:
                other = 'vh' if p == 'vv' else 'vv'
                axis = _axis(row[other])
                row[p]['ship_axis'] = dict(center=axis[:2].tolist(), length_px=float(axis[2]),
                                          width_px=float(axis[3]), angle=float(axis[4]))
                row[p]['geometry_source'] = 'paired_reference_original_detection_corners'
                row[p]['axis_reliable'] = bool(axis[2]/axis[3] >= 1.5)
            row[p].update(valid_mask=str(footprint), background_validated=False)
            row[p].pop('background_mask', None)
        if sid in reviewed:
            r = reviewed[sid]
            assert split['development_split'] == 'train'
            q = np.zeros((h, w), np.uint8)
            left, top = int(r['crop_left']), int(r['crop_top'])
            for x0, y0, x1, y1 in ((4, 52, 12, 60), (52, 52, 60, 60)):
                q[top+y0:top+y1, left+x0:left+x1] = 255
            qpath = masks/f'{sid}_q.png'
            Image.fromarray(q).save(qpath)
            for p in ('vv', 'vh'):
                row[p].update(background_mask=str(qpath), background_validated=True)
            row['q_review'] = dict(case=r['review_case'], scope='two_local_bottom_corner_regions_only',
                                  visual_review='both_polarizations_reviewed_in_six_pages',
                                  water_context='center_1500m_WorldCover_support_not_pixel_georegistration',
                                  raw_geometry_chain='joint_TIFF_Uint8_PNG_exact',
                                  strong_background_independence=False)
            ledger.append(dict(id=sid, case=r['review_case'], class_name=r['class_name'],
                               role='donor' if sid in donors else 'query' if sid in queries else 'scene_only',
                               reviewed_pixels_per_polarization=128, background_validated=True,
                               leakage_regularization_eligible=sid in queries, leakage_controlled=False))
    kept, excluded, samples, detail = [], [], {}, []
    for row in rows:
        row['empirical_experiment_eligible'] = True
        try:
            sample = prepare_record_cached(row, ROOT, input_config, geometry_config)
        except (ValueError, KeyError) as exc:
            excluded.append(dict(id=row['id'], split=row['development_split'], reason=str(exc)))
            continue
        row['empirical_experiment_eligible'] = True
        # Keep the old strong training_qualified field false. Qualification for
        # this explicitly weaker development protocol is separately recorded.
        i = len(kept); kept.append(row); samples[i] = sample
        qcells = [int((torch.nn.functional.avg_pool2d(sample.background, s, s) >= 1-1e-6).sum()) for s in (4, 8, 16)]
        if row['id'] in reviewed:
            assert sample.geometry.canvas_size == 64 and sample.pair_eligible
            assert sample.background.flatten(1).sum(1).tolist() == [128., 128.]
            expected = torch.zeros_like(sample.background)
            expected[:, :, 52:60, 4:12] = 1
            expected[:, :, 52:60, 52:60] = 1
            assert torch.equal(sample.background, expected)
        detail.append(dict(id=row['id'], split=row['development_split'], label=row['label'],
                           canvas=sample.geometry.canvas_size, pair_eligible=sample.pair_eligible,
                           q_pixels=int(sample.background.sum()), pure_q_cells_by_scale=qcells))
        if len(kept) % 100 == 0:
            print('prepared',len(kept),'excluded',len(excluded),flush=True)
    config.pop('target_ports'); config.pop('validation_ports')
    config.update(protocol='development_split', selection_metric='balanced_accuracy',
                  leakage_mode='empirical_regularization', donor_ids=donors,
                  supplemental_batches=[queries.copy() for _ in range(4)])
    config['encoder']['weights_path'] = str(WORK/'model_weights/convnext_tiny-983f1562.pth')
    config['h'].update(batched_matching=True, pair_batch_size=64)
    split = protocol(kept, config)
    SupplementalInterventions(samples, kept, split['train'].tolist(), config)
    for seed in (17, 29, 43):
        for arm in ('baseline', 'disentangle'):
            cfg = json.loads(json.dumps(config)); cfg['seed'] = seed
            cfg['encoder']['gate_enabled'] = arm == 'disentangle'
            write(ROOT/f'configs/{arm}_seed{seed}.json', cfg)
    write(ROOT/'experiment_manifest_local.json', kept)
    write(ROOT/'qualification_ledger_local.json', ledger)
    write(ROOT/'geometry_preflight_local.json', detail)
    write(ROOT/'excluded_geometry_local.json', excluded)
    summary = dict(input_records=len(rows), retained=len(kept), excluded=excluded,
                   split_counts={k:len(v) for k,v in split.items() if k in ('train','validation','test')},
                   split_class_counts={s:dict(Counter(r['class_name'] for r in kept if r['development_split']==s)) for s in ('train','val','test')},
                   canvas_counts=dict(Counter(r['canvas'] for r in detail)),
                   reviewed_q_observations=len(ledger), independent_donors=len(donors), empirical_queries=len(queries),
                   supplemental_query_exposures_per_epoch=32, strong_controlled_samples=0,
                   pure_q_cells_by_scale=np.array([r['pure_q_cells_by_scale'] for r in detail]).sum(0).tolist(),
                   sensor_nodata_masks_recovered=False, availability_convention='native stored pixel footprint; no inferred sensor nodata recovery',
                   evaluation_scope='OSS2 development; historical test exposed; port values are proxies',
                   seeds=[17,29,43], stage1_epochs=20, base_epochs=10, h_epochs=20,
                   selection_metric='validation_balanced_accuracy', test_selection=False)
    write(ROOT/'preflight_summary.json', summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
