"""Create local-only raw-chip regression fixtures, never required at inference."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'validation_local')
    parser.add_argument('--ids', type=int, nargs='+',
                        default=[13372, 26043, 2418, 16596, 18767, 21151, 26694, 26069])
    args = parser.parse_args()
    old = args.workspace/'same_vessel_gallery_20261019'
    base = args.workspace/'scene_aware_classification_20261018/model_closeout_20261023'
    up = base/'regional_utility_pilot/upstream'
    tab = pd.read_csv(up/'table.csv').reset_index().set_index('obs_id')
    cohort = pd.read_csv(old/'conditional_evidence_pilot/cohort.csv').reset_index().set_index('obs_id')
    native = pd.read_csv(base/'evidence_preserving_redesign/native_validation/table.csv').reset_index().set_index('obs_id')
    valid = np.load(up/'valid.npz')['valid']
    disabled = pd.read_csv(old/'local_condition_module/crossview_alignment.csv').set_index('obs_id')['disable_new_vh']
    snapshots = {r['obs_id']: r for r in json.loads((old/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'chips').mkdir(exist_ok=True)
    records, up_idx, old_idx, selected_idx = [], [], [], []
    for oid in args.ids:
        row = tab.loc[oid]
        ui = int(row['index'])
        up_idx.append(ui)
        old_idx.append(int(cohort.loc[oid, 'index']))
        selected_idx.append(int(native.loc[oid, 'index']))
        item = dict(id=str(oid))
        for p, pol in enumerate(('vv', 'vh')):
            if p == 1 and bool(disabled.loc[oid]):
                item[pol] = None
                continue
            priors = snapshots[oid]['images'][pol]['priors']
            filename = f'{oid}_{pol}.png'
            shutil.copy2(row['image_'+pol], args.output/'chips'/filename)
            item[pol] = dict(image='chips/'+filename, target_index=priors['target_index'],
                detections=[{k:d[k] for k in ('corners', 'confidence', 'shape_usable')} for d in priors['detections']],
                ship_axis=priors.get('ship_axis'))
            if priors.get('radar_kind') == 'orbit_los' and priors.get('radar_axis') is not None:
                item[pol]['radar_axis_rad'] = priors['radar_axis']
        records.append(item)
    (args.output/'requests.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    expected = dict(obs_ids=np.array(args.ids), up_index=np.array(up_idx),
                    truth=tab.loc[args.ids].class_id.to_numpy(int))
    with np.load(old/'native_axis_probe_20261020/descriptors.npz') as z:
        expected['axis'] = np.concatenate((z['base'][old_idx], z['axis'][old_idx]), 1)
    expected['five'] = np.load(old/'sampling_preservation_20261020/keep_refs.npy', mmap_mode='r')[old_idx].reshape(len(up_idx), 60, 64)
    with np.load(old/'neighborhood_level_split_20261021/features.npz') as z:
        for key in ('centered', 'means', 'usable'):
            expected[key] = z[key][old_idx]
    with np.load(up/'summaries.npz') as z:
        for key in ('mean3', 'hb3', 'summary1', 'quality', 'geometry', 'valid'):
            expected[key] = z[key][up_idx]
    for seed in (17, 29, 43):
        with np.load(up/f'base_seed{seed}.npz') as z:
            expected[f'base_{seed}'] = z['B'][up_idx]
        with np.load(up/f'strategies_seed{seed}.npz') as z:
            expected[f's12_{seed}'] = z['logits'][up_idx, 2]
        with np.load(base/f'evidence_preserving_redesign/anchored_polar_adjustment_20261025/results/frequency_{seed}.npz') as z:
            expected[f'final_{seed}'] = z['score'][selected_idx]
            expected[f'weight_{seed}'] = z['weight'][selected_idx]
    np.savez_compressed(args.output/'expected.npz', **expected)
    print(json.dumps(dict(n=len(records), output=str(args.output))))


if __name__ == '__main__':
    main()
