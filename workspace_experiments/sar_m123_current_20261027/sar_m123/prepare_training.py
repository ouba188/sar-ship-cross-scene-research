"""Prepare source-only training features directly from chip/OBB JSON records.

The frozen pretrained source reference bank must be included in these records.
This retains the accepted experiment's training initialization and protocol.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .aligned import AlignedClassifier
from .train_aligned import FEATURES, validate_roles


def prepare_bundle(requests_path, assets, output, device='cpu'):
    requests_path, assets, output = Path(requests_path), Path(assets), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite a training bundle: {output}')
    records = json.loads(requests_path.read_text(encoding='utf-8-sig'))
    if not isinstance(records, list) or not records:
        raise ValueError('Expected a nonempty list of source chip records.')
    rows = []
    for r in records:
        if any(r[k] is None or not str(r[k]).strip() for k in ('id', 'mmsi', 'pass_group')):
            raise ValueError('Source preparation requires observation, vessel and acquisition identities.')
        row = dict(obs_id=str(r['id']), mmsi=str(r['mmsi']),
                   pass_group=str(r['pass_group']), role=r['role'])
        row['class_id'] = -1 if row['role'] == 'evaluation' else int(r['class_id'])
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.obs_id.is_unique:
        raise ValueError('Duplicate observation IDs in the source manifest.')
    model = AlignedClassifier(assets, device)
    bank = model.m3.bank
    lookup = {str(v): i for i, v in enumerate(table.obs_id)}
    missing = set(bank['obs_id'].astype(str))-set(lookup)
    if missing:
        raise ValueError(f'Manifest is missing {len(missing)} frozen source-bank observations.')
    bankrows = np.array([lookup[str(v)] for v in bank['obs_id']])
    raw, aux, locals_ = [], [], []
    for i, record in enumerate(records):
        result = model.extract(record, requests_path.parent)
        raw.append(result['raw'])
        aux.append(result['aux'])
        locals_.append(result['local'])
        if (i+1) % 50 == 0:
            print(f'Extracted {i+1}/{len(records)}', flush=True)
    features = {k: np.stack([x[k] for x in locals_]) for k in FEATURES}
    retrieved = model.m3.retrieve(features, table.mmsi.to_numpy(str))
    data = dict(obs_id=table.obs_id.to_numpy(str), raw=np.stack(raw), aux=np.stack(aux),
                bank=bankrows.reshape(4, 8), refs=bankrows[retrieved], **features)
    data.update({'covariance_'+k: v for k, v in model.m3.covariance_scaler.items()})
    validate_roles(table, data)
    output.mkdir(parents=True)
    np.savez_compressed(output/'features.npz', **data)
    table.to_csv(output/'table.csv', index=False)
    (output/'provenance.json').write_text(json.dumps(dict(
        source=model.manifest['source'], raw_chip_extraction=True,
        rows=len(records), class_order=model.manifest['class_names'],
        frontend='frozen source-pretrained readers, PCA and bank',
        evaluation_labels_used=False), indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    prepare_bundle(args.input, args.assets, args.output, args.device)


if __name__ == '__main__':
    main()
