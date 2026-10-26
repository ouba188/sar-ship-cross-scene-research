"""Exercise chip-manifest preparation with the real source bank and all roles."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sar_m123.prepare_training import prepare_bundle
from sar_m123.train_aligned import load_npz, FEATURES
from validate_aligned import native_record


def run(root, assets, bundle, output, report):
    source = json.loads((assets/'manifest.json').read_text(encoding='utf-8'))['source']
    source_root = root/f'{source}_m123_replication_20261026'
    table = pd.read_csv(source_root/'stack_table.csv', dtype={'mmsi': str, 'pass_group': str})
    inventory = pd.read_csv(source_root/'view_inventory.csv').set_index(['obs_id', 'pol'])
    bank = load_npz(assets/'m3/bank.npz')
    rows = list(bank['source_index'])
    for role in ('fit_validation', 'utility_train', 'selection', 'evaluation'):
        for c in range(4):
            rows.append(int(table.index[(table.role == role) & (table.class_id == c)][0]))
    records = []
    for i in rows:
        row = table.iloc[i]
        record = native_record(row, inventory)
        record.update(role=row.role, class_id=-1 if row.role == 'evaluation' else int(row.class_id),
                      pass_group=str(row.pass_group))
        records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.parent/'native_source_records.json'
    manifest.write_text(json.dumps(records), encoding='utf-8')
    prepare_bundle(manifest, assets, output, device='cuda')
    fresh, cached = load_npz(output/'features.npz'), load_npz(bundle/'features.npz')
    ix = pd.Index(cached['obs_id'].astype(str)).get_indexer(fresh['obs_id'].astype(str))
    checks = {}
    for key in ('raw', 'aux')+FEATURES:
        error = float(abs(fresh[key]-cached[key][ix]).max())
        np.testing.assert_allclose(fresh[key], cached[key][ix], rtol=2e-5, atol=2e-4)
        checks[key] = error
    np.testing.assert_array_equal(ix[fresh['refs']], cached['refs'][ix])
    receipt = dict(status='passed', source=source, rows=len(rows), source_bank=32,
                   heldout_role_counts={r:sum(v['role']==r for v in records) for r in
                       ('fit_train','fit_validation','utility_train','selection','evaluation')},
                   max_abs=checks, references_identical=True,
                   evaluation_labels_removed=True, training_run=False)
    report.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', required=True, type=Path)
    parser.add_argument('--assets', type=Path, default=Path('assets_aligned/oss1'))
    parser.add_argument('--bundle', type=Path, default=Path('validation_local/aligned_training/oss1'))
    parser.add_argument('--output', type=Path, default=Path('validation_local/aligned_native_bundle'))
    parser.add_argument('--report', type=Path, default=Path('validation_reports/aligned_native_training_input.json'))
    args = parser.parse_args()
    run(args.legacy_root, args.assets, args.bundle, args.output, args.report)
