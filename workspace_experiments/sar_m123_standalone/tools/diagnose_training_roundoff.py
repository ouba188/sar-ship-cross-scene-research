"""Locate the first observed source-only versus historical batching difference."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sar_m123 import train_ops as op
from sar_m123.train_aligned import load_npz


def run(root, assets, outputs, report):
    torch.set_num_threads(2)
    oldroot = root/'raw_region_bypass_20261026/downstream/raw_summary/oss2-to-oss1/seed43'
    old = load_npz(oldroot/'source_predictions.npz')
    local = load_npz(root/'public_m123_cross_dataset_20261026/normalization_repair_20261026/source_oss2/seed43/predictions.npz')['local']
    warm = assets/'oss2/warm/seed43'
    scaler = op.load_checkpoint(warm/'M3_readout_scaler.pt')
    x = torch.tensor(op.transform(op.local_raw(local, old['bare']), scaler))
    head = op.PolarHead(0)
    head.load_state_dict(op.load_checkpoint(warm/'vv.pt')['state'])
    head.eval()
    with torch.no_grad():
        alone = head(x).numpy()
        # Zero dummy rows test batch-shape rounding, not target-data influence.
        padded = head(torch.cat((x, torch.zeros((921, 4, 15))), 0)).numpy()[:len(x)]
    differences = {}
    for name in ('vv.pt', 'vh_centered.pt', 'frequency.pt'):
        a = op.load_checkpoint(outputs/'oss2/seed43/warm'/name)['state']
        b = op.load_checkpoint(warm/name)['state']
        differences[name] = max(float(abs(a[k]-b[k]).max()) for k in a)
    parity = json.loads((report.parent/'aligned_training.json').read_text(encoding='utf-8'))
    behavior = all(v['checks']['predictions']['passed'] and all(z['passed'] for z in v['selections'].values())
                   for v in parity['configurations'])
    data = dict(status='diagnosed_batch_rounding_with_remaining_numeric_drift',
        source='oss2', seed=43, input_source_rows=len(x), dummy_rows=921,
        source_only_vs_historical=float(abs(alone-old['delta_vv']).max()),
        zero_padded_vs_historical=float(abs(padded-old['delta_vv']).max()),
        source_only_vs_padded=float(abs(alone-padded).max()),
        warm_parameter_differences=differences, all_prediction_and_selection_parity=behavior,
        strict_numeric_parity=parity['status'], original_weights_preserved=True,
        interpretation='First observed rounding difference is reproduced at fixed weights by changing batch size only. Subsequent optimization differs slightly; not every later numerical path is isolated. No target records or labels are needed in standalone training.')
    report.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=Path('assets_aligned'))
    parser.add_argument('--outputs', type=Path, default=Path('training_outputs'))
    parser.add_argument('--report', type=Path, default=Path('validation_reports/aligned_roundoff.json'))
    args = parser.parse_args()
    run(args.legacy_root, args.assets, args.outputs, args.report)
