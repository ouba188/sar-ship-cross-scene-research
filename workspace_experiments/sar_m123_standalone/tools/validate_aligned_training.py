"""Compare standalone retraining with the frozen accepted source predictions."""
import argparse
import json
from pathlib import Path

import numpy as np


def run(root, outputs, report):
    results = []
    for source, target in (('oss1', 'oss2'), ('oss2', 'oss1')):
        for seed in (17, 29, 43):
            current = outputs/source/f'seed{seed}'
            old = root/'m3_aligned_training_20261026'/f'{source}-to-{target}'/f'seed{seed}'
            with np.load(current/'predictions.npz', allow_pickle=False) as a, np.load(old/'source_predictions.npz', allow_pickle=False) as b:
                np.testing.assert_array_equal(a['obs_id'], b['obs_id'])
                checks = {}
                for key in ('bare', 'local', 'frequency', 'delta_vv', 'delta_vh', 'weight_frequency', 'gate_raw', 'refs'):
                    error = float(np.max(abs(a[key]-b[key])))
                    checks[key] = dict(max_abs=error, passed=bool(np.allclose(a[key], b[key], atol=2e-4, rtol=2e-5)))
                flips = int(np.sum(a['frequency'].argmax(1) != b['frequency'].argmax(1)))
                checks['predictions'] = dict(flips=flips, passed=flips == 0)
                n = len(a['obs_id'])
            aa, bb = (json.loads((p/'selection.json').read_text(encoding='utf-8')) for p in (current, old))
            keys = ('metric_step', 'vv_step', 'vh_step', 'alpha', 'frequency_step')
            selections = {k: dict(actual=aa[k], expected=bb[k], passed=aa[k] == bb[k]) for k in keys}
            results.append(dict(source=source, seed=seed, n=n, checks=checks, selections=selections,
                                passed=all(v['passed'] for v in checks.values()) and all(v['passed'] for v in selections.values())))
    data = dict(status='passed' if all(v['passed'] for v in results) else 'failed',
                configurations=results, target_labels_used=False,
                interpretation='engineering retraining parity, not new independent research evidence')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps(data, indent=2))
    if data['status'] != 'passed':
        raise SystemExit('Training parity failed; see report before installing outputs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', required=True, type=Path)
    parser.add_argument('--outputs', type=Path, default=Path('training_outputs'))
    parser.add_argument('--report', type=Path, default=Path('validation_reports/aligned_training.json'))
    args = parser.parse_args()
    run(args.legacy_root, args.outputs, args.report)
