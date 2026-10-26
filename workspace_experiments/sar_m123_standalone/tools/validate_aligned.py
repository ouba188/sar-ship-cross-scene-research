"""Offline parity: full frozen feature cohorts and freshly extracted native chips.

Historical files supply expected values only. Inference imports no experiment
code and receives only exported assets plus new query features or native chips.
Reports preserve failures and the actual sample IDs, without altering weights.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from sar_m123.aligned import AlignedClassifier, FEATURES, load_npz, project_raw, raw_features, summary_features

SEEDS = (17, 29, 43)


def selected(path, table, keys):
    with np.load(path, allow_pickle=False) as z:
        ix = pd.Index(z['obs_id']).get_indexer(table.obs_id)
        if (ix < 0).any():
            raise ValueError(f'Missing observation IDs: {path}')
        return {k: z[k][ix] for k in keys}


def features(raw_path, summary_path, table, projection):
    raw = selected(raw_path, table, ('layer1', 'joint_raw', 'mass', 'e_image', 'coverage', 'valid'))
    summaries = selected(summary_path, table, ('summary1', 'quality', 'geometry'))
    return dict(raw=raw_features(raw, raw['valid']), aux=summary_features(summaries),
                local=project_raw(raw, projection), valid=raw['valid'], mmsi=table.mmsi.to_numpy(str))


def take(features, indices):
    return {k: {n: v[indices] for n, v in value.items()} if isinstance(value, dict) else value[indices]
            for k, value in features.items()}


def comparison(actual, expected, atol=2e-4, rtol=2e-5):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return dict(passed=False, actual_shape=list(actual.shape), expected_shape=list(expected.shape))
    error = abs(actual.astype(float)-expected.astype(float))
    finite = np.isfinite(actual).all() and np.isfinite(expected).all()
    return dict(passed=bool(finite and np.all(error <= atol+rtol*abs(expected))),
                max_abs=float(error.max(initial=0)), mean_abs=float(error.mean()), atol=atol, rtol=rtol)


def score_checks(scores, details, expected, source_ids):
    pairs = dict(final=(scores, expected['frequency']), bare=(details['base'], expected['bare']))
    match = details['matching']
    for key in ('local', 'delta_vv', 'delta_vh', 'gate_raw', 'available'):
        pairs[key] = (match[key], expected[key])
    pairs['weights'] = (match['polar_weights'], expected['weight_frequency'])
    checks = {key: comparison(*pair) for key, pair in pairs.items()}
    checks['references'] = comparison(source_ids, expected['refs'], atol=0, rtol=0)
    checks['predictions'] = comparison(np.asarray(scores).argmax(-1), expected['frequency'].argmax(-1), atol=0, rtol=0)
    return checks


def native_record(row, inventory):
    record = dict(id=str(row.obs_id), mmsi=str(row.mmsi))
    for pol in ('vv', 'vh'):
        view = inventory.loc[(row.obs_id, pol)]
        if not bool(view['exists']):
            record[pol] = None
            continue
        detections, axis = [], None
        if bool(view.detected) and float(view.confidence) >= .25:
            cx, cy, length, width, angle = json.loads(view.raw_xywhr)
            uv = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])*[length/2, width/2]
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            corners = uv@rotation.T+[cx, cy]
            detections = [dict(corners=corners.tolist(), confidence=float(view.confidence))]
            if bool(view.axis_valid):
                axis = dict(center=[float(view.cx), float(view.cy)], length_px=float(view.length),
                            width_px=float(view.width), angle=float(view.ship_axis))
        record[pol] = dict(image=str(view.path), detections=detections,
                           target_index=0 if detections else None, ship_axis=axis)
    return record


def choose_native(table, cached, predictions, per_class):
    """Fixed class sampling plus available-view extremes and changed decisions."""
    reasons = {}
    def add(index, reason):
        reasons.setdefault(int(index), []).append(reason)
    rng = np.random.default_rng(20261026)
    for cls in range(4):
        ids = np.flatnonzero(table.class_id.to_numpy() == cls)
        for index in rng.choice(ids, min(per_class, len(ids)), replace=False):
            add(index, f'class_{cls}_seed20261026')
    for p in range(2):
        ids = np.flatnonzero(cached['valid'][:, p] & ~cached['valid'][:, 1-p])
        if len(ids):
            add(ids[0], f'only_polarization_{p}')
    area = cached['local']['mass'].sum((1, 2))
    add(area.argmax(), 'largest_token_mass')
    add(area.argmin(), 'smallest_token_mass')
    y = table.class_id.to_numpy()
    for seed, pred in predictions.items():
        a, b = pred['old'].argmax(1) == y, pred['frequency'].argmax(1) == y
        for name, mask in (('rescued', b & ~a), ('harmed', a & ~b), ('always_correct', a & b), ('always_wrong', ~a & ~b)):
            ids = np.flatnonzero(mask)
            if len(ids):
                add(ids[0], f'{seed}_{name}')
    return reasons


def run(args):
    start = time.perf_counter()
    report = dict(device=args.device, seeds=list(SEEDS), cache_checks=[], native_checks=[],
                  native_sampling_seed=20261026, source_code_imported=False,
                  score_tolerance=dict(atol=2e-4, rtol=2e-5), status='running')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report['seconds'] = time.perf_counter()-start
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        for source in args.sources:
            target = 'oss2' if source == 'oss1' else 'oss1'
            direction = f'{source}-to-{target}'
            crossroot = args.legacy_root/'public_m123_cross_dataset_20261026'/direction
            protocol = json.loads((crossroot/'protocol.json').read_text(encoding='utf-8'))
            net = AlignedClassifier(args.assets_root/source, args.device)
            for side in ('source', 'cross'):
                source_root = args.legacy_root/f'{source}_m123_replication_20261026'
                table_path = source_root/'stack_table.csv' if side == 'source' else crossroot/'target_table.csv'
                table = pd.read_csv(table_path, dtype={'mmsi': str, 'sample_id': str})
                rawroot = source_root/'frontend_cache' if side == 'source' else Path(protocol['target_cache'])
                sums = source_root/'frontend/summaries.npz' if side == 'source' else crossroot/'summaries/summaries.npz'
                cached = features(rawroot/'m3_raw.npz', sums, table, net.m3.projection)
                expected = {}
                for seed in SEEDS:
                    pred = load_npz(args.legacy_root/'m3_aligned_training_20261026'/direction/f'seed{seed}/{side}_predictions.npz')
                    np.testing.assert_array_equal(pred['obs_id'], table.obs_id)
                    expected[seed] = pred
                    results, detail = [], []
                    for first in range(0, len(table), args.batch):
                        score, info = net.score(take(cached, slice(first, first+args.batch)), seed)
                        results.append(score)
                        detail.append(info)
                    scores = np.concatenate(results)
                    details = dict(base=np.concatenate([v['base'] for v in detail]),
                                   matching={k: np.concatenate([v['matching'][k] for v in detail]) for k in detail[0]['matching']})
                    checks = score_checks(scores, details, pred, details['matching']['reference_ids'])
                    result = dict(source=source, side=side, seed=seed, n=len(table), checks=checks,
                                  prediction_disagreements=int((scores.argmax(1) != pred['frequency'].argmax(1)).sum()),
                                  passed=all(c['passed'] for c in checks.values()))
                    report['cache_checks'].append(result)
                    print('CACHE', source, side, seed, result['passed'], checks['final'], flush=True)
                    save()
                if side == 'cross' and args.native_per_class:
                    inventory = pd.read_csv(protocol['view_inventory']).set_index(['obs_id', 'pol'])
                    selection = choose_native(table, cached, expected, args.native_per_class)
                    for index, why in selection.items():
                        row = table.iloc[index]
                        request = native_record(row, inventory)
                        fresh = net.extract(request)
                        checks = {k: comparison(fresh[k], cached[k][index]) for k in ('raw', 'aux', 'valid')}
                        checks.update({k: comparison(fresh['local'][k], cached['local'][k][index]) for k in FEATURES})
                        item = dict(source=source, obs_id=int(row.obs_id), class_id=int(row.class_id),
                                    sample_id=str(row.sample_id), reasons=why, request=request, feature_checks=checks,
                                    placement=fresh['placement'], seeds={})
                        for seed in SEEDS:
                            scores, details = net.score(fresh, seed)
                            pred = {k: v[index] for k, v in expected[seed].items()}
                            test = score_checks(scores, details, pred, details['matching']['reference_ids'])
                            item['seeds'][str(seed)] = dict(checks=test, actual=scores.tolist(), expected=pred['frequency'].tolist(),
                                                           prediction=int(scores.argmax()), expected_prediction=int(pred['frequency'].argmax()))
                        item['passed'] = all(c['passed'] for c in checks.values()) and all(
                            c['passed'] for result in item['seeds'].values() for c in result['checks'].values())
                        report['native_checks'].append(item)
                        print('NATIVE', source, row.obs_id, item['passed'], flush=True)
                        save()
            del net
        report['status'] = 'passed' if all(v['passed'] for v in report['cache_checks']+report['native_checks']) else 'failed'
    except Exception as exc:
        report['status'] = 'error'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
    print('REPORT', args.report, report['status'], round(report['seconds'], 1), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--legacy-root', required=True, type=Path)
    parser.add_argument('--assets-root', type=Path, default=root/'assets_aligned')
    parser.add_argument('--report', type=Path, default=root/'validation_reports/aligned_parity.json')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sources', nargs='+', choices=('oss1', 'oss2'), default=('oss1', 'oss2'))
    parser.add_argument('--batch', type=int, default=48)
    parser.add_argument('--native-per-class', type=int, default=2)
    args = parser.parse_args()
    if args.batch < 1 or args.native_per_class < 0:
        parser.error('batch must be positive and native-per-class nonnegative')
    result = run(args)
    raise SystemExit(0 if result['status'] == 'passed' else 1)
