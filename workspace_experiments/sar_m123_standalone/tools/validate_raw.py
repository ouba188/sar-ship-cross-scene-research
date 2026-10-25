"""Compare fresh raw-chip inference with historical regression fixtures."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from sar_m123.input import prepare
from sar_m123.backbone import Backbone
from sar_m123.baseline import Baseline, extract
from sar_m123.m12 import SummaryReader, M12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--fixture', type=Path, default=root/'validation_local')
    parser.add_argument('--assets', type=Path, default=root/'assets')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--upstream-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    backbone = Backbone(args.assets/'backbone/resnet50.pth', args.device)
    summary_reader = SummaryReader(args.assets)
    seeds = (17, 29, 43)
    baselines = {seed: Baseline(args.assets, seed) for seed in seeds}
    readers = {seed: M12(args.assets, seed) for seed in seeds}
    if not args.upstream_only:
        from sar_m123.m3 import M3
        matcher = M3(args.assets, device=args.device)
    requests = json.loads((args.fixture/'requests.json').read_text(encoding='utf-8'))
    expected = dict(np.load(args.fixture/'expected.npz'))
    records = []
    start = time.perf_counter()
    for i, request in enumerate(requests):
        prepared = prepare(request, args.fixture)
        maps = backbone(prepared)
        feature = extract(prepared, maps, backbone)
        target_prepared = [v if v is not None and v['valid_target'] else None for v in prepared]
        target_maps = [m if v is not None else None for v, m in zip(target_prepared, maps)]
        summaries = summary_reader.summarize(target_prepared, target_maps)
        if not args.upstream_only:
            local = matcher.extract(target_prepared, target_maps)
        result = dict(id=request['id'], errors={})
        for key in ('axis', 'five', 'centered', 'means', 'usable'):
            result['errors'][key] = float(abs(feature[key].astype(float)-expected[key][i]).max())
        for key in ('mean3', 'hb3', 'summary1', 'quality', 'geometry', 'valid'):
            result['errors'][key] = float(abs(summaries[key].astype(float)-expected[key][i]).max())
        for seed in seeds:
            base, _ = baselines[seed](feature)
            s12, _ = readers[seed](base, summaries)
            result['errors'][f'base_{seed}'] = float(abs(base-expected[f'base_{seed}'][i]).max())
            result['errors'][f's12_{seed}'] = float(abs(s12-expected[f's12_{seed}'][i]).max())
            if not args.upstream_only:
                final, details = matcher(local, s12, seed)
                result['errors'][f'final_{seed}'] = float(abs(final-expected[f'final_{seed}'][i]).max())
                result['errors'][f'weight_{seed}'] = float(abs(details['polar_weights']-expected[f'weight_{seed}'][i]).max())
                result[f'prediction_{seed}'] = int(final.argmax())
                result[f'expected_prediction_{seed}'] = int(expected[f'final_{seed}'][i].argmax())
        records.append(result)
        print(json.dumps(result), flush=True)
    report = dict(scope='fresh-chip upstream' if args.upstream_only else 'fresh-chip full',
                  device=args.device, seconds=time.perf_counter()-start, records=records)
    if not args.upstream_only:
        report['all_predictions_match'] = all(r[f'prediction_{s}'] == r[f'expected_prediction_{s}'] for r in records for s in seeds)
        report['maximum_final_score_error'] = max(r['errors'][f'final_{s}'] for r in records for s in seeds)
    out = root/'outputs'
    out.mkdir(exist_ok=True)
    (out/('upstream_validation.json' if args.upstream_only else 'full_validation.json')).write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print('MAX', {key:max(r['errors'][key] for r in records) for key in records[0]['errors']})
    if not args.upstream_only:
        assert report['all_predictions_match'], 'Fresh-chip predictions differ from selected research model.'
        assert report['maximum_final_score_error'] < .002, 'Numerical migration error exceeds declared FP32 tolerance.'


if __name__ == '__main__':
    main()
