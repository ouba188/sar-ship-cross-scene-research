"""Per-reader OSS2 class response means and local distributions."""
import json
import time
import sys
import importlib.util
import numpy as np
import pandas as pd
import torch
from extract_features import HERE, EIGHT, load_batch, network, feature_map, spatial_weights

spec = importlib.util.spec_from_file_location('ports_signed_moments',
    EIGHT / 'calibrated_signed_transfer/extract_moments.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
weighted_moments = original.weighted_moments


def main():
    started = time.perf_counter()
    table = pd.read_csv(HERE / 'cohort.csv', dtype={'mmsi': str})
    raw = np.load(HERE / 'features.npz')
    np.testing.assert_array_equal(raw['sample_ids'], table.sample_id)
    names, readers = [], []
    for fold in range(6):
        for suffix in ['outer'] + [f'inner{k}' for k in range(4)]:
            name = f'fold{fold}_{suffix}'
            with np.load(HERE / 'readers/models' / f'{name}.npz') as z:
                readers.append({k: z[k] for k in ['W', 'intercept']})
            names.append(name)
    w = torch.tensor(np.stack([r['W'].reshape(8, 2, 1024).transpose(1, 0, 2)
        for r in readers]), device='cuda', dtype=torch.float64)
    means = np.zeros((30, len(table), 2, 8), np.float64)
    moments = np.zeros((30, len(table), 2, 8, 2), np.float64)
    model = network()
    last = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, len(table), 16):
            stop = min(first+16, len(table))
            x, mask = load_batch(table, first, stop)
            wp = w[:, torch.arange(len(x), device='cuda') % 2]
            for rotation in range(4):
                z = feature_map(model, x, rotation)
                weight = spatial_weights(mask, z, rotation).flatten(2).double()[None]
                response = torch.einsum('fbck,bku->fbcu', wp, z.flatten(2).double())
                moment, mu, _ = weighted_moments(response, weight)
                means[:, first:stop] += mu.cpu().numpy().reshape(30, stop-first, 2, 8)/4
                moments[:, first:stop] += moment.cpu().numpy().reshape(30, stop-first, 2, 8, 2)/4
            if time.perf_counter()-last > 25:
                print('OSS2 class distributions', stop, '/', len(table), 'seconds',
                    round(time.perf_counter()-started, 1), flush=True)
                last = time.perf_counter()
    checks = []
    for i, name in enumerate(names):
        expected = raw['layer3'].astype(np.float64) @ readers[i]['W'].T + readers[i]['intercept']
        actual = means[i].sum(1) + readers[i]['intercept']
        error = float(np.max(abs(actual-expected)))
        assert error < 1e-4
        flips = int((actual.argmax(1) != expected.argmax(1)).sum())
        checks.append(dict(reader=name, max_mean_replay_error=error, top1_changes=flips))
    assert np.isfinite(means).all() and np.isfinite(moments).all()
    assert np.all(abs(moments[..., 1]) <= moments[..., 0]+1e-9)
    np.savez_compressed(HERE / 'moments.npz', sample_ids=table.sample_id.to_numpy(str),
        reader_names=np.array(names), mean_pol=means, moments_pol=moments,
        valid=raw['valid'])
    pd.DataFrame(checks).to_csv(HERE / 'moment_replays.csv', index=False)
    info = dict(rows=len(table), readers=30, seconds=time.perf_counter()-started,
        max_mean_replay_error=max(r['max_mean_replay_error'] for r in checks),
        replay_top1_changes=sum(r['top1_changes'] for r in checks),
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (HERE / 'moments_summary.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
