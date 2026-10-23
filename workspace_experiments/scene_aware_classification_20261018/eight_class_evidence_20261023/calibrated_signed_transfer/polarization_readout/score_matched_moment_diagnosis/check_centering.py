"""Measure centering's effect before deciding whether corrected replay is needed."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import pandas as pd
from run_diagnosis import HERE, source, parent, match_pairs


def mean_stats(b, prefix):
    m = b.mean(1)
    return {f'{prefix}_rowmean_{k}': float(v) for k, v in
            [('min', m.min()), ('max', m.max()), ('std', m.std()), ('range', np.ptp(m))]}


def main():
    p = pd.read_csv(HERE / 'predictions.csv', dtype={'mmsi': str, 'donor_mmsi': str})
    table = parent.load_table()
    cache = dict(np.load(parent.HERE / 'signed_features.npz'))
    checks, pairs, scales = [], [], []
    for fold in range(3):
        d, _, _ = parent.load_fold(fold, table, cache)
        for oldseed, seed in source.SEED_PAIRS:
            base, train_base = source.load_old(fold, oldseed, table, d)
            t = train_base[d['train']]
            tc = t - t.mean(1, keepdims=True)
            mu, sd = t.mean(0), t.std(0).clip(1e-8)
            cmu, csd = tc.mean(0), tc.std(0).clip(1e-8)
            for c in range(8):
                scales.append(dict(fold=fold, seed=seed, class_id=c, raw_mean=mu[c], raw_scale=sd[c],
                                   centered_mean=cmu[c], centered_scale=csd[c]))
            for role in ['validation', 'test']:
                g = p[p.fold.eq(fold) & p.seed.eq(seed) & p.role.eq(role)].reset_index(drop=True)
                b = g[[f'b{c}' for c in range(8)]].to_numpy()
                bc = b - b.mean(1, keepdims=True)
                v = g[['valid_vv', 'valid_vh']].to_numpy(bool)
                ids, mmsi = g.instance_id.to_numpy(), g.mmsi.to_numpy()
                rawdonor, rawdist, *_ = match_pairs(b, v, ids, mmsi, mu, sd)
                donor, dist, *_ = match_pairs(bc, v, ids, mmsi, cmu, csd)
                assert np.array_equal(b.argmax(1), bc.argmax(1))
                assert np.array_equal(ids[rawdonor], g.donor_instance_id)
                rmatched = rawdonor != np.arange(len(g))
                matched = donor != np.arange(len(g))
                # Compare the same complete set of admissible distances, not just selected edges.
                left, right = np.triu_indices(len(g), k=1)
                allowed = ((b.argmax(1)[left] == b.argmax(1)[right]) &
                           (v[left] == v[right]).all(1) & (mmsi[left] != mmsi[right]))
                left, right = left[allowed], right[allowed]
                z, cz = (b-mu)/sd, (bc-cmu)/csd
                rd = np.linalg.norm(z[left]-z[right], axis=1)
                cd = np.linalg.norm(cz[left]-cz[right], axis=1)
                checks.append(dict(fold=fold, seed=seed, role=role, n=len(g),
                                   **mean_stats(t, 'source'), **mean_stats(b, 'evaluation'),
                                   max_scale_change=float(abs(sd-csd).max()), admissible_edges=len(rd),
                                   distance_abs_change_max=float(abs(rd-cd).max()),
                                   distance_abs_change_mean=float(abs(rd-cd).mean()),
                                   distance_abs_change_median=float(np.median(abs(rd-cd))),
                                   changed_donor_rows=int((donor != rawdonor).sum()),
                                   changed_matched_status=int((matched != rmatched).sum()),
                                   raw_matched=int(rmatched.sum()), centered_matched=int(matched.sum())))
                h = g[['instance_id', 'mmsi']].copy()
                h['raw_donor'] = ids[rawdonor]
                h['centered_donor'] = ids[donor]
                h['raw_distance'], h['centered_distance'] = rawdist, dist
                for k, val in dict(fold=fold, seed=seed, role=role).items():
                    h[k] = val
                pairs.append(h)
    pd.DataFrame(checks).to_csv(HERE / 'centering_check.csv', index=False)
    pd.DataFrame(scales).to_csv(HERE / 'centering_scales.csv', index=False)
    pd.concat(pairs, ignore_index=True).to_csv(HERE / 'centering_pair_changes.csv', index=False)
    summary = dict(changed_donor_rows=sum(r['changed_donor_rows'] for r in checks),
                   max_source_rowmean_range=max(r['source_rowmean_range'] for r in checks),
                   max_evaluation_rowmean_range=max(r['evaluation_rowmean_range'] for r in checks),
                   max_distance_change=max(r['distance_abs_change_max'] for r in checks),
                   corrected_replay_needed=any(r['changed_donor_rows'] for r in checks))
    (HERE / 'centering_check.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pd.DataFrame(checks).to_string(index=False), flush=True)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
