"""Reuse the old exact outer reader; fit only its source OOF local readers."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT/'target_anchored_context_20261023/partition_replication'))
import run_replication as prep
sys.path.insert(0, str(HERE.parent))
import extract_signed as extract


def main():
    start = time.perf_counter()
    (HERE/'origins').mkdir(exist_ok=True)
    t = pd.read_csv(prep.EXP/'readout/targets.csv', dtype={'mmsi': str})
    h = np.load(prep.STRONG/'deep_features.npz')['layer3']
    old = ROOT/'contrastive_spatial_pool_20261023_oldsplit'
    readers = [dict(np.load(old/f'fold{f}_reader.npz')) for f in range(6)]
    bins = np.full(len(t), -1, int)
    for f, d in enumerate(readers):
        assert (bins[d['test']] == -1).all()
        bins[d['test']] = f
    assert (bins >= 0).all()
    y, ids = t.class_id.to_numpy(), t.mmsi.to_numpy()
    checks = []
    for fold, d in enumerate(readers):
        np.testing.assert_array_equal(d['instance_ids'], t.instance_id)
        tr, va, te = [d[k] for k in ['train', 'validation', 'test']]
        for a, b in [(tr, va), (tr, te), (va, te)]:
            assert not set(ids[a]) & set(ids[b])
            assert not set(t.iloc[a].pass_group) & set(t.iloc[b].pass_group)
        np.savez_compressed(HERE/'origins'/f'fold{fold}_outer.npz',
                            **{k: d[k] for k in ['W', 'raw_intercept', 'h0', 'fit']})
        oof = np.zeros_like(d['base'])
        count = np.zeros(len(t), int)
        for group in sorted(set(bins[tr])):
            held = tr[bins[tr] == group]
            fit = tr[(bins[tr] != group) & ~np.isin(ids[tr], ids[held])]
            assert not set(ids[fit]) & set(ids[held])
            assert set(y[fit]) == {0, 1, 2}
            v = prep.View(h, fit, 'layer3')
            z = v.transform(h)
            head = prep.fit_head(z, y, ids, fit, float(d['alpha']))
            score, _, _, origin, check = prep.decompose(h, fit, v, head, z)
            oof[held] = score[held]
            count[held] += 1
            np.savez_compressed(HERE/'origins'/f'fold{fold}_inner{group}.npz',
                                fit=fit, held=held, **origin)
            checks.append(dict(fold=fold, group=group, fit_n=len(fit), held_n=len(held), **check))
        assert (count[tr] == 1).all() and count.sum() == len(tr)
        np.savez_compressed(HERE/f'fold{fold}_inputs.npz', oof=oof,
                            **{k: d[k] for k in ['base', 'train', 'validation', 'test', 'alpha', 'instance_ids']})
        print(f'prepared fold{fold}; source={len(tr)}', flush=True)
    pd.DataFrame(checks).to_csv(HERE/'oof_checks.csv', index=False)
    extract.HERE, extract.NEW = HERE, HERE
    # Match the historical batch16 backbone kernel for exact std replay.
    extract.main(reference_std=np.load(old/'delta_features.npz')['delta'][:, :, 0], batch_size=16)
    print(f'preparation_seconds={time.perf_counter()-start:.3f}', flush=True)


if __name__ == '__main__':
    main()
