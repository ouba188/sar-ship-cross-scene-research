"""Frozen output-kernel inference on the existing identity-isolated target set."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
spec = importlib.util.spec_from_file_location('frozen_output_kernel', PARENT/'run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
exp = run.exp
ROOT = exp.ROOT
EXT = ROOT/'external_confirmation'
from match_readout import unit
ARMS = ('independent', 'excess')


def load():
    source, x, mass, bank, idx, tr, va, te = exp.source_data()
    target = pd.read_csv(EXT/'table.csv', dtype={'mmsi': str, 'instance_id': str})
    assert len(target) == 3075
    for field in ('obs_id', 'mmsi', 'pass_group'):
        assert not set(source[field].astype(str)) & set(target[field].astype(str)), field
    with np.load(EXT/'tokens.npz') as tok, np.load(ROOT/'projection_layer1.npz') as pr:
        np.testing.assert_array_equal(tok['instance_ids'].astype(str), target.instance_id)
        np.testing.assert_array_equal(pr['bank'], bank)
        ex = unit((tok['layer1']-pr['mean'])@pr['components'].T).astype(np.float32)
        em = tok['mass'].astype(np.float32)
        ex[em <= 0] = 0
    ids = len(source)+np.arange(len(target))
    return source, target, np.concatenate([x, ex]), np.concatenate([mass, em]), bank, idx, tr, va, te, ids


def reader_for(x, mass, bank, arm):
    reader = run.Reader(x, mass, bank, arm)
    scales = pd.read_csv(PARENT/'scales.csv')
    div = scales[scales.arm.eq(arm)].sort_values('feature').divisor.to_numpy()
    reader.divisor = torch.tensor(div, device='cuda', dtype=torch.float32)
    return reader


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    source, target, x, mass, bank, idx, tr, va, te, ids = load()
    weights = np.load(PARENT/'weights.npz')
    readers = {arm: reader_for(x, mass, bank, arm) for arm in ARMS}
    for reader in readers.values():
        reader.cache_reference(np.r_[idx[te], ids])
    rows, pairs, classes, groups, replays = [], [], [], [], []
    y = target.class_id.to_numpy(int)
    scopes = dict(all=np.ones(len(y), bool), original_fold0=target.original_test_fold.eq(0).to_numpy(),
                  additional_folds=target.original_test_fold.ne(0).to_numpy())
    for seed in exp.SEEDS:
        strategy = np.load(EXT/f'strategies_seed{seed}.npz')
        np.testing.assert_array_equal(strategy['instance_ids'].astype(str), target.instance_id)
        baseline = strategy['logits'][:, 2].astype(np.float32)
        full = np.concatenate([np.load(exp.HERE/f'predictions_{seed}.npz')['full'], baseline])
        output = dict(full=baseline)
        for arm, reader in readers.items():
            model = exp.Metric()
            model.restore({k: weights[f'{arm}_{seed}_{k}'] for k in exp.STATE_KEYS})
            replay = exp.infer(reader, model, idx[te], full)[0]
            saved = np.load(PARENT/f'predictions_{arm}_{seed}.npz')
            np.testing.assert_array_equal(saved['idx'], idx)
            error = float(abs(replay-saved['original'][te]).max())
            assert error < 1e-5, (arm, seed, error)
            np.testing.assert_array_equal(replay.argmax(-1), saved['original'][te].argmax(-1))
            replays.append(dict(arm=arm, seed=seed, n=len(te), error=error))
            value, parts, _ = exp.infer(reader, model, ids, full)
            assert np.isfinite(value).all()
            output[arm] = value
            np.savez_compressed(HERE/f'parts_{arm}_{seed}.npz', parts=parts, a=weights[f'{arm}_{seed}_a'])
        for arm in ('independent', 'body_match', 'pooled', 'anchored'):
            old = np.load(EXT/'results'/f'{arm}_{seed}.npz')['score']
            assert old.shape == baseline.shape
            output['old_'+arm] = old
        np.savez_compressed(HERE/f'predictions_{seed}.npz', instance_ids=target.instance_id.to_numpy(str),
                            obs_id=target.obs_id.to_numpy(), labels=y, **output)
        correct = {arm: value.argmax(-1) == y for arm, value in output.items()}
        for arm, value in output.items():
            for scope, mask in scopes.items():
                rows.append(dict(seed=seed, arm=arm, scope=scope, **exp.dep.metrics(y[mask], value[mask])))
                for reference in ('full', 'old_independent'):
                    before, after = correct[reference][mask], correct[arm][mask]
                    pairs.append(dict(seed=seed, arm=arm, reference=reference, scope=scope,
                        rescued=int((~before & after).sum()), harmed=int((before & ~after).sum()),
                        net=int(after.sum()-before.sum())))
            for c in range(8):
                mask = y == c
                classes.append(dict(seed=seed, arm=arm, class_id=c, n=int(mask.sum()),
                    correct=int(correct[arm][mask].sum()), full_correct=int(correct['full'][mask].sum()),
                    old_independent_correct=int(correct['old_independent'][mask].sum())))
            for field in ('port', 'pass_group'):
                for group, locs in target.groupby(field).groups.items():
                    locs = np.asarray(locs)
                    old, new = correct['full'][locs], correct[arm][locs]
                    groups.append(dict(seed=seed, arm=arm, field=field, group=group, n=len(locs),
                        net=int(new.sum()-old.sum()), rescued=int((~old & new).sum()), harmed=int((old & ~new).sum())))
        print('FINISHED', seed, flush=True)
    for name, values in [('metrics', rows), ('paired', pairs), ('classes', classes), ('groups', groups)]:
        pd.DataFrame(values).to_csv(HERE/f'{name}.csv', index=False)
    summary = pd.DataFrame(rows).groupby(['scope', 'arm'])[['OA', 'BA', 'J', 'NLL', 'Brier']].mean()
    summary.to_csv(HERE/'summary.csv')
    target.to_csv(HERE/'table.csv', index=False)
    receipt = dict(n=len(target), ships=int(target.mmsi.nunique()), passes=int(target.pass_group.nunique()),
        ports=int(target.port.nunique()), new_fits=0, new_selections=0, historical_not_blind=True,
        source_replay=replays, same_source_bank=True, source_divisors_frozen=True,
        seconds=time.perf_counter()-started, raw_review='pending')
    (HERE/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(summary.to_string(), flush=True)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
