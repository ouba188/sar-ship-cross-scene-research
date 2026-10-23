"""Two source-selected information replacements, with the frozen free head."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '2')
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
STANDARD = HERE.parent / 'standard_controls'
sys.path.insert(0, str(STANDARD))
import run_controls as standard
ref, parent = standard.ref, standard.parent
ARMS = ['free_aggregate', 'free_constant']


def replace(moments, valid, arm, constant):
    if arm == 'free_aggregate':
        return ref.organized(moments, valid, 'aggregate_repeated')
    assert arm == 'free_constant'
    return constant[None] * valid[:, :, None, None]


def restore(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = standard.Head('free_shared').eval().requires_grad_(False)
    model.load_state_dict(ck['state'])
    return ck, model


def inputs(base, moments, valid, arm, constant, mu=None, scale=None):
    return ref.inputs(base, replace(moments, valid, arm, constant), valid,
                      'pol_preserved', mu, scale)


def main():
    tick = time.perf_counter()
    assert (HERE / 'PROTOCOL_ZH.md').exists()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    (HERE / 'models').mkdir(exist_ok=True)
    table = parent.load_table()
    cache = dict(np.load(parent.HERE / 'signed_features.npz'))
    valid = cache['valid'].astype(bool)
    y = table.class_id.to_numpy()
    yy = torch.tensor(y, dtype=torch.long)
    selected, curves, costs, checks, constants = [], [], [], [], []
    for fold in range(3):
        d, osum, isum = parent.load_fold(fold, table, cache)
        outer, oof = ref.load_pol(fold, table, cache, d)
        tr, va = d['train'], d['validation']
        counts = valid[tr].sum(0)
        constant = np.stack([oof[tr[valid[tr, pol]], pol].mean(0) for pol in range(2)])
        assert constant.shape == (2, 8, 2) and np.isfinite(constant).all()
        for pol in range(2):
            for c in range(8):
                constants.append(dict(fold=fold, pol=['VV', 'VH'][pol], class_id=c,
                                      source_valid_rows=int(counts[pol]),
                                      sigma=constant[pol, c, 0], kappa=constant[pol, c, 1]))
        for seed in ref.SEEDS:
            base, train_base, _ = parent.calibrated(fold, seed, d, osum, isum)
            b, bv = [torch.tensor(v, dtype=torch.float32) for v in [train_base[tr], base[va]]]
            # The original trainer resets the same RNG immediately before constructing this head.
            torch.manual_seed(seed)
            initial = parent.frozen_state(standard.Head('free_shared'))
            for arm in ARMS:
                start = time.perf_counter()
                x, carrier, mu, scale = inputs(train_base[tr], oof[tr], valid[tr], arm, constant)
                xv, cv, _, _ = inputs(base[va], outer[va], valid[va], arm, constant, mu, scale)
                transformed = replace(oof[tr], valid[tr], arm, constant)
                assert np.all(transformed[~valid[tr]] == 0)
                structural_error = (abs(transformed.sum(1) - oof[tr].sum(1)).max()
                                    if arm == 'free_aggregate' else max(
                                        abs(transformed[valid[tr, p], p] - constant[p]).max() for p in range(2)))
                assert structural_error < 1e-12
                torch.manual_seed(seed)
                model = standard.Head('free_shared')
                assert all(torch.equal(initial[k], v) for k, v in model.state_dict().items())
                parameters = sum(p.numel() for p in model.parameters())
                assert parameters == 209
                with torch.no_grad():
                    assert not model(xv, cv).any()
                best, epoch_best = parent.frozen_state(model), 0
                best_j = parent.exact_j(y[va], bv.numpy().argmax(1))
                meta = dict(fold=fold, seed=seed, arm=arm)
                trace = [dict(**meta, epoch=0, train_loss=np.nan,
                              **parent.assessment(y[va], bv.numpy().astype(float)))]
                opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
                for epoch in range(1, 201):
                    opt.zero_grad()
                    loss = parent.joint_loss(b + model(x, carrier), yy[tr])
                    assert torch.isfinite(loss)
                    loss.backward()
                    opt.step()
                    if epoch % 5:
                        continue
                    with torch.no_grad():
                        score = (bv + model(xv, cv)).numpy().astype(float)
                    j = parent.exact_j(y[va], score.argmax(1))
                    trace.append(dict(**meta, epoch=epoch, train_loss=float(loss.detach()),
                                      **parent.assessment(y[va], score)))
                    if j > best_j:
                        best_j, epoch_best, best = j, epoch, parent.frozen_state(model)
                path = HERE / 'models' / f'fold{fold}_{arm}_seed{seed}.pt'
                torch.save(dict(**meta, architecture='free_shared', state=best, mu=mu, scale=scale,
                                constant=constant, source_valid_counts=counts, epoch=epoch_best,
                                parameters=parameters), path)
                ck, replay = restore(path)
                with torch.inference_mode():
                    delta = replay(xv, cv).numpy().astype(float)
                assert parent.exact_j(y[va], (bv.numpy() + delta).argmax(1)) == best_j
                assert parent.exact_j(y[va], (base[va] + delta).argmax(1)) == best_j
                selected.append(dict(**meta, epoch=epoch_best, parameters=parameters,
                                     validation_J=float(best_j), checkpoint=str(path)))
                curves.extend(trace)
                costs.append(dict(**meta, steps=200, seconds=time.perf_counter() - start))
                checks.append(dict(**meta, zero_head_error=0, initial_parameter_error=0,
                                   information_transform_error=float(structural_error),
                                   validation_replay_J=float(best_j)))
                print(fold, seed, arm, 'epoch', epoch_best, 'J', float(best_j), flush=True)
    assert len(selected) == 18
    for name, rows in [('selections', selected), ('selection_curves', curves), ('costs', costs),
                       ('checks', checks), ('source_constants', constants)]:
        pd.DataFrame(rows).to_csv(HERE / f'{name}.csv', index=False)
    (HERE / 'selection_frozen.json').write_text(json.dumps(dict(heads=18, test_evaluated=False)), encoding='utf-8')

    reference = pd.read_csv(STANDARD / 'predictions.csv')
    key = ['fold', 'seed', 'role', 'instance_id']
    reference = reference[reference.arm.eq('free_shared')].set_index(key)
    frames, replays = [], []
    for fold in range(3):
        d, osum, isum = parent.load_fold(fold, table, cache)
        outer, _ = ref.load_pol(fold, table, cache, d)
        for seed in ref.SEEDS:
            base = parent.calibrated(fold, seed, d, osum, isum)[0]
            for arm in ARMS:
                ck, model = restore(HERE / 'models' / f'fold{fold}_{arm}_seed{seed}.pt')
                for role in ['validation', 'test']:
                    ix = d[role]
                    x, carrier, _, _ = inputs(base[ix], outer[ix], valid[ix], arm,
                                               ck['constant'], ck['mu'], ck['scale'])
                    with torch.inference_mode():
                        delta = model(x, carrier).numpy().astype(float)
                    score = base[ix] + delta
                    assert np.isfinite(score).all()
                    if role == 'validation':
                        s = next(s for s in selected if s['fold'] == fold and s['seed'] == seed and s['arm'] == arm)
                        assert float(parent.exact_j(y[ix], score.argmax(1))) == s['validation_J']
                    g = table.iloc[ix][['instance_id', 'class_id', 'class_name', 'mmsi', 'product', 'port', 'pass_group']].copy()
                    for k, v in dict(fold=fold, seed=seed, arm=arm, role=role, epoch=ck['epoch']).items():
                        g[k] = v
                    g['valid_vv'], g['valid_vh'] = valid[ix].T
                    g['prediction'], g['base_prediction'] = score.argmax(1), base[ix].argmax(1)
                    g['net'] = (g.prediction == g.class_id).astype(int) - (g.base_prediction == g.class_id).astype(int)
                    g['outcome'] = np.select([g.net.eq(1), g.net.eq(-1), g.prediction.eq(g.class_id)],
                                              ['rescued', 'harmed', 'CC'], default='WW')
                    for c in range(8):
                        g[f's{c}'], g[f'b{c}'], g[f'delta{c}'] = score[:, c], base[ix, c], delta[:, c]
                    expected = reference.loc[pd.MultiIndex.from_frame(g[key])]
                    error = float(abs(base[ix] - expected[[f'b{c}' for c in range(8)]].to_numpy()).max())
                    assert error < 2e-6 and np.array_equal(g.base_prediction, expected.base_prediction)
                    replays.append(dict(fold=fold, seed=seed, arm=arm, role=role, base_error=error,
                                        output_sum_error=float(abs(score - base[ix] - delta).max())))
                    frames.append(g)
    p = pd.concat(frames, ignore_index=True)
    p.to_csv(HERE / 'predictions.csv', index=False)
    pooled = []
    for (role, arm, seed), g in p.groupby(['role', 'arm', 'seed']):
        if role == 'test':
            assert len(g) == 4744 and g.instance_id.is_unique
        before = parent.assessment(g.class_id.to_numpy(), g[[f'b{c}' for c in range(8)]].to_numpy())
        after = parent.assessment(g.class_id.to_numpy(), g[[f's{c}' for c in range(8)]].to_numpy())
        pooled.append(dict(role=role, arm=arm, seed=seed, unique_observations=g.instance_id.nunique(),
                           net=int(g.net.sum()), rescued=int(g.net.eq(1).sum()), harmed=int(g.net.eq(-1).sum()),
                           **after, delta_OA=after['OA'] - before['OA'], delta_BA=after['BA'] - before['BA']))
    pd.DataFrame(pooled).to_csv(HERE / 'pooled.csv', index=False)
    pd.DataFrame(replays).to_csv(HERE / 'baseline_replays.csv', index=False)
    summary = dict(heads=18, steps=sum(c['steps'] for c in costs),
                   train_seconds=sum(c['seconds'] for c in costs), total_seconds=time.perf_counter() - tick,
                   reader_fits=0, extractions=0, frozen_real_retrained=False,
                   zero_epoch_heads=sum(s['epoch'] == 0 for s in selected))
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pd.DataFrame(pooled).to_string(index=False), flush=True)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
