"""Source-only information isolation in the unchanged additive architecture."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
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
sys.path.insert(0, str(HERE.parent))
import run_joint as run
parent, strong = run.parent, run.strong
previous = strong.previous
sys.path.insert(0, str(HERE.parent.parent / 'mean_only_control'))
import run_mean as mean_run
ARMS = ('sigma_only', 'kappa_only')
MASKED = {'sigma_only': [5, 6, 11, 12], 'kappa_only': [3, 4, 9, 10]}
SCORE = [f's{c}' for c in range(8)]
BASE = [f'b{c}' for c in range(8)]


def read_validation(path, arm):
    # Read role metadata first; never parse target score rows from the mixed CSV.
    roles = pd.read_csv(path, usecols=['role', 'arm'])
    keep = set(np.flatnonzero(roles.role.eq('validation') & roles.arm.eq(arm)) + 1)
    return pd.read_csv(path, skiprows=lambda i: i != 0 and i not in keep,
                       dtype={'mmsi': str}, float_precision='round_trip')


def mask_slots(x, arm):
    out = x.clone()
    out[..., MASKED[arm]] = 0
    return out


def load_source(fold, table, cache):
    with np.load(parent.PARENT / 'baseline' / f'fold{fold}_inputs.npz') as z:
        np.testing.assert_array_equal(z['instance_ids'], table.instance_id)
        d = {k: z[k] for k in ('train', 'validation')}
    tr, va = d['train'], d['validation']
    assert not set(table.iloc[tr].mmsi) & set(table.iloc[va].mmsi)
    assert table.iloc[tr].time.str[:10].lt('2025-09-01').all()
    assert table.iloc[va].time.str[:7].eq('2025-09').all()
    counts = np.zeros(len(table), int)
    for inner in range(4):
        with np.load(parent.PARENT / 'baseline/models' / f'fold{fold}_inner{inner}.npz') as z:
            fit, held = z['fit'], z['held']
        assert set(fit) <= set(tr) and set(held) <= set(tr)
        assert not set(table.iloc[fit].mmsi) & set(table.iloc[held].mmsi)
        counts[held] += 1
    assert (counts[tr] == 1).all() and counts.sum() == len(tr)
    # Reuse the exact existing OOF assembly, not outer training-in-sample moments.
    means, means_oof, moments, moments_oof = run.load_features(fold, table, cache, d)
    return d, {'train': (means_oof[tr], moments_oof[tr]),
               'validation': (means[va], moments[va])}


def source_baseline(fold, oldseed, seed, table, d, old):
    mapping = old.reset_index().set_index('obs_id')['index'].loc[table.obs_id].to_numpy()
    ix = mapping[d['validation']]
    np.testing.assert_array_equal(old.iloc[ix].instance_id, table.iloc[d['validation']].instance_id)
    with np.load(run.OLD / 'geometry_evidence_gate_20261021' / f'inputs_f{fold}_s{oldseed}.npz') as z:
        raw_val = (z['base'][ix] + z['residual'][ix]).astype(float)
    with np.load(run.OLD / 'held_vessel_gate_20261022' / f'oof_f{fold}.npz') as z:
        rows, assignment = z['rows'], z['inner']
        raw_train = (z['base'][oldseed] + z['residual'][oldseed]).astype(float)
    np.testing.assert_array_equal(old.iloc[rows].instance_id, table.iloc[d['train']].instance_id)
    np.testing.assert_array_equal(old.iloc[rows].class_id, table.iloc[d['train']].class_id)
    for k in range(3):
        assert not set(old.iloc[rows[assignment == k]].mmsi) & set(old.iloc[rows[assignment != k]].mmsi)
    ck, model = previous.restore(previous.HERE / 'models' / f'fold{fold}_score_only_seed{seed}.pt')
    assert not any(p.requires_grad for p in model.parameters())
    result = {}
    for role, raw in [('train', raw_train), ('validation', raw_val)]:
        x, _, _, _ = previous.inputs('score_only', raw, None, None, ck['mu'], ck['scale'])
        with torch.inference_mode():
            result[role] = raw + model(x, None).numpy().astype(float)
    return result, ck['epoch']


def predict_frame(table, ix, valid, base, score, meta):
    g = table.iloc[ix][['instance_id', 'obs_id', 'class_id', 'class_name', 'mmsi',
                        'port', 'product', 'pass_group', 'time']].copy()
    for k, v in meta.items():
        g[k] = v
    y = g.class_id.to_numpy()
    g['valid_vv'], g['valid_vh'] = valid[ix].T
    g['prediction'], g['base_prediction'] = score.argmax(1), base.argmax(1)
    g['net'] = (score.argmax(1) == y).astype(int) - (base.argmax(1) == y).astype(int)
    g['outcome'] = np.select([g.net.eq(1), g.net.eq(-1), g.prediction.eq(g.class_id)],
                             ['rescued', 'harmed', 'CC'], default='WW')
    br, sr = base.copy(), score.copy()
    br[np.arange(len(y)), y] = sr[np.arange(len(y)), y] = -np.inf
    g['old_rival'], g['final_rival'] = br.argmax(1), sr.argmax(1)
    for c in range(8):
        g[f'b{c}'], g[f's{c}'], g[f'delta{c}'] = base[:, c], score[:, c], score[:, c] - base[:, c]
    assert np.isfinite(score).all()
    return g


def restore(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = run.Head('additive').eval().requires_grad_(False)
    model.load_state_dict(ck['state'])
    return ck, model


def main():
    started = time.perf_counter()
    assert (HERE / 'PROTOCOL_ZH.md').exists()
    (HERE / 'models').mkdir(exist_ok=True)
    assert not list((HERE / 'models').glob('*.pt')), 'Do not overwrite completed models.'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    table = parent.load_table()
    with np.load(parent.HERE / 'signed_features.npz') as z:
        cache = {k: z[k] for k in ('instance_ids', 'reader_names', 'mean_pol', 'moments_pol', 'valid')}
    np.testing.assert_array_equal(cache['instance_ids'], table.instance_id)
    valid = cache['valid'].astype(bool)
    old = pd.read_csv(run.OLD / 'conditional_evidence_pilot/cohort.csv', dtype={'mmsi': str})
    reference = read_validation(HERE.parent / 'predictions.csv', 'additive').set_index(['fold', 'seed', 'instance_id'])
    mean_reference = read_validation(mean_run.HERE / 'predictions.csv', 'mean_only').set_index(['fold', 'seed', 'instance_id'])
    curves, selections, checks, scopes, costs, frames = [], [], [], [], [], []
    contexts = {}
    for fold in range(3):
        d, info = load_source(fold, table, cache)
        for role, ix in d.items():
            g = table.iloc[ix]
            scopes.append(dict(fold=fold, role=role, n=len(ix), instances=g.instance_id.nunique(),
                mmsi=g.mmsi.nunique(), passes=g.pass_group.nunique(), products=g['product'].nunique(),
                ports=g.port.nunique(), classes=g.class_id.nunique(), valid_vh=int(valid[ix, 1].sum()),
                date_min=g.time.min(), date_max=g.time.max(), inner_query_overlap=0))
        for oldseed, seed in strong.SEED_PAIRS:
            base, tepoch = source_baseline(fold, oldseed, seed, table, d, old)
            tr, va = d['train'], d['validation']
            x, mu, scale = run.inputs(base['train'], *info['train'], valid[tr])
            xv, _, _ = run.inputs(base['validation'], *info['validation'], valid[va], mu, scale)
            full_ck, full = run.restore(HERE.parent / 'models' / f'fold{fold}_additive_seed{seed}.pt')
            np.testing.assert_array_equal(mu, full_ck['mu'])
            np.testing.assert_array_equal(scale, full_ck['scale'])
            mean_ck, mean_model = mean_run.restore(mean_run.HERE / 'models' / f'fold{fold}_mean_only_seed{seed}.pt')
            keys = pd.MultiIndex.from_arrays([[fold] * len(va), [seed] * len(va), table.iloc[va].instance_id])
            expected = reference.loc[keys]
            baseline_error = float(np.abs(base['validation'] - expected[BASE].to_numpy()).max())
            assert baseline_error < 1e-12
            with torch.inference_mode():
                full_score = base['validation'] + full(xv).numpy().astype(float)
            replay_error = float(np.abs(full_score - expected[SCORE].to_numpy()).max())
            assert replay_error < 1e-6
            np.testing.assert_array_equal(full_score.argmax(1), expected.prediction)
            meta = dict(fold=fold, oldseed=oldseed, seed=seed, T_epoch=tepoch)
            for role, xx in [('train', x), ('validation', xv)]:
                ix = d[role]
                means, _ = info[role]
                carrier_mean = np.repeat(means[..., None], 2, axis=-1)
                xm, cm, _, _ = mean_run.inputs('mean_only', base[role], carrier_mean, valid[ix], mean_ck['mu'], mean_ck['scale'])
                with torch.inference_mode():
                    fs = base[role] + full(xx).numpy().astype(float)
                    ms = base[role] + mean_model(xm, cm).numpy().astype(float)
                if role == 'validation':
                    me = mean_reference.loc[keys]
                    assert np.abs(base[role] - me[BASE].to_numpy()).max() < 1e-12
                    assert np.abs(ms - me[SCORE].to_numpy()).max() < 1e-6
                for arm, score, epoch in [('full', fs, full_ck['epoch']), ('mean_only', ms, mean_ck['epoch']), ('B', base[role], 0)]:
                    frames.append(predict_frame(table, ix, valid, base[role], score,
                        dict(**meta, arm=arm, role=role, epoch=epoch)))
            contexts[fold, seed] = dict(d=d, base=base, x=x, xv=xv, meta=meta)
            ytr = torch.tensor(table.iloc[tr].class_id.to_numpy(), dtype=torch.long)
            yv = table.iloc[va].class_id.to_numpy()
            bt = torch.tensor(base['train'], dtype=torch.float32)
            initial = None
            for arm in ARMS:
                tick = time.perf_counter()
                xt, xval = mask_slots(x, arm), mask_slots(xv, arm)
                kept = [i for i in range(15) if i not in MASKED[arm]]
                assert torch.equal(xt[..., kept], x[..., kept])
                assert not xt[..., MASKED[arm]].any() and not xval[..., MASKED[arm]].any()
                torch.manual_seed(seed)
                model = run.Head('additive')
                assert sum(p.numel() for p in model.parameters()) == 266
                init = parent.frozen_state(model)
                if initial is None:
                    initial = init
                else:
                    assert all(torch.equal(init[k], initial[k]) for k in init)
                with torch.no_grad():
                    assert not model(xt).any() and not model(xval).any()
                best, chosen = init, 0
                best_j = parent.exact_j(yv, base['validation'].argmax(1))
                armmeta = dict(**meta, arm=arm)
                curves.append(dict(**armmeta, epoch=0, train_loss=np.nan,
                                   **parent.assessment(yv, base['validation'])))
                opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
                for epoch in range(1, 201):
                    opt.zero_grad()
                    loss = parent.joint_loss(bt + model(xt), ytr)
                    assert torch.isfinite(loss)
                    loss.backward()
                    opt.step()
                    if epoch % 5:
                        continue
                    with torch.no_grad():
                        score = base['validation'] + model(xval).numpy().astype(float)
                    j = parent.exact_j(yv, score.argmax(1))
                    curves.append(dict(**armmeta, epoch=epoch, train_loss=float(loss.detach()),
                                       **parent.assessment(yv, score)))
                    if j > best_j:
                        best, chosen, best_j = parent.frozen_state(model), epoch, j
                path = HERE / 'models' / f'fold{fold}_{arm}_seed{seed}.pt'
                torch.save(dict(**armmeta, state=best, terminal_state=parent.frozen_state(model),
                    initial_state=init, mu=mu, scale=scale, epoch=chosen, parameters=266,
                    masked_columns=MASKED[arm], calibration_is_oof=False), path)
                ck, saved = restore(path)
                with torch.inference_mode():
                    score = base['validation'] + saved(xval).numpy().astype(float)
                assert parent.exact_j(yv, score.argmax(1)) == best_j
                selections.append(dict(**armmeta, epoch=chosen, validation_J=float(best_j), parameters=266, checkpoint=str(path)))
                costs.append(dict(**armmeta, steps=200, seconds=time.perf_counter() - tick))
                checks.append(dict(**armmeta, baseline_max_error=baseline_error, full_replay_max_error=replay_error,
                    scaler_equal=True, init_equal=True, zero_head_error=0, masked_zero=True,
                    retained_slots_equal=True, selected_replay=True))
                print(fold, seed, arm, 'epoch', chosen, 'J', float(best_j), flush=True)
    for name, rows in [('selections', selections), ('selection_curves', curves), ('checks', checks),
                       ('source_scope', scopes), ('costs', costs)]:
        pd.DataFrame(rows).to_csv(HERE / f'{name}.csv', index=False)
    assert len(selections) == 18
    (HERE / 'selection_frozen.json').write_text(json.dumps(dict(heads=18, test_evaluated=False,
        test_scores_read=False, calibration_is_oof=False), indent=2), encoding='utf-8')
    for q in selections:
        ck, model = restore(Path(q['checkpoint']))
        context = contexts[q['fold'], q['seed']]
        for role, key in [('train', 'x'), ('validation', 'xv')]:
            ix = context['d'][role]
            xx = mask_slots(context[key], q['arm'])
            with torch.inference_mode():
                score = context['base'][role] + model(xx).numpy().astype(float)
            frames.append(predict_frame(table, ix, valid, context['base'][role], score,
                dict(**context['meta'], arm=q['arm'], role=role, epoch=q['epoch'])))
    p = pd.concat(frames, ignore_index=True)
    assert set(p.role) == {'train', 'validation'}
    p.to_csv(HERE / 'source_predictions.csv', index=False)
    pooled = []
    for (role, arm, seed), g in p.groupby(['role', 'arm', 'seed']):
        m = parent.assessment(g.class_id.to_numpy(), g[SCORE].to_numpy())
        b = parent.assessment(g.class_id.to_numpy(), g[BASE].to_numpy())
        pooled.append(dict(role=role, arm=arm, seed=seed, **m, net=int(g.net.sum()),
            unique_instances=g.instance_id.nunique(), unique_mmsi=g.mmsi.nunique(),
            **{f'delta_{k}_B': m[k]-b[k] for k in ['OA', 'BA', 'NLL', 'Brier']}))
    pd.DataFrame(pooled).to_csv(HERE / 'source_pooled.csv', index=False)
    summary = dict(new_heads=18, reused_full=9, reused_mean_only=9, training_steps=3600,
        train_seconds=sum(v['seconds'] for v in costs), total_seconds=time.perf_counter()-started,
        zero_epochs=sum(v['epoch'] == 0 for v in selections), test_evaluated=False,
        feature_extractions=0, reader_fits=0, calibration_fits=0,
        source_predictions=len(p), source_validation_rows_per_seed=int(len(p.query("role=='validation' and arm=='B' and seed==17"))))
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pd.DataFrame(pooled).query("role=='validation'").to_string(index=False), flush=True)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
