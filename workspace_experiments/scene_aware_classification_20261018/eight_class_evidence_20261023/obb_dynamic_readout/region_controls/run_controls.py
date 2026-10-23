"""Matched-head region identity and incremental-region controls on frozen caches."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
import run_heads as old
import numpy as np
import pandas as pd
import torch
import argparse
import time

ARMS = ('shuffled', 'body', 'body_end', 'body_side')
SEEDS = old.SEEDS


def raw_inputs(base, info, arm, obs_ids):
    changed = dict(info)
    value = info['summary'].copy()
    original = value[:, :, 1:4].copy()
    if arm == 'shuffled':
        permutations = np.stack([np.random.default_rng([4711, int(o)]).permutation(3) for o in obs_ids])
        value[:, :, 1:4] = np.take_along_axis(original, permutations[:, None, :, None, None], axis=2)
    elif arm in ('body', 'body_end', 'body_side'):
        drop = {'body': (2, 3), 'body_end': (3,), 'body_side': (2,)}[arm]
        for region in drop:
            value[:, :, region] = 0
    elif arm != 'dynamic':
        raise ValueError(arm)
    changed['summary'] = value
    return old.raw_inputs(base, changed, 'dynamic')


def check_inputs(base, info, obs_ids):
    full = old.raw_inputs(base, info, 'dynamic')
    np.testing.assert_array_equal(raw_inputs(base, info, 'dynamic', obs_ids), full)
    for arm in ARMS:
        raw = raw_inputs(base, info, arm, obs_ids)
        assert raw.shape == full.shape and raw.shape[-1] == 69
        np.testing.assert_array_equal(raw[..., -32:], full[..., -32:])
        np.testing.assert_array_equal(raw[..., 0], full[..., 0])
        # Compare ordered blocks directly; catches partial-class or cross-polarization shuffles.
        for i in range(min(5, len(base))):
            perm = np.random.default_rng([4711, int(obs_ids[i])]).permutation(3)
            for p in range(2):
                for r in range(3):
                    dest = 1 + p * 18 + r * 6
                    src_r = int(perm[r]) if arm == 'shuffled' else r
                    src = 1 + p * 18 + src_r * 6
                    keep = arm == 'shuffled' or r == 0 or (arm == 'body_end' and r == 1) or (arm == 'body_side' and r == 2)
                    expected = full[i, :, src:src+6] if keep else np.zeros((8, 6))
                    np.testing.assert_array_equal(raw[i, :, dest:dest+6], expected)


def source_train():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    assert not (HERE / 'source_selection_frozen.json').exists()
    (HERE / 'models').mkdir(exist_ok=True)
    assert not list((HERE / 'models').glob('*.pt'))
    table, _, _ = old.load_data()
    ref = old.reference('source')
    curves, selections, checks, frames, costs = [], [], [], [], []
    for fold in range(3):
        infos = {role: old.load_info(fold, role, table) for role in ('train', 'validation')}
        labels = {role: table.iloc[info['indices']].class_id.to_numpy() for role, info in infos.items()}
        ids = {role: table.iloc[info['indices']].obs_id.to_numpy() for role, info in infos.items()}
        y = torch.tensor(labels['train'], dtype=torch.long)
        for seed in SEEDS:
            bases = {role: old.baseline(ref, table, info['indices'], fold, seed, role) for role, info in infos.items()}
            bt = torch.tensor(bases['train'], dtype=torch.float32)
            if seed == SEEDS[0]:
                for role in infos:
                    check_inputs(bases[role], infos[role], ids[role])
            full_ck = torch.load(PARENT / 'models' / f'fold{fold}_dynamic_seed{seed}.pt', map_location='cpu', weights_only=False)
            for arm in ARMS:
                tick = time.perf_counter()
                raw = raw_inputs(bases['train'], infos['train'], arm, ids['train'])
                rv = raw_inputs(bases['validation'], infos['validation'], arm, ids['validation'])
                xt, mu, scale = old.standardize(raw)
                xv, _, _ = old.standardize(rv, mu, scale)
                np.testing.assert_array_equal(mu[..., -32:], full_ck['mu'][..., -32:])
                np.testing.assert_array_equal(scale[..., -32:], full_ck['scale'][..., -32:])
                torch.manual_seed(seed)
                model = old.Head(69)
                init = old.frozen(model)
                assert all(torch.equal(init[k], full_ck['initial_state'][k]) for k in init)
                assert sum(p.numel() for p in model.parameters()) == 853
                with torch.no_grad():
                    assert not model(xt).any() and not model(xv).any()
                best, epoch_best = init, 0
                best_j = old.exact_j(labels['validation'], bases['validation'].argmax(1))
                meta = dict(fold=fold, seed=seed, arm=arm)
                curves.append(dict(**meta, epoch=0, train_loss=np.nan, **old.assessment(labels['validation'], bases['validation'])))
                opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
                for epoch in range(1, 201):
                    opt.zero_grad()
                    loss = old.joint_loss(bt + model(xt), y)
                    assert torch.isfinite(loss)
                    loss.backward()
                    opt.step()
                    if epoch % 5:
                        continue
                    with torch.no_grad():
                        s = bases['validation'] + model(xv).numpy().astype(float)
                    j = old.exact_j(labels['validation'], s.argmax(1))
                    curves.append(dict(**meta, epoch=epoch, train_loss=float(loss.detach()), **old.assessment(labels['validation'], s)))
                    if j > best_j:
                        best, epoch_best, best_j = old.frozen(model), epoch, j
                ck = dict(**meta, state=best, initial_state=init, terminal_state=old.frozen(model), epoch=epoch_best,
                          mu=mu, scale=scale, input_dim=69, parameters=853, calibration_and_keep_fully_oof=False)
                torch.save(ck, HERE / 'models' / f'fold{fold}_{arm}_seed{seed}.pt')
                model.load_state_dict(best)
                model.eval()
                for role, x in [('train', xt), ('validation', xv)]:
                    with torch.inference_mode():
                        s = bases[role] + model(x).numpy().astype(float)
                    if epoch_best == 0:
                        np.testing.assert_array_equal(s, bases[role])
                    if role == 'validation':
                        assert old.exact_j(labels[role], s.argmax(1)) == best_j
                    frames.append(old.predict_frame(table, infos[role]['indices'], bases[role], s, fold, seed, role, arm, epoch_best))
                selections.append(dict(**meta, epoch=epoch_best, validation_J=float(best_j), parameters=853,
                                       nonconstant_columns=int((raw.std((0, 1)) > 1e-8).sum()), selected_utc=old.now()))
                checks.append(dict(**meta, common_quality_exact=True, full_init_exact=True, zero_init_exact=True, source_only_scaler=True))
                costs.append(dict(**meta, seconds=time.perf_counter()-tick))
                for name, rows in [('selection_curves', curves), ('selections', selections), ('training_checks', checks), ('training_costs', costs)]:
                    pd.DataFrame(rows).to_csv(HERE / f'{name}.csv', index=False)
                print(fold, seed, arm, 'epoch', epoch_best, 'J', float(best_j), flush=True)
    pred = pd.concat(frames, ignore_index=True)
    pred.to_csv(HERE / 'source_predictions.csv', index=False)
    old.pooled(pred).to_csv(HERE / 'source_pooled.csv', index=False)
    assert len(selections) == 36
    old.save_json(HERE / 'source_selection_frozen.json', dict(completed_utc=old.now(), heads=36, steps=7200,
        seconds=time.perf_counter()-started, test_evaluated=False, old_stage_fits=0,
        zero_epochs=sum(r['epoch']==0 for r in selections), reused_dynamic_heads=9))


def test_evaluate():
    started = time.perf_counter()
    torch.set_num_threads(2)
    assert (HERE / 'source_selection_frozen.json').exists()
    assert not (HERE / 'test_predictions.csv').exists()
    table, _, _ = old.load_data()
    ref = old.reference('test')
    frames, checks = [], []
    prior = pd.read_csv(PARENT / 'test_predictions.csv', dtype={'mmsi': str}, float_precision='round_trip')
    prior = prior[prior.arm.eq('dynamic')].set_index(['fold', 'seed', 'instance_id'])
    for fold in range(3):
        info = old.load_info(fold, 'test', table)
        ix = info['indices']
        ids = table.iloc[ix].obs_id.to_numpy()
        for seed in SEEDS:
            base = old.baseline(ref, table, ix, fold, seed, 'test')
            if seed == SEEDS[0]:
                check_inputs(base, info, ids)
            for arm in (*ARMS, 'dynamic'):
                parent = PARENT if arm == 'dynamic' else HERE
                ck = torch.load(parent / 'models' / f'fold{fold}_{arm}_seed{seed}.pt', map_location='cpu', weights_only=False)
                raw = raw_inputs(base, info, arm, ids)
                x, _, _ = old.standardize(raw, ck['mu'], ck['scale'])
                model = old.Head(69)
                model.load_state_dict(ck['state'])
                model.eval()
                with torch.inference_mode():
                    score = base + model(x).numpy().astype(float)
                error = 0.
                if arm == 'dynamic':
                    keys = pd.MultiIndex.from_arrays([[fold]*len(ix), [seed]*len(ix), table.iloc[ix].instance_id])
                    error = float(abs(score - prior.loc[keys, old.SCORE].to_numpy()).max())
                    assert error < 1e-6
                if ck['epoch'] == 0:
                    np.testing.assert_array_equal(score, base)
                frames.append(old.predict_frame(table, ix, base, score, fold, seed, 'test', arm, ck['epoch']))
                checks.append(dict(fold=fold, seed=seed, arm=arm, epoch=ck['epoch'], n=len(ix), full_replay_error=error, new_fits=0))
    pred = pd.concat(frames, ignore_index=True)
    pred.to_csv(HERE / 'test_predictions.csv', index=False)
    old.pooled(pred).to_csv(HERE / 'test_pooled.csv', index=False)
    pd.DataFrame(checks).to_csv(HERE / 'test_checks.csv', index=False)
    old.save_json(HERE / 'test_complete.json', dict(completed_utc=old.now(), new_fits=0,
        cached_inference_seconds=time.perf_counter()-started, observations=pred.instance_id.nunique(), reused_heads=9, new_heads=36))
    print(old.pooled(pred).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['source', 'test'])
    args = parser.parse_args()
    source_train() if args.stage == 'source' else test_evaluate()
