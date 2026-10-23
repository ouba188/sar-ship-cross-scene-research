"""Source-selected high/low restrictions, followed by frozen target evaluation."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as u
import run_attention as original
import numpy as np
import pandas as pd
import torch
import json
import time
import argparse

ARMS = ['high', 'low']
MASKED = {'high': [5, 6, 11, 12], 'low': [3, 4, 9, 10]}
SCORE, BASE = u.SCORE, u.BASE


def event(name, **values):
    with (HERE/'stages.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(time_utc=u.now(), event=name, **values))+'\n')


def save_json(name, obj):
    (HERE/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def side_slots(hb, arm):
    result = np.zeros_like(hb)
    if arm == 'high':
        result[..., 0] = hb[..., 0]+hb[..., 1]
        assert not result[..., 1].any() and (result[..., 0] >= -1e-10).all()
    else:
        result[..., 1] = hb[..., 1]-hb[..., 0]
        assert not result[..., 0].any() and (result[..., 1] <= 1e-10).all()
    return result


def restore(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = u.run.Head('additive').eval().requires_grad_(False)
    model.load_state_dict(ck['state'])
    return ck, model


def parent_prediction(stage):
    p = pd.read_csv(HERE.parent/f'{stage}_predictions.csv', dtype={'mmsi': str}, float_precision='round_trip')
    p = p[p.arm.eq(u.ARM)].copy()
    p['arm'] = 'two_sided'
    return p


def train():
    assert (HERE/'PROTOCOL_ZH.md').exists()
    (HERE/'models').mkdir(exist_ok=True)
    assert not list((HERE/'models').glob('*.pt'))
    tick = time.perf_counter()
    event('source_training_start')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    table, cache = u.parent.load_table(), u.cache_load()
    ac = original.load_attention('source', cache)
    valid = cache['valid'].astype(bool)
    previous = parent_prediction('source')
    lookup = previous.set_index(['fold', 'seed', 'role', 'instance_id'])
    frames, selections, curves, checks, costs = [previous], [], [], [], []
    for fold in range(3):
        with np.load(u.parent.PARENT/'baseline'/f'fold{fold}_inputs.npz') as z:
            d = {k: z[k] for k in ['train', 'validation']}
        assert not set(table.iloc[d['train']].mmsi) & set(table.iloc[d['validation']].mmsi)
        # Verify held/fit identities with the existing source-only loader.
        u.source.load_source(fold, table, cache)
        info = original.information(fold, d, cache, ac)
        tr, va = d['train'], d['validation']
        ytr = torch.tensor(table.iloc[tr].class_id.to_numpy(), dtype=torch.long)
        yv = table.iloc[va].class_id.to_numpy()
        for oldseed, seed in u.strong.SEED_PAIRS:
            base = {}
            for role, ix in d.items():
                keys = pd.MultiIndex.from_arrays([[fold]*len(ix), [seed]*len(ix), [role]*len(ix), table.iloc[ix].instance_id])
                ref = lookup.loc[keys]
                np.testing.assert_array_equal(ref.class_id, table.iloc[ix].class_id)
                base[role] = ref[BASE].to_numpy()
                tepoch = int(ref.T_epoch.iloc[0])
            parent_ck, parent_head = u.restore(HERE.parent/'models'/f'fold{fold}_seed{seed}.pt')
            np.testing.assert_array_equal(parent_ck['tau'], ac['tau'][fold])
            meta = dict(fold=fold, seed=seed, oldseed=oldseed, T_epoch=tepoch)
            for arm in ARMS:
                start = time.perf_counter()
                inputs = {role: (means, side_slots(hb, arm)) for role, (means, hb) in info.items()}
                xt, mu, scale = u.run.inputs(base['train'], *inputs['train'], valid[tr])
                xv, _, _ = u.run.inputs(base['validation'], *inputs['validation'], valid[va], mu, scale)
                assert not xt[..., MASKED[arm]].any() and not xv[..., MASKED[arm]].any()
                kept_common = [0, 1, 2, 7, 8, 13, 14]
                np.testing.assert_array_equal(mu[..., kept_common], parent_ck['mu'][..., kept_common])
                np.testing.assert_array_equal(scale[..., kept_common], parent_ck['scale'][..., kept_common])
                for role, (means, hb) in info.items():
                    qp, qm = side_slots(hb, 'high')[..., 0], side_slots(hb, 'low')[..., 1]
                    assert np.max(abs((qp-qm)/2-hb[..., 0])) < 1e-12
                    assert np.max(abs((qp+qm)/2-hb[..., 1])) < 1e-12
                torch.manual_seed(seed)
                model = u.run.Head('additive')
                initial = u.parent.frozen_state(model)
                assert all(torch.equal(initial[k], parent_ck['initial_state'][k]) for k in initial)
                assert sum(p.numel() for p in model.parameters()) == 266
                with torch.no_grad():
                    assert not model(xt).any() and not model(xv).any()
                best, chosen = initial, 0
                best_j = u.parent.exact_j(yv, base['validation'].argmax(1))
                armmeta = dict(**meta, arm=arm)
                curves.append(dict(**armmeta, epoch=0, train_loss=np.nan, **u.parent.assessment(yv, base['validation'])))
                bt = torch.tensor(base['train'], dtype=torch.float32)
                opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
                for epoch in range(1, 201):
                    opt.zero_grad()
                    loss = u.parent.joint_loss(bt+model(xt), ytr)
                    assert torch.isfinite(loss)
                    loss.backward()
                    opt.step()
                    if epoch % 5:
                        continue
                    with torch.no_grad():
                        score = base['validation']+model(xv).numpy().astype(float)
                    j = u.parent.exact_j(yv, score.argmax(1))
                    curves.append(dict(**armmeta, epoch=epoch, train_loss=float(loss.detach()), **u.parent.assessment(yv, score)))
                    if j > best_j:
                        best, chosen, best_j = u.parent.frozen_state(model), epoch, j
                path = HERE/'models'/f'fold{fold}_{arm}_seed{seed}.pt'
                torch.save(dict(**armmeta, state=best, initial_state=initial,
                    terminal_state=u.parent.frozen_state(model), epoch=chosen, mu=mu, scale=scale,
                    parameters=266, tau=parent_ck['tau'], calibration_is_oof=False), path)
                ck, saved = restore(path)
                for role, xx in [('train', xt), ('validation', xv)]:
                    with torch.inference_mode():
                        score = base[role]+saved(xx).numpy().astype(float)
                    if role == 'validation':
                        assert u.parent.exact_j(yv, score.argmax(1)) == best_j
                    frames.append(u.frame(table, d[role], valid, base[role], score, **armmeta, role=role, epoch=chosen))
                selections.append(dict(**armmeta, epoch=chosen, validation_J=float(best_j), checkpoint=str(path), selected_utc=u.now()))
                checks.append(dict(**armmeta, zero_slots_exact=True, common_scaler_equal=True,
                    saved_parent_initial_equal=True, zero_head=True, selected_replay=True))
                costs.append(dict(**armmeta, steps=200, seconds=time.perf_counter()-start))
                for name, rows in [('selections', selections), ('selection_curves', curves), ('checks', checks), ('costs', costs)]:
                    pd.DataFrame(rows).to_csv(HERE/f'{name}.csv', index=False)
                print(fold, seed, arm, 'epoch', chosen, 'J', float(best_j), flush=True)
    assert len(selections) == 18
    p = pd.concat(frames, ignore_index=True)
    p.to_csv(HERE/'source_predictions.csv', index=False)
    u.pooled(p).to_csv(HERE/'source_pooled.csv', index=False)
    receipt = dict(frozen_utc=u.now(), new_heads=18, steps=3600, reused_two_sided=9,
        cached_adaptation_seconds=sum(r['seconds'] for r in costs), total_seconds=time.perf_counter()-tick,
        zero_epochs=sum(r['epoch']==0 for r in selections), test_evaluated=False, gpu_extractions=0)
    save_json('selection_frozen.json', receipt)
    event('source_selection_frozen', **receipt)
    print(u.pooled(p).query("role=='validation'").to_string(index=False), flush=True)


def freeze():
    assert (HERE/'selection_frozen.json').exists() and (HERE/'SOURCE_FINDINGS_ZH.md').exists()
    assert not (HERE/'source_report_frozen.json').exists()
    receipt = dict(frozen_utc=u.now(), test_evaluated=False, source_report='SOURCE_FINDINGS_ZH.md')
    save_json('source_report_frozen.json', receipt)
    event('source_report_frozen', **receipt)


def test():
    assert (HERE/'source_report_frozen.json').exists()
    assert not (HERE/'test_predictions.csv').exists()
    tick = time.perf_counter()
    event('test_inference_start')
    torch.set_num_threads(2)
    table, cache = u.parent.load_table(), u.cache_load()
    ac = original.load_attention('test', cache)
    valid = cache['valid'].astype(bool)
    previous = parent_prediction('test')
    lookup = previous.set_index(['fold', 'seed', 'instance_id'])
    frames, checks = [previous], []
    for fold in range(3):
        with np.load(u.parent.PARENT/'baseline'/f'fold{fold}_inputs.npz') as z:
            d = {'test': z['test']}
        ix = d['test']
        means, hb = original.information(fold, d, cache, ac)['test']
        for oldseed, seed in u.strong.SEED_PAIRS:
            keys = pd.MultiIndex.from_arrays([[fold]*len(ix), [seed]*len(ix), table.iloc[ix].instance_id])
            ref = lookup.loc[keys]
            np.testing.assert_array_equal(ref.class_id, table.iloc[ix].class_id)
            b = ref[BASE].to_numpy()
            pck, pm = u.restore(HERE.parent/'models'/f'fold{fold}_seed{seed}.pt')
            xp, _, _ = u.run.inputs(b, means, hb, valid[ix], pck['mu'], pck['scale'])
            with torch.inference_mode():
                replay = b+pm(xp).numpy().astype(float)
            replay_error = float(abs(replay-ref[SCORE].to_numpy()).max())
            assert replay_error < 1e-6
            np.testing.assert_array_equal(replay.argmax(1), ref.prediction)
            for arm in ARMS:
                ck, model = restore(HERE/'models'/f'fold{fold}_{arm}_seed{seed}.pt')
                np.testing.assert_array_equal(ck['tau'], ac['tau'][fold])
                x, _, _ = u.run.inputs(b, means, side_slots(hb, arm), valid[ix], ck['mu'], ck['scale'])
                assert not x[..., MASKED[arm]].any()
                with torch.inference_mode():
                    s = b+model(x).numpy().astype(float)
                meta = dict(fold=fold, oldseed=oldseed, seed=seed, T_epoch=ck['T_epoch'], arm=arm, role='test', epoch=ck['epoch'])
                frames.append(u.frame(table, ix, valid, b, s, **meta))
                checks.append(dict(**meta, parent_replay_error=replay_error, tau_fixed=True, missing_side_zero=True, new_fits=0))
    p = pd.concat(frames, ignore_index=True)
    for _, g in p.groupby(['seed', 'arm']):
        assert len(g)==4744 and g.instance_id.is_unique
    p.to_csv(HERE/'test_predictions.csv', index=False)
    u.pooled(p).to_csv(HERE/'test_pooled.csv', index=False)
    pd.DataFrame(checks).to_csv(HERE/'test_checks.csv', index=False)
    receipt = dict(completed_utc=u.now(), new_fits=0, cached_evaluation_seconds=time.perf_counter()-tick)
    save_json('test_evaluation.json', receipt)
    event('test_inference_complete', **receipt)
    print(u.pooled(p).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['source', 'freeze_source', 'test'])
    {'source': train, 'freeze_source': freeze, 'test': test}[parser.parse_args().stage]()
