"""Source fitting and frozen historical-target evaluation are separate commands."""
from common import *
import argparse
import time


def information(fold, d, cache, attention_cache):
    names = list(cache['reader_names'])
    ri = names.index(f'fold{fold}_outer')
    result = {}
    for role, ix in d.items():
        if role != 'train':
            assert attention_cache['computed'][ri, ix].all()
            result[role] = (cache['mean_pol'][ri, ix], attention_cache['attention_pol'][ri, ix])
        else:
            rows = {int(i): k for k, i in enumerate(ix)}
            a = np.zeros((len(ix), 2, 8, 2))
            mu = np.zeros((len(ix), 2, 8))
            count = np.zeros(len(ix), int)
            for inner in range(4):
                j = names.index(f'fold{fold}_inner{inner}')
                with np.load(parent.PARENT/'baseline/models'/f'fold{fold}_inner{inner}.npz') as z:
                    held = z['held']
                assert attention_cache['computed'][j, held].all()
                loc = [rows[int(i)] for i in held]
                a[loc] = attention_cache['attention_pol'][j, held]
                mu[loc] = cache['mean_pol'][j, held]
                count[loc] += 1
            assert (count == 1).all()
            result[role] = (mu, a)
    return result


def load_attention(stage, cache):
    z = dict(np.load(HERE/f'attention_{stage}.npz'))
    for k in ['instance_ids', 'reader_names', 'valid']:
        np.testing.assert_array_equal(z[k], cache[k])
    return z


def source_train():
    tick = time.perf_counter()
    assert (HERE/'PROTOCOL_ZH.md').exists()
    assert not (HERE/'selection_frozen.json').exists()
    (HERE/'models').mkdir(exist_ok=True)
    assert not list((HERE/'models').glob('*.pt'))
    event('source_training_start')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    table, cache = parent.load_table(), cache_load()
    ac = load_attention('source', cache)
    valid = cache['valid'].astype(bool)
    old = pd.read_csv(run.OLD/'conditional_evidence_pilot/cohort.csv', dtype={'mmsi': str})
    reference = read_reference('validation').set_index(['fold', 'seed', 'instance_id'])
    selections, curves, costs, checks, frames = [], [], [], [], []
    for fold in range(3):
        d, original_info = source.load_source(fold, table, cache)
        info = information(fold, d, cache, ac)
        tr, va = d['train'], d['validation']
        for oldseed, seed in strong.SEED_PAIRS:
            start = time.perf_counter()
            base, tepoch = source.source_baseline(fold, oldseed, seed, table, d, old)
            xt, mu, scale = run.inputs(base['train'], *info['train'], valid[tr])
            xv, _, _ = run.inputs(base['validation'], *info['validation'], valid[va], mu, scale)
            full_ck, full = run.restore(HERE.parent/'models'/f'fold{fold}_additive_seed{seed}.pt')
            retained = [0, 1, 2, 7, 8, 13, 14]
            np.testing.assert_array_equal(mu[..., retained], full_ck['mu'][..., retained])
            np.testing.assert_array_equal(scale[..., retained], full_ck['scale'][..., retained])
            meta = dict(fold=fold, oldseed=oldseed, seed=seed, T_epoch=tepoch)
            for role, ix in d.items():
                xf, _, _ = run.inputs(base[role], *original_info[role], valid[ix], full_ck['mu'], full_ck['scale'])
                with torch.inference_mode():
                    fs = base[role]+full(xf).numpy().astype(float)
                if role == 'validation':
                    keys = pd.MultiIndex.from_arrays([[fold]*len(ix), [seed]*len(ix), table.iloc[ix].instance_id])
                    ref = reference.loc[keys]
                    base_error = float(abs(base[role]-ref[BASE].to_numpy()).max())
                    full_error = float(abs(fs-ref[SCORE].to_numpy()).max())
                    assert base_error < 1e-12 and full_error < 1e-6
                    np.testing.assert_array_equal(fs.argmax(1), ref.prediction)
                frames.append(frame(table, ix, valid, base[role], fs, **meta, role=role, arm='full', epoch=full_ck['epoch']))
            torch.manual_seed(seed)
            model = run.Head('additive')
            initial = parent.frozen_state(model)
            torch.manual_seed(seed)
            reference_init = run.Head('additive').state_dict()
            assert all(torch.equal(initial[k], reference_init[k]) for k in initial)
            assert sum(p.numel() for p in model.parameters()) == 266
            with torch.no_grad():
                assert not model(xt).any() and not model(xv).any()
            best, chosen = initial, 0
            ytr = torch.tensor(table.iloc[tr].class_id.to_numpy(), dtype=torch.long)
            yv = table.iloc[va].class_id.to_numpy()
            bt = torch.tensor(base['train'], dtype=torch.float32)
            best_j = parent.exact_j(yv, base['validation'].argmax(1))
            curves.append(dict(**meta, arm=ARM, epoch=0, train_loss=np.nan,
                **parent.assessment(yv, base['validation'])))
            opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
            for epoch in range(1, 201):
                opt.zero_grad()
                loss = parent.joint_loss(bt+model(xt), ytr)
                assert torch.isfinite(loss)
                loss.backward()
                opt.step()
                if epoch % 5:
                    continue
                with torch.no_grad():
                    s = base['validation']+model(xv).numpy().astype(float)
                j = parent.exact_j(yv, s.argmax(1))
                curves.append(dict(**meta, arm=ARM, epoch=epoch, train_loss=float(loss.detach()),
                    **parent.assessment(yv, s)))
                if j > best_j:
                    best, chosen, best_j = parent.frozen_state(model), epoch, j
            path = HERE/'models'/f'fold{fold}_seed{seed}.pt'
            torch.save(dict(**meta, arm=ARM, state=best, initial_state=initial,
                terminal_state=parent.frozen_state(model), epoch=chosen, mu=mu, scale=scale,
                tau=ac['tau'][fold], parameters=266, calibration_is_oof=False), path)
            ck, saved = restore(path)
            for role, xx in [('train', xt), ('validation', xv)]:
                with torch.inference_mode():
                    s = base[role]+saved(xx).numpy().astype(float)
                if role == 'validation':
                    assert parent.exact_j(yv, s.argmax(1)) == best_j
                frames.append(frame(table, d[role], valid, base[role], s, **meta, role=role, arm=ARM, epoch=chosen))
            selections.append(dict(**meta, epoch=chosen, validation_J=float(best_j), checkpoint=str(path), selected_utc=now()))
            costs.append(dict(**meta, seconds=time.perf_counter()-start, steps=200))
            checks.append(dict(**meta, baseline_error=base_error, full_replay_error=full_error,
                common_scaler_equal=True, initial_parameters_equal=True, zero_residual=True, selected_replay=True))
            print(fold, seed, 'epoch', chosen, 'J', float(best_j), flush=True)
            for name, data in [('selections', selections), ('selection_curves', curves), ('costs', costs), ('checks', checks)]:
                pd.DataFrame(data).to_csv(HERE/f'{name}.csv', index=False)
    assert len(selections) == 9
    pred = pd.concat(frames, ignore_index=True)
    pred.to_csv(HERE/'source_predictions.csv', index=False)
    pooled(pred).to_csv(HERE/'source_pooled.csv', index=False)
    receipt = dict(frozen_utc=now(), heads=9, new_steps=1800, test_evaluated=False,
        test_scores_read=False, tau_frozen=True, calibration_is_oof=False,
        train_seconds=sum(x['seconds'] for x in costs), total_seconds=time.perf_counter()-tick,
        zero_epochs=sum(x['epoch'] == 0 for x in selections))
    write_json('selection_frozen.json', receipt)
    event('source_selection_frozen', **receipt)
    print(pooled(pred).query("role=='validation'").to_string(index=False), flush=True)


def freeze_source():
    assert (HERE/'selection_frozen.json').exists()
    assert (HERE/'SOURCE_FINDINGS_ZH.md').exists()
    assert (HERE/'SOURCE_IMAGE_NOTES_ZH.md').exists()
    assert not (HERE/'source_report_frozen.json').exists()
    receipt = dict(frozen_utc=now(), source_report='SOURCE_FINDINGS_ZH.md',
        raw_review='SOURCE_IMAGE_NOTES_ZH.md', test_evaluated=False,
        authorization='Both stages authorized before implementation by current main workflow; no new target-stage fit.')
    write_json('source_report_frozen.json', receipt)
    event('source_report_frozen', **receipt)


def test_evaluate():
    assert (HERE/'source_report_frozen.json').exists()
    assert not (HERE/'test_predictions.csv').exists()
    tick = time.perf_counter()
    event('test_inference_start')
    torch.set_num_threads(2)
    table, cache = parent.load_table(), cache_load()
    ac = load_attention('test', cache)
    valid = cache['valid'].astype(bool)
    reference = read_reference('test').set_index(['fold', 'seed', 'instance_id'])
    frames, checks = [], []
    for fold in range(3):
        with np.load(parent.PARENT/'baseline'/f'fold{fold}_inputs.npz') as z:
            d = {'test': z['test']}
        ix = d['test']
        info = information(fold, d, cache, ac)['test']
        ri = list(cache['reader_names']).index(f'fold{fold}_outer')
        for oldseed, seed in strong.SEED_PAIRS:
            keys = pd.MultiIndex.from_arrays([[fold]*len(ix), [seed]*len(ix), table.iloc[ix].instance_id])
            ref = reference.loc[keys]
            np.testing.assert_array_equal(ref.class_id, table.iloc[ix].class_id)
            base = ref[BASE].to_numpy()
            ck, model = restore(HERE/'models'/f'fold{fold}_seed{seed}.pt')
            np.testing.assert_array_equal(ck['tau'], ac['tau'][fold])
            x, _, _ = run.inputs(base, *info, valid[ix], ck['mu'], ck['scale'])
            full_ck, full = run.restore(HERE.parent/'models'/f'fold{fold}_additive_seed{seed}.pt')
            xf, _, _ = run.inputs(base, cache['mean_pol'][ri, ix], cache['moments_pol'][ri, ix], valid[ix], full_ck['mu'], full_ck['scale'])
            with torch.inference_mode():
                s = base+model(x).numpy().astype(float)
                fs = base+full(xf).numpy().astype(float)
            err = float(abs(fs-ref[SCORE].to_numpy()).max())
            assert err < 1e-6
            np.testing.assert_array_equal(fs.argmax(1), ref.prediction)
            for arm, score, epoch in [(ARM, s, ck['epoch']), ('full', fs, full_ck['epoch'])]:
                frames.append(frame(table, ix, valid, base, score, fold=fold, seed=seed, oldseed=oldseed,
                    role='test', arm=arm, epoch=epoch, T_epoch=ck['T_epoch']))
            checks.append(dict(fold=fold, seed=seed, n=len(ix), full_replay_error=err, fixed_tau=True, new_fits=0))
    pred = pd.concat(frames, ignore_index=True)
    for _, g in pred.groupby(['arm', 'seed']):
        assert len(g) == 4744 and g.instance_id.is_unique
    pred.to_csv(HERE/'test_predictions.csv', index=False)
    pooled(pred).to_csv(HERE/'test_pooled.csv', index=False)
    pd.DataFrame(checks).to_csv(HERE/'test_checks.csv', index=False)
    receipt = dict(completed_utc=now(), seconds=time.perf_counter()-tick, new_fits=0,
        objects=4744, epochs_changed=False, temperature_changed=False)
    write_json('test_evaluation.json', receipt)
    event('test_inference_complete', **receipt)
    print(pooled(pred).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['source', 'freeze_source', 'test'])
    stage = parser.parse_args().stage
    {'source': source_train, 'freeze_source': freeze_source, 'test': test_evaluate}[stage]()
