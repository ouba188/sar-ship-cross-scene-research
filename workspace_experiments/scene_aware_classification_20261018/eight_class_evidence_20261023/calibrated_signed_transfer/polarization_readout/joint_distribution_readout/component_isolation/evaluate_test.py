"""Once-only authorized fixed-checkpoint test inference; no fitting."""
import json
import time
import numpy as np
import pandas as pd
import torch
import run_components as fit

HERE, parent, run, previous = fit.HERE, fit.parent, fit.run, fit.previous


def read_reference(path, arm):
    roles = pd.read_csv(path, usecols=['role', 'arm'])
    keep = set(np.flatnonzero(roles.role.eq('test') & roles.arm.eq(arm)) + 1)
    return pd.read_csv(path, skiprows=lambda i: i != 0 and i not in keep,
                       dtype={'mmsi': str}, float_precision='round_trip').set_index(['fold', 'seed', 'instance_id'])


def main():
    start = time.perf_counter()
    assert (HERE / 'TEST_ADDENDUM_ZH.md').exists()
    assert (HERE / 'source_review_cases.csv').exists()
    assert not (HERE / 'test_predictions.csv').exists(), 'Retain once-only test output.'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    selected = pd.read_csv(HERE / 'selections.csv').set_index(['fold', 'seed', 'arm'])
    assert len(selected) == 18 and json.loads((HERE / 'selection_frozen.json').read_text())['heads'] == 18
    table = parent.load_table()
    old = pd.read_csv(run.OLD / 'conditional_evidence_pilot/cohort.csv', dtype={'mmsi': str})
    mapping = old.reset_index().set_index('obs_id')['index'].loc[table.obs_id].to_numpy()
    with np.load(parent.HERE / 'signed_features.npz') as z:
        valid, names = z['valid'].astype(bool), list(z['reader_names'])
        means, moments = z['mean_pol'], z['moments_pol']
        np.testing.assert_array_equal(z['instance_ids'], table.instance_id)
    full = read_reference(HERE.parent / 'predictions.csv', 'additive')
    mean = read_reference(fit.mean_run.HERE / 'predictions.csv', 'mean_only')
    frames, checks = [], []
    for fold in range(3):
        with np.load(parent.PARENT / 'baseline' / f'fold{fold}_inputs.npz') as z:
            ix = z['test']
        oi = mapping[ix]
        np.testing.assert_array_equal(old.iloc[oi].instance_id, table.iloc[ix].instance_id)
        reader = names.index(f'fold{fold}_outer')
        m, mom = means[reader, ix], moments[reader, ix]
        for oldseed, seed in fit.strong.SEED_PAIRS:
            with np.load(run.OLD / 'geometry_evidence_gate_20261021' / f'inputs_f{fold}_s{oldseed}.npz') as z:
                raw = (z['base'][oi] + z['residual'][oi]).astype(float)
            tc, tm = previous.restore(previous.HERE / 'models' / f'fold{fold}_score_only_seed{seed}.pt')
            xt, _, _, _ = previous.inputs('score_only', raw, None, None, tc['mu'], tc['scale'])
            with torch.inference_mode():
                base = raw + tm(xt, None).numpy().astype(float)
            keys = pd.MultiIndex.from_arrays([[fold]*len(ix), [seed]*len(ix), table.iloc[ix].instance_id])
            fg, mg = full.loc[keys], mean.loc[keys]
            err = float(np.abs(base-fg[fit.BASE].to_numpy()).max())
            assert err < 1e-12 and np.abs(base-mg[fit.BASE].to_numpy()).max() < 1e-12
            meta = dict(fold=fold, oldseed=oldseed, seed=seed, role='test', T_epoch=tc['epoch'])
            for arm, g in [('full', fg), ('mean_only', mg)]:
                np.testing.assert_array_equal(g.class_id, table.iloc[ix].class_id)
                assert g.epoch.nunique() == 1
                frames.append(fit.predict_frame(table, ix, valid, base, g[fit.SCORE].to_numpy(),
                    dict(**meta, arm=arm, epoch=int(g.epoch.iloc[0]))))
            frames.append(fit.predict_frame(table, ix, valid, base, base, dict(**meta, arm='B', epoch=0)))
            for arm in fit.ARMS:
                ck, model = fit.restore(HERE / 'models' / f'fold{fold}_{arm}_seed{seed}.pt')
                assert ck['epoch'] == selected.loc[(fold, seed, arm), 'epoch']
                assert ck['masked_columns'] == fit.MASKED[arm]
                x, _, _ = run.inputs(base, m, mom, valid[ix], ck['mu'], ck['scale'])
                with torch.inference_mode():
                    score = base + model(fit.mask_slots(x, arm)).numpy().astype(float)
                frames.append(fit.predict_frame(table, ix, valid, base, score,
                    dict(**meta, arm=arm, epoch=ck['epoch'])))
                checks.append(dict(fold=fold, seed=seed, arm=arm, epoch=ck['epoch'], n=len(ix),
                    B_replay_error=err, source_selected_unchanged=True, trained=False))
    p = pd.concat(frames, ignore_index=True)
    p.to_csv(HERE / 'test_predictions.csv', index=False)
    pd.DataFrame(checks).to_csv(HERE / 'test_checks.csv', index=False)
    rows = []
    for (arm, seed), g in p.groupby(['arm', 'seed']):
        assert len(g) == 4744 and g.instance_id.is_unique
        a = parent.assessment(g.class_id.to_numpy(), g[fit.SCORE].to_numpy())
        b = parent.assessment(g.class_id.to_numpy(), g[fit.BASE].to_numpy())
        rows.append(dict(role='test', arm=arm, seed=seed, **a, net_B=int(g.net.sum()),
            **{f'delta_{k}_B': a[k]-b[k] for k in ['OA', 'BA', 'NLL', 'Brier']}))
    pooled = pd.DataFrame(rows)
    pooled.to_csv(HERE / 'test_pooled.csv', index=False)
    summary = dict(new_fits=0, fixed_heads=18, test_n_per_seed=4744,
        cached_inference_and_io_seconds=time.perf_counter()-start,
        zero_epoch_heads=int(sum(c['epoch']==0 for c in checks)),
        B_replay_max_error=max(c['B_replay_error'] for c in checks),
        after_source_authorization=True, blind_test=False, source_selection_unchanged=True)
    (HERE / 'test_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pooled[['arm','seed','net_B','OA','BA','delta_OA_B','delta_BA_B','NLL','Brier']].to_string(index=False), flush=True)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
