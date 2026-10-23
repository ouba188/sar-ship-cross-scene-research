"""Reuse label-free B/mean donor matches for the frozen attention readout."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as attention
run = attention.run


def main():
    torch.set_num_threads(2)
    table = run.parent.load_table()
    cache = attention.cache_load()
    valid = cache['valid'].astype(bool)
    loc = pd.Series(np.arange(len(table)), index=table.instance_id)
    donors = pd.read_csv(run.HERE/'conditional_distribution_diagnosis/predictions.csv',
        dtype={'mmsi': str, 'donor_mmsi': str}, float_precision='round_trip')
    donors = donors[donors.match.eq('B_and_mean')].copy()
    source_cache = dict(np.load(HERE.parent/'attention_source.npz'))
    test_cache = dict(np.load(HERE.parent/'attention_test.npz'))
    for ac in (source_cache, test_cache):
        for name in ('instance_ids', 'reader_names', 'valid'):
            np.testing.assert_array_equal(ac[name], cache[name])
    references = {}
    for role, name in [('validation', 'source'), ('test', 'test')]:
        frame = pd.read_csv(HERE.parent/f'{name}_predictions.csv',
            dtype={'mmsi': str}, float_precision='round_trip')
        references[role] = frame[frame.role.eq(role) & frame.arm.eq(attention.ARM)].set_index(
            ['fold', 'seed', 'instance_id'])
    rows, checks = [], []
    for (fold, seed, role), part in donors.groupby(['fold', 'seed', 'role']):
        part = part.copy()
        ix = loc.loc[part.instance_id].to_numpy()
        di = loc.loc[part.donor_id].to_numpy()
        ri = list(cache['reader_names']).index(f'fold{fold}_outer')
        ac = source_cache if role == 'validation' else test_cache
        assert ac['computed'][ri, ix].all()
        eligible = part.eligible.to_numpy(bool)
        assert source_cache['computed'][ri, di[eligible]].all()
        base = part[[f'b{c}' for c in range(8)]].to_numpy()
        means = cache['mean_pol'][ri, ix]
        own_hb = ac['attention_pol'][ri, ix]
        borrowed_hb = own_hb.copy()
        borrowed_hb[eligible] = source_cache['attention_pol'][ri, di[eligible]]
        ck, model = attention.restore(HERE.parent/'models'/f'fold{fold}_seed{seed}.pt')
        scores = []
        for hb in (own_hb, borrowed_hb):
            x, _, _ = run.inputs(base, means, hb, valid[ix], ck['mu'], ck['scale'])
            with torch.inference_mode():
                scores.append(base + model(x).numpy().astype(float))
        own, borrowed = scores
        keys = pd.MultiIndex.from_arrays([[fold]*len(ix), [seed]*len(ix), part.instance_id])
        ref = references[role].loc[keys]
        base_error = float(np.max(abs(base-ref[attention.BASE].to_numpy())))
        own_error = float(np.max(abs(own-ref[attention.SCORE].to_numpy())))
        assert base_error < 1e-12 and own_error < 1e-6
        np.testing.assert_array_equal(own.argmax(1), ref.prediction)
        np.testing.assert_array_equal(valid[ix[eligible]], valid[di[eligible]])
        assert not (part.loc[eligible].mmsi == part.loc[eligible].donor_mmsi).any()
        assert not (part.loc[eligible].pass_group == part.loc[eligible].donor_pass).any()
        part['epoch'] = ck['epoch']
        part['own_prediction'] = own.argmax(1)
        part['borrowed_prediction'] = borrowed.argmax(1)
        part['net_own_vs_borrowed'] = (part.own_prediction.eq(part.class_id).astype(int)
            - part.borrowed_prediction.eq(part.class_id).astype(int))
        part['outcome'] = np.select([part.net_own_vs_borrowed.eq(1), part.net_own_vs_borrowed.eq(-1),
            part.own_prediction.eq(part.class_id)], ['rescued', 'harmed', 'CC'], default='WW')
        for c in range(8):
            part[f'own{c}'], part[f'borrowed{c}'] = own[:, c], borrowed[:, c]
        rows.append(part)
        checks.append(dict(fold=fold, seed=seed, role=role, n=len(ix), eligible=int(eligible.sum()),
            baseline_error=base_error, own_replay_error=own_error, epoch=ck['epoch'],
            donor_matches_reused=True, B_mean_valid_frozen=True))
    pred = pd.concat(rows, ignore_index=True)
    pred.to_csv(HERE/'predictions.csv', index=False)
    metrics, classes = [], []
    for (role, seed), g in pred.groupby(['role', 'seed']):
        y = g.class_id.to_numpy()
        a = run.parent.assessment(y, g[[f'own{c}' for c in range(8)]].to_numpy())
        b = run.parent.assessment(y, g[[f'borrowed{c}' for c in range(8)]].to_numpy())
        metrics.append(dict(role=role, seed=seed, n=len(g), net=int(g.net_own_vs_borrowed.sum()),
            rescued=int(g.net_own_vs_borrowed.eq(1).sum()), harmed=int(g.net_own_vs_borrowed.eq(-1).sum()),
            **{f'own_{k}': a[k] for k in ['OA', 'BA', 'NLL', 'Brier']},
            **{f'borrowed_{k}': b[k] for k in ['OA', 'BA', 'NLL', 'Brier']},
            **{f'delta_{k}': a[k]-b[k] for k in ['OA', 'BA', 'NLL', 'Brier']}))
        for c, h in g.groupby('class_id'):
            classes.append(dict(role=role, seed=seed, class_id=c, n=len(h),
                net=int(h.net_own_vs_borrowed.sum())))
    pd.DataFrame(metrics).to_csv(HERE/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(HERE/'classes.csv', index=False)
    pd.DataFrame(checks).to_csv(HERE/'checks.csv', index=False)
    (HERE/'summary.json').write_text(json.dumps(dict(new_fits=0, rows=len(pred),
        matching='prior frozen B_and_mean donor mapping', exact_condition_match=False,
        max_replay_error=max(x['own_replay_error'] for x in checks)), indent=2), encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == '__main__':
    main()
