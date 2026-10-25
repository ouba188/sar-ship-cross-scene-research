"""Separate spectral/coverage dependence and common gain from polar preference."""
import train as run
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp


def effect(a, b, y):
    ca, cb = a.argmax(1) == y, b.argmax(1) == y
    d = ca.astype(float)-cb
    return dict(rescued=int((ca & ~cb).sum()), harmed=int((~ca & cb).sum()),
        OA_pp=100*d.mean(), BA_pp=100*np.mean([d[y == c].mean() for c in np.unique(y)]))


def main():
    data = run.Data(); rows, groups, traces = [], [], []
    ids = data.te; y = data.table.iloc[ids].class_id.to_numpy(int)
    fields = np.arange(144).reshape(2, 3, 3, 8)+41
    for seed in run.SEEDS:
        raw, z = run.inputs(data, seed)
        model, x, ck = run.restore(seed, 'frequency', raw)
        with torch.no_grad():
            s, g = run.prediction(model, x, z, np.arange(data.n))
        actual, weights = s.numpy(), g.numpy()
        saved = np.load(run.HERE / 'results' / f'frequency_{seed}.npz')
        err = float(abs(saved['score']-actual).max()); assert err < 1e-6
        available = z['available'].numpy().astype(bool)
        variants = {}
        for name, columns in (
            ('without_energy', fields[..., :4].ravel()),
            ('without_covariance', fields[..., 4:6].ravel()),
            ('without_coverage', fields[..., 6:].ravel()),
            ('without_spectral_values', fields[..., :6].ravel()),
            ('without_query_spectral', fields[..., 0].ravel())):
            changed = x.clone(); changed[..., columns] = 0
            with torch.no_grad():
                ss, gg = run.prediction(model, changed, z, np.arange(data.n))
            variants[name] = ss.numpy()
            rows.append(dict(seed=seed, mode=name, replay_error=err)
                | effect(actual[ids], ss.numpy()[ids], y) | run.metrics(y, ss.numpy()[ids]))
            for obs in (5585, 17182, 22572, 2080, 24640):
                j = int(np.flatnonzero(data.table.obs_id.eq(obs) & data.table.role.eq('evaluation'))[0])
                yy = int(data.table.iloc[j].class_id)
                rival = int(np.where(np.arange(8) == yy, -np.inf, actual[j]).argmax())
                traces.append(dict(seed=seed, obs_id=obs, mode=name, true=yy, rival=rival,
                    before_margin=float(actual[j, yy]-actual[j, rival]),
                    neutral_margin=float(ss[j, yy]-ss[j, rival]),
                    before_VV=float(g[j, 0]), before_VH=float(g[j, 1]),
                    neutral_VV=float(gg[j, 0]), neutral_VH=float(gg[j, 1]),
                    before_pred=int(actual[j].argmax()), neutral_pred=int(ss[j].argmax())))
        vv, vh = (z['base']+z[k] for k in ('delta_vv', 'delta_vh'))
        vcorrect, hcorrect = vv[ids].numpy().argmax(1) == y, vh[ids].numpy().argmax(1) == y
        paired = available[ids].all(1)
        for name, mask in (('paired_all', paired), ('VV_only_correct', paired & vcorrect & ~hcorrect),
                           ('VH_only_correct', paired & hcorrect & ~vcorrect),
                           ('VH_missing', ~available[ids, 1])):
            w = weights[ids][mask]
            corr = float(np.corrcoef(w.T)[0, 1]) if len(w) > 1 and w[:, 1].std() > 0 else None
            groups.append(dict(seed=seed, group=name, n=len(w), mean_VV=float(w[:, 0].mean()),
                mean_VH=float(w[:, 1].mean()), correlation=corr,
                VV_larger_fraction=float((w[:, 0] > w[:, 1]).mean()),
                mean_absolute_difference=float(abs(w[:, 0]-w[:, 1]).mean())))
        np.savez_compressed(run.HERE / 'results' / f'block_diagnostics_{seed}.npz', **variants, up_index=data.up[:data.n])
    pd.DataFrame(rows).to_csv(run.HERE / 'frequency_block_diagnostics.csv', index=False)
    pd.DataFrame(groups).to_csv(run.HERE / 'preference_groups.csv', index=False)
    pd.DataFrame(traces).to_csv(run.HERE / 'frequency_case_blocks.csv', index=False)
    print(pd.DataFrame(rows).groupby('mode')[['OA_pp', 'BA_pp', 'rescued', 'harmed']].mean().to_string())
    print(pd.DataFrame(groups).to_string(index=False))


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
