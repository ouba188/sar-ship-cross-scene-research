"""Paired source outcomes and numerical evidence, without further fitting."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UP = ROOT.parent / 'regional_utility_pilot/upstream'
ARMS = ('fixed', 'uncoupled_fixed', 'uncoupled', 'coupled')


def main():
    table = pd.read_csv(UP / 'table.csv', dtype={'mmsi': str})
    cache = np.load(HERE / 'scores.npz'); idx = cache['idx']
    evaluation = np.flatnonzero(table.iloc[idx].role.eq('evaluation'))
    rows = idx[evaluation]; y = table.iloc[rows].class_id.to_numpy()
    metrics = pd.read_csv(HERE / 'source_metrics.csv')
    results = metrics.query("role == 'evaluation'").groupby('arm')[['OA', 'BA', 'NLL', 'Brier']].agg(['mean', 'std'])
    results.to_csv(HERE / 'summary.csv')
    pairs = []; groups = []; margins = []
    for seed in (17, 29, 43):
        pred = np.load(HERE / f'predictions_{seed}.npz')
        for arm in ARMS:
            for reference in ('full', 'fixed', 'uncoupled'):
                if arm == reference:
                    continue
                a = pred[arm][rows].argmax(-1) == y
                b = pred[reference][rows].argmax(-1) == y
                delta = a.astype(int) - b.astype(int)
                pairs.append(dict(seed=seed, arm=arm, reference=reference,
                                  rescued=int((delta == 1).sum()), harmed=int((delta == -1).sum()),
                                  net=int(delta.sum()), OA_delta=float(delta.mean()),
                                  BA_delta=float(np.mean([delta[y == c].mean() for c in range(8)]))))
                for field in ('class_name', 'port'):
                    values = table.iloc[rows][field].to_numpy()
                    for label in np.unique(values):
                        use = values == label
                        groups.append(dict(seed=seed, arm=arm, reference=reference, field=field,
                                           label=label, n=int(use.sum()), net=int(delta[use].sum()),
                                           rescued=int((delta[use] == 1).sum()), harmed=int((delta[use] == -1).sum())))
    valid = cache['available'][evaluation]
    a = cache['baseline_mass'][evaluation]
    w = cache['coupled_weights'][evaluation]
    logratio = np.log(np.maximum(w, 1e-30)) - np.log(np.maximum(a[:, None], 1e-30))
    kl = (w * logratio).sum(-1)
    neff = 1 / np.maximum((w ** 2).sum(-1), 1e-20)
    base_neff = 1 / np.maximum((a ** 2).sum(-1), 1e-20)
    diagnostic = []
    for p, pol in enumerate(('VV', 'VH')):
        for r, region in enumerate(('body', 'side', 'end')):
            mask = valid[:, :, p, r]
            fixed = cache['fixed'][evaluation, :, p, r]
            coupled = cache['coupled'][evaluation, :, p, r]
            truth = np.arange(len(y)), y
            wrong = np.arange(8)[None] != y[:, None]
            fm = fixed[truth] - np.where(wrong, fixed, -np.inf).max(-1)
            cm = coupled[truth] - np.where(wrong, coupled, -np.inf).max(-1)
            has = mask.all(-1)
            diagnostic.append(dict(pol=pol, region=region, candidate_regions=int(mask.sum()),
                                   mean_KL=float(kl[:, :, p, r][mask].mean()),
                                   mean_effective_slots=float(neff[:, :, p, r][mask].mean()),
                                   mean_baseline_effective_slots=float(np.broadcast_to(base_neff[:, None, p, r], mask.shape)[mask].mean()),
                                   max_weight_ratio=float((w[:, :, p, r] / np.maximum(a[:, None, p, r], 1e-20)).max()),
                                   mean_score_increase=float((coupled - fixed)[mask].mean()),
                                   fully_available_queries=int(has.sum()),
                                   true_margin_increased=int(((cm - fm > 1e-6) & has).sum()),
                                   true_margin_decreased=int(((cm - fm < -1e-6) & has).sum()),
                                   mean_true_margin_change=float((cm - fm)[has].mean())))
            for j in range(len(y)):
                margins.append(dict(obs_id=int(table.iloc[rows[j]].obs_id), pol=pol, region=region,
                                    fully_available=bool(has[j]), fixed_true_margin=float(fm[j]),
                                    coupled_true_margin=float(cm[j]), margin_change=float(cm[j] - fm[j])))
    pd.DataFrame(pairs).to_csv(HERE / 'paired_results.csv', index=False)
    pd.DataFrame(groups).to_csv(HERE / 'group_results.csv', index=False)
    pd.DataFrame(diagnostic).to_csv(HERE / 'weight_diagnostics.csv', index=False)
    pd.DataFrame(margins).to_csv(HERE / 'regional_margins.csv', index=False)
    print(results.to_string())
    print(pd.DataFrame(pairs).query("arm == 'coupled'").to_string(index=False))
    print(pd.DataFrame(diagnostic).to_string(index=False))


if __name__ == '__main__':
    main()
