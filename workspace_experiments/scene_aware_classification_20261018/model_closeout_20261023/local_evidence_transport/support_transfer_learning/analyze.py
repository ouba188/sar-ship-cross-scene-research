"""Matched interventions and native raw-chip case panels, no retraining."""
import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
import run
from run import ROOT, UP, Readout, metrics
sys.path.insert(0, str(ROOT))
from extract_tokens import geo, slots


def flatten_within_reference(v, mass, bank):
    flat = v.copy()
    for p in range(2):
        for lo, hi in run.REGIONS:
            b = mass[bank, p, lo:hi].copy()
            b /= np.maximum(b.sum(-1, keepdims=True), 1e-20)
            logb = np.log(np.maximum(b, 1e-30))
            m = logsumexp(logb + v[:, :, p, lo:hi], axis=-1)
            flat[:, :, p, lo:hi] = m[..., None]
    return flat


def main(learned=('token_image', 'token_pass'), comparison='token_pass'):
    table = pd.read_csv(UP / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
    data = np.load(ROOT / 'tokens.npz')
    cache = np.load(HERE / 'kernel.npz')
    local = np.load(HERE / 'local_scores.npz')
    weights = np.load(HERE / 'weights.npz')
    idx, bank = local['idx'], local['bank']
    y = table.class_id.to_numpy()
    tr = np.flatnonzero(table.iloc[idx].role.eq('utility_train'))
    te = np.flatnonzero(table.iloc[idx].role.eq('evaluation'))
    reader = Readout(cache['kernel'][64:], data['mass'], bank, idx, np.ones((len(idx), 8, 8), bool))
    values = {key: local[key].copy() for key in local.files if key not in ('idx', 'bank')}
    decomposition = []
    regions, shares = {}, {}
    for arm in ('uniform', *learned):
        with torch.no_grad():
            s, r, share = reader(torch.tensor(weights[arm]), 'token', return_reference=True)
        error = float(np.max(np.abs(s.numpy() - local[arm])))
        assert error < 2e-6, (arm, error)
        regions[arm], shares[arm] = r.numpy(), share.numpy()
        if arm == 'uniform':
            continue
        flat = flatten_within_reference(weights[arm], data['mass'], bank)
        with torch.no_grad():
            values[arm + '_flat'] = reader(torch.tensor(flat), 'token').numpy()
        values[arm + '_mean_delta'] = local['uniform'] + (local[arm] - local['uniform'])[tr].mean(0)
        decomposition.append(dict(arm=arm, replay_error=error))
    direct, perclass = [], []
    for arm, value in values.items():
        direct.append(dict(arm=arm, **metrics(y[idx[te]], value[te])))
        correct = value[te].argmax(-1) == y[idx[te]]
        old = local['uniform'][te].argmax(-1) == y[idx[te]]
        for c in range(8):
            mask = y[idx[te]] == c
            perclass.append(dict(level='local', arm=arm, seed=0, class_id=c, n=int(mask.sum()),
                                 correct=int(correct[mask].sum()), rescued=int((correct & ~old & mask).sum()),
                                 harmed=int((~correct & old & mask).sum())))
    metrics_table = pd.read_csv(HERE / 'source_metrics.csv')
    scale = json.loads((HERE / 'receipt.json').read_text())['scale']
    interventions = []
    for seed in (17, 29, 43):
        saved = np.load(HERE / f'predictions_{seed}.npz')
        base = saved['full'][idx]
        old = saved['uniform'][idx[te]].argmax(-1) == y[idx[te]]
        for arm, value in values.items():
            parent = arm.split('_flat')[0].split('_mean_delta')[0]
            gain = float(metrics_table.query('seed == @seed and arm == @parent and role == "evaluation"').gain.iloc[0])
            score = base + gain * value / scale
            interventions.append(dict(seed=seed, arm=arm, gain=gain, **metrics(y[idx[te]], score[te])))
            correct = score[te].argmax(-1) == y[idx[te]]
            for c in range(8):
                mask = y[idx[te]] == c
                perclass.append(dict(level='full', arm=arm, seed=seed, class_id=c, n=int(mask.sum()),
                                     correct=int(correct[mask].sum()), rescued=int((correct & ~old & mask).sum()),
                                     harmed=int((~correct & old & mask).sum())))
    pd.DataFrame(direct).to_csv(HERE / 'interventions_local.csv', index=False)
    pd.DataFrame(interventions).to_csv(HERE / 'interventions_full.csv', index=False)
    pd.DataFrame(perclass).to_csv(HERE / 'class_results.csv', index=False)

    preds = np.load(HERE / 'predictions_17.npz')
    a = preds['uniform'][idx[te]].argmax(-1) == y[idx[te]]
    b = preds[comparison][idx[te]].argmax(-1) == y[idx[te]]
    cases = []
    for label, mask in [('rescued', ~a & b), ('harmed', a & ~b), ('CC', a & b), ('WW', ~a & ~b)]:
        cand = te[mask]
        if not len(cand):
            continue
        j = int(cand[np.argmin(table.iloc[idx[cand]].obs_id.to_numpy())])
        i = int(idx[j])
        true = int(y[i])
        rival = int(preds['uniform'][i].argmax()) if label == 'rescued' else int(np.argmax(np.where(np.arange(8) == true, -np.inf, preds[comparison][i])))
        strongest = shares[comparison][j].mean(0)
        refs = {c: bank[c, np.argsort(-strongest[c], kind='stable')[:2]].tolist() for c in (true, rival)}
        cases.append(dict(outcome=label, group_n=int(len(cand)), local_index=j, table_index=i,
                          obs_id=int(table.iloc[i].obs_id), true_class=true, rival_class=rival, refs=refs))
    out = HERE / 'analysis'
    out.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(out / '.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wanted = {int(table.iloc[i].obs_id) for case in cases for i in
              [case['table_index'], *case['refs'][case['true_class']], *case['refs'][case['rival_class']]]}
    snapshots = {int(s['obs_id']): s for s in json.loads((geo.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))
                 if int(s['obs_id']) in wanted}
    records, evidence = [], []
    for case in cases:
        i, j, true, rival = [case[k] for k in ('table_index', 'local_index', 'true_class', 'rival_class')]
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        fig.subplots_adjust(top=.83, bottom=.14, wspace=.15, hspace=.3)
        views = [(i, 'QUERY')] + [(v, f'TRUE REF {n+1}') for n, v in enumerate(case['refs'][true])] + [(v, f'RIVAL REF {n+1}') for n, v in enumerate(case['refs'][rival])]
        for p, pol in enumerate(('vv', 'vh')):
            for col, (ix, name) in enumerate(views):
                row = table.iloc[ix]
                pix, fp, g, placement = geo.native(table, snapshots, ix, p)
                mask = slots(g, fp)
                valid = bool(data['valid'][ix, p])
                if valid:
                    np.testing.assert_allclose(mask.sum((-1, -2)), data['mass'][ix, p], atol=2e-4, rtol=2e-5)
                ax = axes[p, col]
                ax.imshow(pix, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                if valid:
                    for (lo, hi), color in zip(run.REGIONS, ('#24a878', '#2c84c6', '#d39222')):
                        field = mask[lo:hi].sum(0)
                        if field.min() < .5 < field.max():
                            ax.contour(field, levels=[.5], colors=[color], linewidths=.7)
                ax.set(title=f'{pol.upper()} {name} | obs {int(row.obs_id)}\n{row.class_name}\n{row.port}; valid={int(valid)}',
                       xlim=(-.5,159.5), ylim=(159.5,-.5), xticks=[0,80,159], yticks=[0,80,159])
                records.append(dict(query_obs=case['obs_id'], pol=pol, obs_id=int(row.obs_id), panel=name,
                                    path=row['image_' + pol], placement=placement, valid=valid,
                                    pass_group=row.pass_group, mmsi=row.mmsi))
        lines = []
        for arm in ('full', 'uniform', *learned):
            margin = float(preds[arm][i, true] - preds[arm][i, rival])
            lines.append(f'{arm}: pred={int(preds[arm][i].argmax())}; true-rival={margin:+.5f}')
            evidence.append(dict(obs_id=case['obs_id'], outcome=case['outcome'], arm=arm,
                                 prediction=int(preds[arm][i].argmax()), true_class=true, rival=rival, margin=margin))
        fig.suptitle(f"Source evaluation | seed17 | {case['outcome']} | obs {case['obs_id']} | group n={case['group_n']}\n"
                     f'All 8 refs/class used; displaying top 2 by {comparison} average kernel contribution, not new retrieval.', fontsize=12)
        fig.text(.04, .025, '\n'.join(lines) + '\nNative160 crop/pad, no resizing; VV/VH separate, 0..255 display. Green body / blue side / amber end.', fontsize=9)
        fig.savefig(out / f"case_{case['outcome']}_{case['obs_id']}.png", dpi=130)
        plt.close(fig)
    pd.DataFrame(evidence).to_csv(out / 'case_margins.csv', index=False)
    perregion = []
    for case in cases:
        j = case['local_index']
        for arm in regions:
            for r in range(6):
                for c in (case['true_class'], case['rival_class']):
                    for k in range(8):
                        perregion.append(dict(query_obs=case['obs_id'], arm=arm, region=r, class_id=c,
                                              ref_obs=int(table.iloc[bank[c,k]].obs_id), fraction=float(shares[arm][j,r,c,k]),
                                              region_score=float(regions[arm][j,r,c])))
    pd.DataFrame(perregion).to_csv(out / 'reference_contributions.csv', index=False)
    (out / 'cases.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
    (out / 'image_records.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    (out / 'replay.json').write_text(json.dumps(decomposition, indent=2), encoding='utf-8')
    print(pd.DataFrame(direct).to_string(index=False))
    print(pd.DataFrame(cases)[['outcome','group_n','obs_id']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--factorial', action='store_true')
    args = parser.parse_args()
    if args.factorial:
        HERE = HERE / 'mask_factorial'
        main(('token_crosspos', 'token_crossneg'), 'token_crosspos')
    else:
        main()
