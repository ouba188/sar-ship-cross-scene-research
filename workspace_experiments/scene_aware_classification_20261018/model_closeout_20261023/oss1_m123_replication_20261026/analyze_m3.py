"""Paired OSS1 evaluation and native query/reference chip evidence."""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract_frontend import prepare_inventory, prepare
from train_m3 import metrics, write_json

ARMS = ('old_both', 'B_fixed', 'spatial', 'frequency')
CLASSES = ('bulk', 'general', 'chemical', 'container')


def margin(s, y):
    return s[np.arange(len(y)), y]-np.where(np.arange(4)[None] == y[:, None], -np.inf, s).max(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.out
    table = pd.read_csv(root/'table.csv', dtype={'mmsi': str})
    te = np.flatnonzero(table.role.eq('evaluation'))
    y = table.class_id.to_numpy(int)
    data = {int(p.name[4:]): dict(np.load(p/'predictions.npz'))
            for p in sorted(root.glob('seed*')) if (p/'predictions.npz').exists()}
    frontend_rows = []
    for seed in data:
        with np.load(HERE/'frontend'/f'scores_seed{seed}.npz') as z:
            np.testing.assert_array_equal(z['obs_id'].astype(str), table.obs_id.astype(str))
            for arm in ('base', 'm1', 's12'):
                frontend_rows.append(dict(seed=seed, arm=arm, **metrics(y[te], z[arm][te])))
    pd.DataFrame(frontend_rows).to_csv(root/'frontend_stage_metrics.csv', index=False)
    gate_rows = []
    for seed, z in data.items():
        for role, ix in table.groupby('role').indices.items():
            for arm in ('B_fixed', 'spatial', 'frequency'):
                valid = z['available'][ix].all(1)
                a = z[f'weight_{arm}'][ix, 0][valid]
                gate_rows.append(dict(seed=seed, role=role, arm=arm, paired_n=len(a),
                    vv_min=float(a.min()), vv_max=float(a.max()), vv_mean=float(a.mean()),
                    vv_std=float(a.std()), at_zero=int((a == 0).sum()), at_one=int((a == 1).sum())))
    pd.DataFrame(gate_rows).to_csv(root/'gate_statistics.csv', index=False)
    pairs = [('B_fixed', 'old_both'), ('spatial', 'old_both'), ('frequency', 'old_both'),
             ('spatial', 'B_fixed'), ('frequency', 'B_fixed'), ('frequency', 'spatial')]
    rows, classes, predictions = [], [], []
    for seed, z in data.items():
        for arm in ARMS:
            f = table.iloc[te][['obs_id', 'mmsi', 'pass_group', 'class_id']].copy()
            f['seed'], f['arm'] = seed, arm
            f['prediction'] = z[arm][te].argmax(1)
            f['correct'] = f.prediction.eq(f.class_id)
            f['margin'] = margin(z[arm][te], y[te])
            predictions.append(f)
        for arm, baseline in pairs:
            a, b = z[arm][te].argmax(1) == y[te], z[baseline][te].argmax(1) == y[te]
            d = a.astype(float)-b
            rows.append(dict(seed=seed, arm=arm, baseline=baseline, n=len(te),
                             rescued=int((a & ~b).sum()), harmed=int((b & ~a).sum()),
                             net=int(d.sum()), OA_pp=100*d.mean(),
                             BA_pp=100*np.mean([d[y[te] == c].mean() for c in range(4)])))
            for c in range(4):
                mask = y[te] == c
                classes.append(dict(seed=seed, arm=arm, baseline=baseline, class_name=CLASSES[c],
                    n=int(mask.sum()), rescued=int((a[mask] & ~b[mask]).sum()),
                    harmed=int((b[mask] & ~a[mask]).sum()), recall_pp=100*d[mask].mean()))
    pd.DataFrame(rows).to_csv(root/'paired.csv', index=False)
    pd.DataFrame(classes).to_csv(root/'by_class.csv', index=False)
    pd.concat(predictions).to_csv(root/'predictions.csv.gz', index=False, compression='gzip')
    # Class-stratified resampling conditions on this observed acquisition mix.
    rng = np.random.default_rng(1026)
    draws = [rng.choice(np.flatnonzero(y[te] == c), size=(2000, (y[te] == c).sum())) for c in range(4)]
    intervals = []
    for arm, baseline in pairs:
        d = np.mean([(z[arm][te].argmax(1) == y[te]).astype(float)-(z[baseline][te].argmax(1) == y[te])
                     for z in data.values()], 0)
        ds = [d[idx] for idx in draws]
        oa = np.concatenate(ds, 1).mean(1)*100
        ba = np.stack([v.mean(1) for v in ds], 1).mean(1)*100
        intervals.append(dict(arm=arm, baseline=baseline,
             OA_low=np.quantile(oa, .025), OA_high=np.quantile(oa, .975),
             BA_low=np.quantile(ba, .025), BA_high=np.quantile(ba, .975),
             scope='class-stratified ships within the fixed target acquisition; not cross-acquisition uncertainty'))
    pd.DataFrame(intervals).to_csv(root/'uncertainty.csv', index=False)
    records = prepare_inventory(table)
    case_specs = []
    for arm, baseline, statuses in [('B_fixed', 'old_both', ('rescue', 'harm')),
                                    ('frequency', 'B_fixed', ('rescue', 'harm', 'CC', 'WW')),
                                    ('frequency', 'spatial', ('rescue', 'harm'))]:
        for status in statuses:
            candidates = []
            for seed, z in data.items():
                b, a = z[baseline][te].argmax(1) == y[te], z[arm][te].argmax(1) == y[te]
                masks = dict(rescue=a & ~b, harm=b & ~a, CC=a & b, WW=~a & ~b)
                diff = margin(z[arm][te], y[te])-margin(z[baseline][te], y[te])
                for k in np.flatnonzero(masks[status]):
                    candidates.append((abs(diff[k]), seed, int(te[k])))
            if candidates:
                _, seed, idx = max(candidates)
                case_specs.append((arm, baseline, status, seed, idx))
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    cases = []
    for arm, baseline, status, seed, i in case_specs:
        z = data[seed]
        true = int(y[i])
        before, after = z[baseline][i], z[arm][i]
        rival = int(np.where(np.arange(4) == true, -np.inf, before).argmax())
        refs = z['refs'][i, [true, rival]].ravel()
        wa = z[f'weight_{arm}'][i]
        wb = None if baseline == 'old_both' else z[f'weight_{baseline}'][i]
        pdeltas = np.array([z['delta_vv'][i, true]-z['delta_vv'][i, rival],
                           z['delta_vh'][i, true]-z['delta_vh'][i, rival]])
        mb, ma = before[true]-before[rival], after[true]-after[rival]
        if wb is None:
            old_delta = mb-(z['base'][i, true]-z['base'][i, rival])
            contribution = np.r_[wa*pdeltas, -old_delta]
            contribution_text = f'VV={contribution[0]:+.4f}, VH={contribution[1]:+.4f}, remove old joint={contribution[2]:+.4f}'
            weight_text = f'joint readout -> VV/VH {wa.round(3)}'
        else:
            contribution = (wa-wb)*pdeltas
            contribution_text = f'VV={contribution[0]:+.4f}, VH={contribution[1]:+.4f}'
            weight_text = f'VV/VH {wb.round(3)} -> {wa.round(3)}'
        assert abs(ma-mb-contribution.sum()) < 5e-6
        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        provenance = []
        for col, j in enumerate([i, *refs]):
            prepared = prepare(records[j])
            for p, pol in enumerate(('VV', 'VH')):
                view = prepared[p]
                ax = axes[p, col]
                if view is not None:
                    ax.imshow(view['pixels'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                    g = view['geometry']
                    if g is not None:
                        cx, cy, l, w, ang = g[:5]
                        uv = np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*[l/2,w/2]
                        r = np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
                        corners = uv@r.T+[cx,cy]
                        ax.plot(corners[:,0],corners[:,1],color='#20a780',linewidth=.8)
                    provenance.append(dict(obs_id=str(table.iloc[j].obs_id), polarization=pol,
                        path=records[j][pol.lower()]['image'], placement=view['placement']))
                label = 'QUERY' if col == 0 else ('TRUE REF' if col < 3 else 'RIVAL REF')
                ax.set_title(f'{pol} {label} {table.iloc[j].obs_id}\n{CLASSES[y[j]]}', fontsize=8)
                ax.set(xticks=[], yticks=[])
        fig.suptitle(f'{arm} vs {baseline}: {status}, seed{seed}, true={CLASSES[true]}\n'
                     f'pred {CLASSES[before.argmax()]} -> {CLASSES[after.argmax()]}; margin {mb:+.4f} -> {ma:+.4f}; '
                     f'{weight_text}')
        fig.text(.02, .025, f'Native crop/pad; no resizing. Fixed-pair changes {contribution_text}. Not physical attribution.', fontsize=9)
        fig.subplots_adjust(top=.82, bottom=.09, hspace=.2, wspace=.15)
        name = f'{arm}_{baseline}_{status}_{i}_{seed}.png'
        fig.savefig(out/name, dpi=120)
        plt.close(fig)
        cases.append(dict(arm=arm, baseline=baseline, status=status, seed=seed, row=i,
            obs_id=str(table.iloc[i].obs_id), true=true, rival=rival, before_pred=int(before.argmax()),
            after_pred=int(after.argmax()), margin_before=float(mb), margin_after=float(ma),
            weights_before=None if wb is None else wb.tolist(), weights_after=wa.tolist(), contribution_change=contribution.tolist(),
            refs=refs.tolist(), reference_ids=table.iloc[refs].obs_id.astype(str).tolist(), image=name,
            provenance=provenance))
    write_json(out/'cases.json', cases)
    write_json(root/'analysis_receipt.json', dict(panels=len(cases), test_n=len(te),
               target_passes=table.iloc[te].pass_group.nunique(), raw_chip_visual_review_pending=True))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(cases).drop(columns=['provenance']).to_string(index=False))


if __name__ == '__main__':
    main()
