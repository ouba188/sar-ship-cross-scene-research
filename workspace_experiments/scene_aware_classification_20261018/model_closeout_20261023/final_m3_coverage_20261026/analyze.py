"""Paired group/class analysis and native query/reference review panels."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluate as run


def margin(score, y):
    return score[np.arange(len(y)), y]-np.where(np.arange(8)[None] == y[:, None], -np.inf, score).max(1)


def main():
    table = pd.read_csv(HERE/'table.csv', dtype={'mmsi': str})
    y = table.class_id.to_numpy(int)
    data = {s: dict(np.load(HERE/'results'/f'seed{s}.npz')) for s in run.SEEDS}
    comparisons = [('B_fixed', 'old_both'), ('spatial', 'old_both'), ('frequency', 'old_both'),
                   ('spatial', 'B_fixed'), ('frequency', 'B_fixed'), ('frequency', 'spatial')]
    rows, predictions, classrows, group_rows, strata = [], [], [], [], []
    for seed, z in data.items():
        for arm in run.ARMS:
            f = table[['obs_id', 'mmsi', 'class_id', 'class_name', 'pass_group', 'port']].copy()
            f['seed'], f['arm'] = seed, arm
            f['prediction'] = z[arm].argmax(1)
            f['correct'] = f.prediction.eq(f.class_id)
            f['margin'] = margin(z[arm], y)
            predictions.append(f)
            for paired in (True, False):
                mask = z['available'].all(1) == paired
                if mask.any():
                    strata.append(dict(seed=seed, arm=arm, paired_vv_vh=paired,
                                       **run.metrics(y[mask], z[arm][mask])))
        for arm, baseline in comparisons:
            before, after = z[baseline].argmax(1) == y, z[arm].argmax(1) == y
            d = after.astype(float)-before
            rows.append(dict(seed=seed, arm=arm, baseline=baseline, rescued=int((after & ~before).sum()),
                harmed=int((before & ~after).sum()), net=int(d.sum()), OA_pp=100*d.mean(),
                BA_pp=100*np.mean([d[y == c].mean() for c in range(8)])))
            for c in range(8):
                mask = y == c
                classrows.append(dict(seed=seed, arm=arm, baseline=baseline, class_id=c,
                    class_name=table.loc[mask, 'class_name'].iloc[0], n=int(mask.sum()),
                    rescued=int((after[mask] & ~before[mask]).sum()), harmed=int((before[mask] & ~after[mask]).sum()),
                    recall_pp=100*d[mask].mean()))
            for field in ('pass_group', 'port'):
                for group, ids in table.groupby(field).groups.items():
                    ids = np.asarray(ids)
                    group_rows.append(dict(seed=seed, arm=arm, baseline=baseline, field=field,
                        group=group, n=len(ids), OA_pp=100*d[ids].mean()))
    pd.DataFrame(rows).to_csv(HERE/'paired.csv', index=False)
    pd.DataFrame(classrows).to_csv(HERE/'by_class.csv', index=False)
    pd.DataFrame(group_rows).to_csv(HERE/'by_group.csv', index=False)
    pd.DataFrame(strata).to_csv(HERE/'by_polar_availability.csv', index=False)
    pd.concat(predictions).to_csv(HERE/'predictions.csv.gz', index=False, compression='gzip')
    table.groupby(['class_id', 'class_name']).agg(n=('obs_id', 'size'), ships=('mmsi', 'nunique'),
        passes=('pass_group', 'nunique')).to_csv(HERE/'coverage.csv')
    groups = list(table.groupby('pass_group').indices.values())
    counts = np.array([np.bincount(y[g], minlength=8) for g in groups])
    draws = np.random.default_rng(1026).integers(len(groups), size=(2000, len(groups)))
    ct = counts[draws].sum(1)
    complete = (ct > 0).all(1)
    intervals = []
    for arm, baseline in comparisons:
        d = np.mean([(z[arm].argmax(1) == y).astype(float)-(z[baseline].argmax(1) == y) for z in data.values()], 0)
        sums = np.array([np.bincount(y[g], weights=d[g], minlength=8) for g in groups])
        sm = sums[draws].sum(1)
        oa = 100*sm.sum(1)/ct.sum(1)
        ba = 100*(sm[complete]/ct[complete]).mean(1)
        intervals.append(dict(arm=arm, baseline=baseline, OA_low=float(np.quantile(oa, .025)),
            OA_high=float(np.quantile(oa, .975)), BA_low=float(np.quantile(ba, .025)),
            BA_high=float(np.quantile(ba, .975)), complete_draws=int(complete.sum())))
    pd.DataFrame(intervals).to_csv(HERE/'uncertainty.csv', index=False)
    # Deterministic post-prediction case selection is explanatory, not model selection.
    cases = []
    for arm, baseline, statuses in [('frequency', 'B_fixed', ('rescue', 'harm', 'CC', 'WW')),
                                    ('spatial', 'B_fixed', ('rescue', 'harm')),
                                    ('frequency', 'spatial', ('rescue', 'harm'))]:
        for status in statuses:
            candidates = []
            for seed, z in data.items():
                b, a = z[baseline].argmax(1) == y, z[arm].argmax(1) == y
                mask = {'rescue': a & ~b, 'harm': b & ~a, 'CC': a & b, 'WW': ~a & ~b}[status]
                delta = margin(z[arm], y)-margin(z[baseline], y)
                for j in np.flatnonzero(mask):
                    candidates.append((abs(delta[j]), seed, int(j)))
            if candidates:
                _, seed, j = max(candidates)
                cases.append(dict(arm=arm, baseline=baseline, status=status, seed=seed, index=j))
    ex = run.load('final_review_native', run.COV/'extract.py')
    snapshots = {int(v['obs_id']): v for v in json.loads((ex.joint.geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    source = pd.read_csv(run.UP/'table.csv', dtype={'mmsi': str})
    bank = np.load(run.PACKAGE/'assets/m3/bank.npz')
    st = np.load(run.LOCAL/'tokens.npz')
    tt = np.load(run.EXT/'tokens.npz')
    out = HERE/'analysis'
    out.mkdir(exist_ok=True)
    reviewed = []
    for case in cases:
        j, seed, arm, baseline = (case[k] for k in ('index', 'seed', 'arm', 'baseline'))
        z = data[seed]
        true = int(y[j])
        before, after = z[baseline][j], z[arm][j]
        rival = int(np.where(np.arange(8) == true, -np.inf, before).argmax())
        refs = bank['source_index'][z['refs'][j, [true, rival]]].ravel()
        fig, ax = plt.subplots(2, 5, figsize=(16, 7))
        provenance = []
        for p, pol in enumerate(('VV', 'VH')):
            for k, i in enumerate([j, *refs]):
                f, tok = (table, tt) if k == 0 else (source, st)
                pix, fp, g, placement = ex.joint.geo.native(f, snapshots, int(i), p)
                valid = bool(tok['valid'][i, p])
                ax[p, k].imshow(pix, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                if valid and g is not None:
                    mask = ex.joint.geo.soft_regions(g)[0]*fp
                    if mask.max() > .5:
                        ax[p, k].contour(mask, levels=[.5], colors=['#20a780'], linewidths=.7)
                label = 'QUERY' if k == 0 else ('TRUE REF' if k < 3 else 'RIVAL REF')
                ax[p, k].set_title(f'{pol} {label} {int(f.iloc[i].obs_id)} valid={int(valid)}\n{f.iloc[i].class_name}', fontsize=8)
                ax[p, k].set(xticks=[], yticks=[])
                provenance.append(dict(obs_id=int(f.iloc[i].obs_id), polarization=pol, valid=valid,
                    path=str(f.iloc[i]['image_'+pol.lower()]), geometry=None if g is None else np.asarray(g).tolist()))
        weights_before = z[baseline+'_weight'][j]
        weights_after = z[arm+'_weight'][j]
        deltas = np.array([z['delta_vv'][j, true]-z['delta_vv'][j, rival],
                           z['delta_vh'][j, true]-z['delta_vh'][j, rival]])
        contributions = (weights_after-weights_before)*deltas
        change = (after[true]-after[rival])-(before[true]-before[rival])
        assert abs(contributions.sum()-change) < 3e-6
        fig.suptitle(f"{arm} vs {baseline}: {case['status']} seed{seed}, true={true}, pred {before.argmax()}->{after.argmax()}\n"
            f"fixed pair margin {before[true]-before[rival]:+.4f}->{after[true]-after[rival]:+.4f}; "
            f"VV/VH weights {weights_before.round(3)}->{weights_after.round(3)}")
        fig.text(.02, .025, f'Native crop/pad, no resize. Fixed-pair contribution changes VV={contributions[0]:+.4f}, VH={contributions[1]:+.4f}. Not physical attribution.', fontsize=9)
        fig.subplots_adjust(top=.82, bottom=.09, wspace=.15, hspace=.2)
        name = f"{arm}_vs_{baseline}_{case['status']}_{int(table.iloc[j].obs_id)}_{seed}.png"
        fig.savefig(out/name, dpi=120)
        plt.close(fig)
        reviewed.append(dict(**case, obs_id=int(table.iloc[j].obs_id), true=true, rival=rival,
            before_pred=int(before.argmax()), after_pred=int(after.argmax()),
            before_margin=float(before[true]-before[rival]), after_margin=float(after[true]-after[rival]),
            before_weights=weights_before.tolist(), after_weights=weights_after.tolist(),
            polar_margin_deltas=deltas.tolist(), contribution_change=contributions.tolist(),
            references=source.iloc[refs].obs_id.astype(int).tolist(), image=name, provenance=provenance))
    run.write_json(out/'cases.json', reviewed)
    run.write_json(HERE/'analysis_receipt.json', dict(cases=len(reviewed), new_training=0,
        bootstrap_unit='pass groups, averaging paired configuration effects before sampling', draws=2000))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(pd.DataFrame(reviewed)[['obs_id', 'seed', 'arm', 'baseline', 'status', 'before_margin', 'after_margin']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
