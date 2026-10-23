"""Descriptive source or fixed-checkpoint test analysis, without fitting."""
import argparse
import json
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_components as fit

HERE, run, parent = fit.HERE, fit.run, fit.parent
sys.path.insert(0, str(parent.PARENT))
from prepare_features import crop_pad, masks_for
PAIRS = [('sigma_only', 'full'), ('kappa_only', 'full'),
         ('sigma_only', 'mean_only'), ('kappa_only', 'mean_only'),
         ('full', 'mean_only'), ('kappa_only', 'sigma_only')]
METRICS = ['OA', 'BA', 'NLL', 'balanced_NLL', 'Brier', 'balanced_Brier']


def measures(g):
    y = g.class_id.to_numpy()
    m = parent.assessment(y, g[fit.SCORE].to_numpy())
    b = parent.assessment(y, g[fit.BASE].to_numpy())
    return dict(**m, instances=g.instance_id.nunique(), mmsi=g.mmsi.nunique(),
                net_B=int(g.net.sum()), rescued_B=int(g.net.eq(1).sum()),
                harmed_B=int(g.net.eq(-1).sum()),
                **{f'delta_{k}_B': m[k]-b[k] for k in METRICS})


def compare(a, b):
    np.testing.assert_array_equal(a.class_id, b.class_id)
    np.testing.assert_allclose(a[fit.BASE], b[fit.BASE], atol=1e-12, rtol=0)
    y = a.class_id.to_numpy()
    ac, bc = a.prediction.to_numpy() == y, b.prediction.to_numpy() == y
    am, bm = measures(a), measures(b)
    return dict(n=len(a), instances=a.instance_id.nunique(), mmsi=a.mmsi.nunique(),
        net=int(ac.sum()-bc.sum()), arm_only_correct=int((ac & ~bc).sum()),
        reference_only_correct=int((bc & ~ac).sum()), both_correct=int((ac & bc).sum()),
        both_wrong=int((~ac & ~bc).sum()), **{f'delta_{k}': am[k]-bm[k] for k in METRICS})


def tables(p, prefix):
    pooled, stratified, paired, pairgroups = [], [], [], []
    units = ['fold', 'class_id', 'port', 'pass_group', 'valid_vh']
    for (role, seed), allarms in p.groupby(['role', 'seed']):
        index = {arm: g.set_index(['fold', 'instance_id'], drop=False).sort_index()
                 for arm, g in allarms.groupby('arm')}
        for arm, g in index.items():
            meta = dict(role=role, seed=seed, arm=arm)
            pooled.append(dict(**meta, **measures(g)))
            for unit in units:
                for value, sub in g.groupby(g[unit]):
                    stratified.append(dict(**meta, unit=unit, value=value, **measures(sub)))
        for arm, reference in PAIRS:
            a, b = index[arm], index[reference].loc[index[arm].index]
            meta = dict(role=role, seed=seed, arm=arm, reference=reference)
            paired.append(dict(**meta, **compare(a, b)))
            for unit in units:
                for value, sub in a.groupby(a[unit]):
                    pairgroups.append(dict(**meta, unit=unit, value=value, **compare(sub, b.loc[sub.index])))
    for suffix, rows in [('pooled', pooled), ('stratified', stratified),
                         ('paired', paired), ('paired_groups', pairgroups)]:
        pd.DataFrame(rows).to_csv(HERE / f'{prefix}_{suffix}.csv', index=False)
    print(pd.DataFrame(paired).query("role != 'train'")[
        ['role', 'seed', 'arm', 'reference', 'net', 'delta_OA', 'delta_BA', 'delta_NLL', 'delta_Brier']].to_string(index=False))


def render(p, prefix):
    role = 'validation' if prefix == 'source' else 'test'
    t = parent.load_table().set_index('instance_id', drop=False)
    snapshots = {v['obs_id']: v for v in json.loads((run.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    with np.load(parent.HERE / 'signed_features.npz') as z:
        moment, mean = z['moments_pol'], z['mean_pol']
        names, ids = list(z['reader_names']), list(z['instance_ids'])
    position = {v: i for i, v in enumerate(ids)}
    refs = p[p.seed.eq(17) & p.role.eq(role)].set_index(['arm', 'fold', 'instance_id'])
    out = HERE / f'{prefix}_panels'
    out.mkdir(exist_ok=True)
    cases, absent, slots = [], [], []
    for arm in fit.ARMS:
        g = p[p.arm.eq(arm) & p.seed.eq(17) & p.role.eq(role)]
        for outcome in ['rescued', 'harmed', 'CC', 'WW']:
            choices = g[g.outcome.eq(outcome)].sort_values(['instance_id', 'fold'])
            if choices.empty:
                absent.append(dict(arm=arm, outcome=outcome))
                continue
            q = choices.iloc[0]
            row = t.loc[q.instance_id]
            fold, y, rival = int(q.fold), int(q.class_id), int(q.final_rival)
            full = refs.loc[('full', fold, q.instance_id)]
            meanq = refs.loc[('mean_only', fold, q.instance_id)]
            scores = {'B': q[fit.BASE].to_numpy(float), arm: q[fit.SCORE].to_numpy(float),
                      'full': full[fit.SCORE].to_numpy(float), 'mean_only': meanq[fit.SCORE].to_numpy(float)}
            fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout='constrained')
            dims = []
            for j, pol in enumerate(['vv', 'vh']):
                view = snapshots[int(row.obs_id)]['images'][pol]
                with Image.open(row[f'image_{pol}']) as im:
                    raw = np.asarray(im.convert('L'))
                pixel, masks, place = crop_pad(raw, masks_for(view), 160)
                dims.append(f'{pol}:{raw.shape[1]}x{raw.shape[0]}')
                for ax, image in zip(axes[j, :2], [raw, pixel]):
                    ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                    ax.set_xlabel('pixel x'); ax.set_ylabel('pixel y')
                axes[j, 0].set_title(f'{pol.upper()} native {raw.shape[1]}x{raw.shape[0]}')
                axes[j, 1].set_title(f'160 crop/pad; valid={q[f"valid_{pol}"]}; no resize')
                prior = view['priors']; target = prior.get('target_index')
                det = prior.get('detections', [])
                if target is not None and 0 <= target < len(det):
                    corners = np.asarray(det[target]['corners'])
                    corners = np.r_[corners, corners[:1]]
                    axes[j, 0].plot(corners[:, 0], corners[:, 1], color='#13b758', lw=1)
                    shifted = corners - [place['crop_x'], place['crop_y']] + [place['pad_left'], place['pad_top']]
                    axes[j, 1].plot(shifted[:, 0], shifted[:, 1], color='#13b758', lw=1)
                m = moment[names.index(f'fold{fold}_outer'), position[q.instance_id], j]
                mu = mean[names.index(f'fold{fold}_outer'), position[q.instance_id], j]
                for c in range(8):
                    slots.append(dict(arm=arm, outcome=outcome, instance_id=q.instance_id,
                        fold=fold, pol=pol, class_id=c, mu=mu[c], sigma=m[c, 0], kappa=m[c, 1]))
            colors = {'B': '#666666', arm: '#2370bc', 'full': '#23915d', 'mean_only': '#c97c19'}
            for name, s in scores.items():
                axes[0, 2].plot(range(8), s, marker='o', label=name, color=colors[name], lw=1.3)
                if name != 'B':
                    axes[1, 2].plot(range(8), s-scores['B'], marker='o', label=name, color=colors[name], lw=1.3)
            for ax in axes[:, 2]:
                ax.set_xticks(range(8)); ax.axhline(0, color='black', lw=.5); ax.legend(fontsize=8)
            axes[0, 2].set_title('Actual scores; shared frozen B')
            axes[1, 2].set_title('Learned direct residual, not physical energy')
            margin = scores[arm][y]-scores[arm][rival]
            base_margin = scores['B'][y]-scores['B'][rival]
            fig.suptitle(f'{role} / {arm} / seed17 / fold{fold} / epoch{int(q.epoch)} / {outcome}\n'
                f'{q.instance_id} / true={y} {row.class_name} / {row.port}\n'
                f'winner B/full/mean/new={int(q.base_prediction)}/{int(full.prediction)}/{int(meanq.prediction)}/{int(q.prediction)}; '
                f'B rival={int(q.old_rival)}; final rival={rival}\n'
                f'Fixed final-rival margin: {base_margin:+.5f} + {margin-base_margin:+.5f} = {margin:+.5f}; '
                'zero epoch means fallback', fontsize=10)
            path = out / f'{arm}_{outcome}.png'
            fig.savefig(path, dpi=100); plt.close(fig)
            cases.append(dict(role=role, arm=arm, outcome=outcome, seed=17, fold=fold,
                epoch=int(q.epoch), instance_id=q.instance_id, obs_id=int(row.obs_id),
                class_id=y, class_name=row.class_name, port=row.port,
                B_prediction=int(q.base_prediction), prediction=int(q.prediction),
                full_prediction=int(full.prediction), mean_prediction=int(meanq.prediction),
                old_rival=int(q.old_rival), final_rival=rival, B_final_rival_margin=base_margin,
                final_margin=margin, delta_final_margin=margin-base_margin,
                dimensions=';'.join(dims), image_vv=row.image_vv, image_vh=row.image_vh,
                panel=str(path), selection='seed17 first instance_id then fold per outcome'))
    pd.DataFrame(cases).to_csv(HERE / f'{prefix}_review_cases.csv', index=False)
    pd.DataFrame(slots).to_csv(HERE / f'{prefix}_case_slots.csv', index=False)
    (HERE / f'{prefix}_missing_outcomes.json').write_text(json.dumps(absent, indent=2), encoding='utf-8')
    print(pd.DataFrame(cases)[['arm', 'outcome', 'instance_id', 'fold', 'epoch', 'B_prediction', 'prediction']].to_string(index=False))


def main(scope):
    filename = 'source_predictions.csv' if scope == 'source' else 'test_predictions.csv'
    p = pd.read_csv(HERE / filename, dtype={'mmsi': str}, float_precision='round_trip')
    assert set(p.role) <= ({'train', 'validation'} if scope == 'source' else {'test'})
    tables(p, scope)
    render(p, scope)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['source', 'test'], required=True)
    main(parser.parse_args().scope)
