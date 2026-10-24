"""Bounded region x scale interventions on a frozen classification diagnostic."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import json
import time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, distance_transform_edt
import directional_intervention as d
import joint_probe as j

ROOT = d.ROOT
OUT = ROOT / 'round21_classification_joint'


def regions(im, view):
    sh = view['priors']['ship_axis']
    masks = j.masks_for(im, sh, sh['length_px'], sh['width_px'])
    masks['body'] = d.ip.masks_for(view)[0].astype(bool)
    masks['end'] &= ~masks['body']
    masks['side'] &= ~masks['body']
    return masks


def bands(im):
    g1 = gaussian_filter(im, 1, mode='reflect')
    g2 = gaussian_filter(im, 2, mode='reflect')
    return {'fine': im-g1, 'middle': g1-g2}, g2


def perturb(im, view):
    masks = regions(im, view)
    components, low = bands(im)
    assert np.max(abs(low + components['fine'] + components['middle'] - im)) < 1e-12
    proposals = {}
    for region, mask in masks.items():
        taper = np.minimum(distance_transform_edt(mask)/2, 1)*mask
        for band, component in components.items():
            delta = np.clip(im-.5*taper*component, 0, 1)-im
            proposals[(region, band)] = (delta, taper, mask)
    target = .5*min(np.linalg.norm(v[0]) for v in proposals.values())
    if target < 1e-8:
        return []
    output = []
    for (region, band), (delta, taper, mask) in proposals.items():
        for dose in (.5, 1.):
            altered = im + dose*target*delta/np.linalg.norm(delta)
            change = altered-im
            mean_shift = change.sum()
            basis = taper*((1-im) if mean_shift >= 0 else im)
            coefficient = mean_shift/basis.sum() if basis.sum() > 0 else 0.
            assert abs(coefficient) <= 1
            level = im + coefficient*basis
            assert np.max(abs(change[~mask])) == 0
            assert abs(np.linalg.norm(change)-dose*target) < 1e-10
            assert abs((level-im).sum()-mean_shift) < 1e-10
            assert altered.min() >= 0 and altered.max() <= 1
            assert level.min() >= 0 and level.max() <= 1
            after, _ = bands(altered)
            meta = dict(region=region, band=band, dose=dose,
                        l2=float(np.linalg.norm(change)), mean_delta=float(change.mean()),
                        level_l2=float(np.linalg.norm(level-im)), pixels=int(mask.sum()),
                        band_energy_before=float(np.mean(components[band][mask]**2)),
                        band_energy_after=float(np.mean(after[band][mask]**2)))
            output.append((altered, level, meta))
    return output


def select(frame, old):
    source = pd.read_csv(ROOT/'round08_boundary_reference/pairs.csv', dtype={'mmsi': str})
    source = source[['history_id', 'query_id', 'mmsi']].drop_duplicates()
    records = []
    for r in source.itertuples():
        if r.history_id not in old.index or r.query_id not in old.index:
            continue
        a, b = old.loc[r.history_id], old.loc[r.query_id]
        if a.fold != b.fold or r.history_id == 31237 or r.query_id == 31237:
            continue
        assert a.class_id == b.class_id
        state = ('C' if a.prediction == a.class_id else 'W') + ('C' if b.prediction == b.class_id else 'W')
        records.append(dict(history_id=r.history_id, query_id=r.query_id,
                            mmsi=r.mmsi, class_id=int(a.class_id), fold=int(a.fold), state=state))
    candidates = pd.DataFrame(records).sort_values(['class_id', 'mmsi', 'history_id'])
    chosen = []
    for state, group in candidates.groupby('state'):
        # First cover different classes, then fill; never select by intervention outcome.
        first = group.drop_duplicates('class_id')
        order = pd.concat([first, group.drop(first.index)]).drop_duplicates('mmsi')
        chosen.append(order.head(3))
    candidates.to_csv(OUT/'candidate_pairs.csv', index=False)
    selection = pd.concat(chosen, ignore_index=True)
    selection.to_csv(OUT/'selected_pairs.csv', index=False)
    return selection


def analyze(table, pairs):
    keys = ['pol', 'region', 'band', 'dose']
    table.groupby(keys).agg(n=('obs_id', 'size'), ships=('mmsi', 'nunique'),
        mean_delta=('delta_fixed', 'mean'), mean_abs=('delta_fixed', lambda x: abs(x).mean()),
        residual_mean=('residual', 'mean'), residual_abs=('residual', lambda x: abs(x).mean()),
        rescued=('rescued', 'sum'), harmed=('harmed', 'sum')).to_csv(OUT/'summary.csv')
    wide = table.pivot(index=['obs_id', 'mmsi', 'pol', 'dose'],
                       columns=['region', 'band'], values='residual')
    interaction = []
    for key, row in wide.iterrows():
        for neighbor in ('end', 'side'):
            value = (row['body', 'fine']-row['body', 'middle'])-(row[neighbor, 'fine']-row[neighbor, 'middle'])
            interaction.append(dict(obs_id=key[0], mmsi=key[1], pol=key[2], dose=key[3],
                                    contrast='body_vs_'+neighbor, interaction=value))
    pd.DataFrame(interaction).to_csv(OUT/'region_scale_interactions.csv', index=False)
    changes = []
    for pair in pairs.itertuples():
        a = table[table.obs_id.eq(pair.history_id)].set_index(keys)
        b = table[table.obs_id.eq(pair.query_id)].set_index(keys)
        for key in a.index.intersection(b.index):
            x, y = a.loc[key], b.loc[key]
            changes.append(dict(history_id=pair.history_id, query_id=pair.query_id,
                mmsi=pair.mmsi, state=pair.state, **dict(zip(keys, key)),
                original_margin_change=y.pair_margin-x.pair_margin,
                altered_margin_change=y.pair_margin_after-x.pair_margin_after,
                intervention_gap_change=(y.pair_margin_after-x.pair_margin_after)-(y.pair_margin-x.pair_margin),
                energy_change=y.band_energy_before-x.band_energy_before,
                history_sensitivity=x.residual, query_sensitivity=y.residual))
    pd.DataFrame(changes).to_csv(OUT/'same_ship_changes.csv', index=False)


def panels(table, pairs, frame, views):
    os.environ['MPLCONFIGDIR'] = str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = []
    display = pairs.groupby('state', sort=True).head(1).copy()
    flips = set(table.loc[table.rescued | table.harmed, 'obs_id'])
    covered = set(display.history_id) | set(display.query_id)
    for oid in sorted(flips-covered):
        extra = pairs[pairs.history_id.eq(oid) | pairs.query_id.eq(oid)].iloc[[0]].copy()
        extra['state'] = extra.state+'_flip'
        display = pd.concat([display, extra], ignore_index=True)
    for _, pair in display.iterrows():
        state = pair.state
        sub = table[table.obs_id.isin([pair.history_id, pair.query_id]) & table.dose.eq(1)]
        # Case identity fixed before effects; strongest region is descriptive, not confirmatory.
        focus = sub[sub.rescued | sub.harmed] if state.endswith('_flip') else sub
        effect = focus.loc[focus.residual.abs().idxmax()]
        selected.append(dict(state=state, history_id=int(pair.history_id), query_id=int(pair.query_id),
                             region=effect.region, band=effect.band, selected_pol=effect.pol))
        fig, axes = plt.subplots(4, 4, figsize=(12, 11), constrained_layout=True)
        for endpoint, oid in enumerate((pair.history_id, pair.query_id)):
            i = frame.index[frame.obs_id.eq(oid)][0]
            for p, pol in enumerate(('vv', 'vh')):
                axrow = axes[endpoint*2+p]
                view = views[i]['images'][pol]
                im = np.asarray(d.ip.Image.open(frame.loc[i, 'image_'+pol]).convert('L'), dtype=float)/255
                entry = next((a, l, m) for a, l, m in perturb(im, view)
                             if m['region'] == effect.region and m['band'] == effect.band and m['dose'] == 1)
                altered, level, meta = entry
                z = sub[sub.obs_id.eq(oid) & sub.pol.eq(pol) & sub.region.eq(effect.region) & sub.band.eq(effect.band)]
                component = bands(im)[0][effect.band]
                limit = max(abs(component).max(), 1e-8)
                for col, (label, image) in enumerate((('native', im), ('selected component', component),
                                                      ('attenuated', altered), ('mean control', level))):
                    ax = axrow[col]
                    ax.imshow(image, cmap='RdBu_r' if col == 1 else 'gray',
                              vmin=-limit if col == 1 else 0, vmax=limit if col == 1 else 1,
                              interpolation='nearest')
                    ax.contour(regions(im, view)[effect.region], levels=[.5], colors='lime', linewidths=.6)
                    ax.set_title(f'{oid} {pol.upper()} {label}', fontsize=8)
                    ax.set_aspect('equal')
                if not z.empty:
                    r = z.iloc[0]
                    axrow[2].set_xlabel(f'margin {r.delta_fixed:+.5f}; {r.base_pred}->{r.pred}', fontsize=9)
                    axrow[3].set_xlabel(f'margin {r.level_delta:+.5f}', fontsize=9)
        fig.suptitle(f'Frozen diagnostic {state}: {pair.history_id}->{pair.query_id}\n'
                     f'{effect.region}/{effect.band}; each row perturbed separately, native scale')
        fig.savefig(OUT/f'case_{state}_{int(pair.query_id)}.png', dpi=115)
        plt.close(fig)
    pd.DataFrame(selected).to_csv(OUT/'case_selection.csv', index=False)


def main():
    OUT.mkdir(exist_ok=True)
    started = time.perf_counter()
    d.torch.set_num_threads(2)
    frame, views, disabled = d.ip.load_data()
    old = pd.read_csv(d.CONTROL/'predictions.csv')
    old = old[old.arm.eq('full_global')].set_index('obs_id')
    pairs = select(frame, old)
    cache = np.load(d.CONTROL/'features.npz')['full_global']
    heads = [np.load(d.CONTROL/f'head_f{k}_full_global.npz') for k in range(3)]
    model = d.resnet50(weights=None)
    model.load_state_dict(d.torch.load(d.ip.WEIGHTS, map_location='cpu', weights_only=True))
    stem = d.torch.nn.Sequential(model.conv1, model.bn1, model.relu).eval()
    del model
    pair_rivals = {}
    for pair in pairs.itertuples():
        ix = [frame.index[frame.obs_id.eq(oid)][0] for oid in (pair.history_id, pair.query_id)]
        scores = d.score(cache[ix], heads[pair.fold]).sum(0)
        scores[pair.class_id] = -np.inf
        for oid in (pair.history_id, pair.query_id):
            pair_rivals[oid] = int(scores.argmax())
    checks = dict(feature_replay=0., margin_replay=0.)
    records = []
    with d.torch.inference_mode():
        for count, oid in enumerate(sorted(pair_rivals)):
            i = frame.index[frame.obs_id.eq(oid)][0]
            row = frame.loc[i]
            head = heads[int(row.fold)]
            base = d.score(cache[i], head)
            y = int(row.class_id)
            wrong = base.copy(); wrong[y] = -np.inf
            rival = int(wrong.argmax()); common = pair_rivals[oid]
            margin = float(base[y]-base[rival]); bp = int(base.argmax())
            checks['margin_replay'] = max(checks['margin_replay'], abs(margin-old.loc[oid, 'margin']))
            for p, pol in enumerate(('vv', 'vh')):
                if p == 1 and disabled[i]:
                    continue
                view = views[i]['images'][pol]
                transform = np.array(view['priors']['scale']['transform'])[[1, 2, 4, 5]].reshape(2, 2)
                assert np.allclose(transform.T@transform, np.eye(2)*100, atol=1e-5)
                im = np.asarray(d.ip.Image.open(row['image_'+pol]).convert('L'), dtype=float)/255
                items = perturb(im, view)
                assert items, (oid, pol)
                images = [im]+[x for a, l, m in items for x in (a, l)]
                desc = d.ip.describe(stem, *d.prepare_batch(images, view), inputs=('full',))['full_global'].numpy()
                sl = slice(p*128, (p+1)*128)
                err = float(abs(desc[0]-cache[i, sl]).max())
                checks['feature_replay'] = max(checks['feature_replay'], err)
                assert err < 2e-5, (oid, pol, err)
                raw = np.tile(cache[i], (len(desc)-1, 1)); raw[:, sl] = desc[1:]
                scores = d.score(raw, head)
                for n, (_, _, meta) in enumerate(items):
                    s, level = scores[2*n:2*n+2]
                    pred = int(s.argmax())
                    delta = float(s[y]-s[rival]-margin)
                    ld = float(level[y]-level[rival]-margin)
                    records.append(dict(obs_id=oid, mmsi=row.mmsi, class_id=y, fold=int(row.fold),
                        pol=pol, product=row['product'], image_path=row['image_'+pol], **meta,
                        rival=rival, pair_rival=common, base_pred=bp, pred=pred,
                        baseline_margin=margin, delta_fixed=delta, level_delta=ld, residual=delta-ld,
                        true_score_change=float(s[y]-base[y]), rival_score_change=float(s[rival]-base[rival]),
                        pair_margin=float(base[y]-base[common]), pair_margin_after=float(s[y]-s[common]),
                        rescued=bp != y and pred == y, harmed=bp == y and pred != y))
            print('observations', count+1, '/', len(pair_rivals), 'seconds', round(time.perf_counter()-started, 1), flush=True)
    assert checks['margin_replay'] < 1e-5
    table = pd.DataFrame(records)
    table.to_csv(OUT/'interventions.csv', index=False)
    analyze(table, pairs)
    panels(table, pairs, frame, views)
    manifest = dict(pairs=len(pairs), ships=pairs.mmsi.nunique(), observations=len(pair_rivals),
                    interventions=len(table), checks=checks, seconds=time.perf_counter()-started,
                    model='frozen full_global C4 stem diagnostic, not current complete model',
                    support_adaptation=False, new_training=False, physical_causality=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
