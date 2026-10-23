"""Paired frozen-prediction comparisons and eight native review panels."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_information import HERE, STANDARD, parent, ref, restore, replace
sys.path.insert(0, str(parent.HERE))
import analyze_results as old


def render(predictions):
    table = parent.load_table().set_index('instance_id')
    snapshots = {s['obs_id']: s for s in json.loads(
        (old.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    (HERE / 'panels').mkdir(exist_ok=True)
    rows, missing = [], []
    for arm, g in predictions[predictions.seed.eq(17) & predictions.role.eq('test')].groupby('arm'):
        for outcome in ['rescued', 'harmed', 'CC', 'WW']:
            hits = g[g.outcome.eq(outcome)].sort_values('instance_id')
            if not len(hits):
                missing.append(dict(arm=arm, outcome=outcome))
                continue
            q = hits.iloc[0]
            row = table.loc[q.instance_id]
            y = int(q.class_id)
            b = q[[f'b{c}' for c in range(8)]].to_numpy(float)
            s = q[[f's{c}' for c in range(8)]].to_numpy(float)
            before = max((c for c in range(8) if c != y), key=lambda c: b[c])
            after = max((c for c in range(8) if c != y), key=lambda c: s[c])
            fig, axes = plt.subplots(2, 2, figsize=(10, 11), layout='constrained')
            dimensions, coverage = [], []
            for j, pol in enumerate(['vv', 'vh']):
                view = snapshots[int(row.obs_id)]['images'][pol]
                with Image.open(row[f'image_{pol}']) as im:
                    raw = np.array(im.convert('L'))
                pixels, masks, place = old.crop_pad(raw, old.masks_for(view), 160)
                dimensions.append(f'{pol}:{raw.shape[1]}x{raw.shape[0]}')
                coverage.append(float(masks.sum(0).mean()))
                axes[0, j].imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                axes[0, j].set_title(f'{pol.upper()} original {raw.shape[1]}x{raw.shape[0]}')
                axes[1, j].imshow(pixels, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                axes[1, j].set_title(f'Actual 160 crop/pad; valid={q[f"valid_{pol}"]}\nfootprint={coverage[-1]:.3f}')
                prior = view['priors']
                t, det = prior.get('target_index'), prior.get('detections', [])
                if t is not None and 0 <= t < len(det):
                    points = np.array(det[t]['corners'])
                    points = np.r_[points, points[:1]]
                    axes[0, j].plot(points[:, 0], points[:, 1], color='#13c74e', lw=1)
                    points = points - [place['crop_x'], place['crop_y']] + [place['pad_left'], place['pad_top']]
                    axes[1, j].plot(points[:, 0], points[:, 1], color='#13c74e', lw=1)
                for ax in axes[:, j]:
                    ax.set_xlabel('pixel x')
                    ax.set_ylabel('pixel y')
            note = ('Per-image aggregate moments repeated across valid slots; no original pol allocation'
                    if arm == 'free_aggregate' else
                    'B + valid flags + fixed source moments; NO per-image moments / kappa attribution')
            fig.suptitle(f'{arm} / seed17 / fold{int(q.fold)} / epoch{int(q.epoch)} / {outcome}\n'
                         f'{q.instance_id} / {row.class_name} / {row.port}\n{note}\n'
                         f'B (frozen calibration) -> head: {int(q.base_prediction)} -> {int(q.prediction)}; '
                         f'true={y}; strongest rival {before} -> {after}\n'
                         f'True margin {b[y]-b[before]:+.5f} -> {s[y]-s[after]:+.5f}; '
                         f'fixed old-rival change {(s-b)[y]-(s-b)[before]:+.5f}\n'
                         'Green: recorded OBB. Native images shown for review, not a causal contribution map.', fontsize=10)
            path = HERE / 'panels' / f'{arm}_{outcome}_{q.instance_id}.png'
            fig.savefig(path, dpi=110)
            plt.close(fig)
            rows.append(dict(arm=arm, seed=17, fold=int(q.fold), epoch=int(q.epoch), outcome=outcome,
                             instance_id=q.instance_id, class_id=y, class_name=row.class_name, port=row.port,
                             base_prediction=int(q.base_prediction), prediction=int(q.prediction),
                             old_rival=before, new_rival=after, base_margin=b[y]-b[before], margin=s[y]-s[after],
                             old_rival_delta=(s-b)[y]-(s-b)[before], dimensions=';'.join(dimensions),
                             valid_vv=bool(q.valid_vv), valid_vh=bool(q.valid_vh),
                             footprint_vv=coverage[0], footprint_vh=coverage[1],
                             image_vv=row.image_vv, image_vh=row.image_vh, panel=str(path), input_note=note,
                             selection='fixed seed17 first instance_id in each outcome'))
    pd.DataFrame(rows).to_csv(HERE / 'review_cases.csv', index=False)
    return rows, missing


def main():
    p = pd.read_csv(HERE / 'predictions.csv', dtype={'mmsi': str})
    frozen = pd.read_csv(STANDARD / 'predictions.csv', dtype={'mmsi': str})
    frozen = frozen[frozen.arm.eq('free_shared')]
    allp = pd.concat([p, frozen], ignore_index=True)
    strata, pooled, ci, unique_val, matched, classes, pair_rows = [], [], [], [], [], [], []
    for (role, arm, seed), g in allp.groupby(['role', 'arm', 'seed']):
        meta = dict(role=role, arm=arm, seed=seed)
        assert np.array_equal(g.prediction, g[[f's{c}' for c in range(8)]].to_numpy().argmax(1))
        pooled.append(dict(**meta, unique_observations=g.instance_id.nunique(), **old.metrics(g)))
        for unit in ['fold', 'class_id', 'valid_vh', 'port', 'pass_group']:
            for name, h in g.groupby(unit):
                strata.append(dict(**meta, unit=unit, name=name, **old.metrics(h)))
        if role == 'validation':
            h = g.sort_values(['fold', 'instance_id']).drop_duplicates('instance_id')
            unique_val.append(dict(**meta, **old.metrics(h)))
        else:
            ci.extend(dict(**meta, **r) for r in old.intervals(g, g.net.to_numpy(), arm + '-B'))
    for (role, seed), g in frozen.groupby(['role', 'seed']):
        a = g.set_index(['fold', 'instance_id']).sort_index()
        for arm in ['free_aggregate', 'free_constant']:
            h = p[p.role.eq(role) & p.seed.eq(seed) & p.arm.eq(arm)].set_index(['fold', 'instance_id']).loc[a.index]
            assert np.array_equal(a.class_id, h.class_id)
            assert np.array_equal(a.base_prediction, h.base_prediction)
            diff = a.net.to_numpy() - h.net.to_numpy()
            am, hm = old.metrics(a), old.metrics(h)
            comparison = 'free_shared-' + arm
            meta = dict(role=role, arm='free_shared', reference=arm, seed=seed)
            matched.append(dict(**meta, n=len(a), unique_observations=a.index.get_level_values('instance_id').nunique(),
                                net=int(diff.sum()), real_only_correct=int((diff == 1).sum()),
                                control_only_correct=int((diff == -1).sum()),
                                delta_OA=am['OA']-hm['OA'], delta_BA=am['BA']-hm['BA'],
                                delta_NLL=am['NLL']-hm['NLL'], delta_Brier=am['Brier']-hm['Brier']))
            for c in range(8):
                mask = a.class_id.to_numpy() == c
                classes.append(dict(**meta, class_id=c, n=int(mask.sum()), net=int(diff[mask].sum()),
                                    delta_recall=100*float(diff[mask].mean())))
            pairs = a[['class_id', 'mmsi', 'pass_group', 'port', 'prediction', 'base_prediction']].reset_index()
            pairs = pairs.rename(columns={'prediction': 'real_prediction'})
            pairs['control_prediction'], pairs['paired_net'] = h.prediction.to_numpy(), diff
            for k, v in meta.items():
                pairs[k] = v
            pair_rows.append(pairs)
            if role == 'test':
                ci.extend(dict(**meta, **r) for r in old.intervals(a, diff, comparison))
    for name, rows in [('all_pooled', pooled), ('stratified_metrics', strata), ('paired_intervals', ci),
                       ('validation_unique_metrics', unique_val), ('matched_metrics', matched), ('matched_classes', classes)]:
        pd.DataFrame(rows).to_csv(HERE / f'{name}.csv', index=False)
    pd.concat(pair_rows, ignore_index=True).to_csv(HERE / 'paired_predictions.csv', index=False)
    finish(p)


def finish(p):
    cases, missing = render(p)
    summary = dict(panels=[r['panel'] for r in cases], missing_outcomes=missing, images_viewed=False,
                   seed_pooling=False, bootstrap_scope='fixed predictions only; no selection/training uncertainty',
                   frozen_reference=str(STANDARD / 'predictions.csv'), frozen_real_retrained=False)
    (HERE / 'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pd.read_csv(HERE / 'all_pooled.csv').query("role=='test'")[['arm', 'seed', 'OA', 'BA', 'net', 'delta_OA', 'delta_BA', 'NLL', 'Brier']].to_string(index=False))
    print(pd.read_csv(HERE / 'matched_metrics.csv').to_string(index=False))
    print(pd.DataFrame(cases)[['arm', 'outcome', 'instance_id', 'old_rival', 'new_rival', 'base_margin', 'margin']].to_string(index=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
