"""Fixed matching summaries, paired risks, block intervals, and query/donor images."""
import sys
sys.dont_write_bytecode = True
import json
import time
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_diagnosis import HERE, parent
sys.path.insert(0, str(parent.HERE))
import analyze_results as old

VARIANTS = {'B': 'b', 'original': 'o', 'swapped': 'w'}
COMPARISONS = [('original', 'B'), ('swapped', 'B'), ('original', 'swapped')]


def score(g, variant):
    return g[[f'{VARIANTS[variant]}{c}' for c in range(8)]].to_numpy(float)


def risks(y, s):
    lp = s - logsumexp(s, axis=1, keepdims=True)
    return -lp[np.arange(len(y)), y], ((np.exp(lp) - np.eye(8)[y])**2).sum(1)


def compare(g, left, right):
    y = g.class_id.to_numpy()
    a, b = score(g, left), score(g, right)
    am, bm = parent.assessment(y, a), parent.assessment(y, b)
    diff = (a.argmax(1) == y).astype(int) - (b.argmax(1) == y).astype(int)
    return dict(comparison=f'{left}-{right}', n=len(g), net=int(diff.sum()),
                left_only_correct=int((diff == 1).sum()), right_only_correct=int((diff == -1).sum()),
                **{f'delta_{k}': am[k] - bm[k] for k in ['OA', 'BA', 'J', 'NLL', 'balanced_NLL', 'Brier']})


def intervals(g, left, right):
    y = g.class_id.to_numpy()
    present = np.unique(y)
    a, b = score(g, left), score(g, right)
    diff = (a.argmax(1) == y).astype(int) - (b.argmax(1) == y).astype(int)
    an, ar = risks(y, a)
    bn, br = risks(y, b)
    point = compare(g, left, right)
    rows = []
    for block in ['pass_group', 'mmsi']:
        codes, names = pd.factorize(g[block], sort=True)
        counts, nets = np.zeros((len(names), 8)), np.zeros((len(names), 8))
        np.add.at(counts, (codes, y), 1)
        np.add.at(nets, (codes, y), diff)
        losses = np.column_stack([np.bincount(codes, weights=an-bn, minlength=len(names)),
                                  np.bincount(codes, weights=ar-br, minlength=len(names))])
        rng = np.random.default_rng(20261023)
        oa, ba, nll, brier = [], [], [], []
        for start in range(0, 2000, 100):
            ix = rng.integers(0, len(names), size=(min(100, 2000-start), len(names)))
            den, num = counts[ix].sum(1), nets[ix].sum(1)
            total = den.sum(1)
            oa.extend(100*num.sum(1)/total)
            full = (den[:, present] > 0).all(1)
            ba.extend(100*(num[full][:, present]/den[full][:, present]).mean(1))
            loss = losses[ix].sum(1)/total[:, None]
            nll.extend(loss[:, 0])
            brier.extend(loss[:, 1])
        r = dict(**point, block=block, blocks=len(names), draws=2000, bootstrap_seed=20261023,
                 missing_class_draws=2000-len(ba))
        for metric, draws in [('OA', oa), ('BA', ba), ('NLL', nll), ('Brier', brier)]:
            r[f'{metric}_low'], r[f'{metric}_high'] = np.quantile(draws, [.025, .975])
        rows.append(r)
    return rows


def class_intervals(p):
    rows = []
    for (role, seed, c), g in p.groupby(['role', 'seed', 'class_id']):
        for row in intervals(g, 'original', 'swapped'):
            rows.append(dict(role=role, seed=seed, class_id=c, class_name=g.class_name.iloc[0],
                             **row))
    result = pd.DataFrame(rows)
    result.to_csv(HERE / 'class_intervals.csv', index=False)
    print('Saved', len(result), 'per-class paired block intervals; no matching/model changes.', flush=True)


def matching_stats(g):
    m = g[g.matched]
    return dict(n=len(g), unique_observations=g.instance_id.nunique(), mmsi=g.mmsi.nunique(),
                classes=g.class_id.nunique(), passes=g.pass_group.nunique(), ports=g.port.nunique(),
                matched=len(m), matched_fraction=len(m)/len(g), pairs=len(m)//2,
                matched_unique_observations=m.instance_id.nunique(), matched_mmsi=m.mmsi.nunique(),
                matched_passes=m.pass_group.nunique(), matched_classes=m.class_id.nunique(),
                unmatched_mmsi=g.loc[~g.matched, 'mmsi'].nunique(),
                same_class_fraction=m.same_class.mean(), same_pass_fraction=m.same_pass.mean(),
                same_port_fraction=m.same_port.mean(), distance_mean=m.distance.mean(),
                **{f'distance_{name}': m.distance.quantile(q) for name, q in
                   [('min', 0), ('p25', .25), ('p50', .5), ('p75', .75), ('p90', .9), ('p95', .95), ('max', 1)]})


def case_values(row):
    y = int(row.class_id)
    values = {}
    for variant, prefix in VARIANTS.items():
        s = row[[f'{prefix}{c}' for c in range(8)]].to_numpy(float)
        rival = max((c for c in range(8) if c != y), key=lambda c: s[c])
        values[f'{variant}_prediction'] = int(s.argmax())
        values[f'{variant}_rival'] = int(rival)
        values[f'{variant}_margin'] = float(s[y] - s[rival])
    return values


def render(p):
    table = parent.load_table().set_index('instance_id')
    snapshots = {s['obs_id']: s for s in json.loads(
        (old.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    (HERE / 'panels').mkdir(exist_ok=True)
    g = p[p.seed.eq(17) & p.role.eq('test') & p.matched].copy()
    lookup = g.set_index(['fold', 'instance_id'], drop=False)
    rows, missing = [], []
    for outcome in ['rescued', 'harmed', 'CC', 'WW']:
        hits = g[g.swap_outcome.eq(outcome)].sort_values('instance_id')
        if not len(hits):
            missing.append(outcome)
            continue
        query = hits.iloc[0]
        donor = lookup.loc[(query.fold, query.donor_instance_id)]
        assert donor.donor_instance_id == query.instance_id
        qv, dv = case_values(query), case_values(donor)
        fig, axes = plt.subplots(2, 4, figsize=(20, 11), layout='constrained')
        metadata = dict(outcome=outcome, seed=17, fold=int(query.fold), epoch=int(query.epoch),
                        distance=query.distance, same_class=bool(query.same_class),
                        selection='first matched test ID per original-to-swapped outcome, seed17')
        for side, pred, values in [('query', query, qv), ('donor', donor, dv)]:
            native = table.loc[pred.instance_id]
            metadata.update({f'{side}_{k}': v for k, v in dict(
                instance_id=pred.instance_id, class_id=int(pred.class_id), class_name=pred.class_name,
                mmsi=pred.mmsi, port=pred.port, pass_group=pred.pass_group, **values).items()})
            for j, pol in enumerate(['vv', 'vh']):
                col = j + (0 if side == 'query' else 2)
                view = snapshots[int(native.obs_id)]['images'][pol]
                with Image.open(native[f'image_{pol}']) as im:
                    raw = np.array(im.convert('L'))
                pixels, masks, place = old.crop_pad(raw, old.masks_for(view), 160)
                valid = bool(pred[f'valid_{pol}'])
                metadata[f'{side}_image_{pol}'] = native[f'image_{pol}']
                metadata[f'{side}_{pol}_size'] = f'{raw.shape[1]}x{raw.shape[0]}'
                metadata[f'{side}_{pol}_valid'] = valid
                axes[0, col].imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                axes[0, col].set_title(f'{side.upper()} {pol.upper()} native {raw.shape[1]}x{raw.shape[0]}')
                axes[1, col].imshow(pixels, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                axes[1, col].set_title(f'160 crop/pad; valid={valid}; footprint={masks.sum(0).mean():.3f}')
                prior = view['priors']
                t, det = prior.get('target_index'), prior.get('detections', [])
                if t is not None and 0 <= t < len(det):
                    pts = np.array(det[t]['corners'])
                    pts = np.r_[pts, pts[:1]]
                    axes[0, col].plot(pts[:, 0], pts[:, 1], color='#13c74e', lw=1)
                    pts = pts - [place['crop_x'], place['crop_y']] + [place['pad_left'], place['pad_top']]
                    axes[1, col].plot(pts[:, 0], pts[:, 1], color='#13c74e', lw=1)
                for ax in axes[:, col]:
                    ax.set_xlabel('pixel x')
                    ax.set_ylabel('pixel y')
        fig.suptitle(
            f'Frozen old A+D / free_real / seed17 / fold{int(query.fold)} / epoch{int(query.epoch)}; '
            f'query ORIGINAL -> SWAPPED: {outcome}; distance={query.distance:.5f}\n'
            f'QUERY {query.instance_id} / {query.class_name} / {query.port}: '
            f'B {qv["B_prediction"]}, original {qv["original_prediction"]}, swapped {qv["swapped_prediction"]}; '
            f'margin {qv["original_margin"]:+.5f} -> {qv["swapped_margin"]:+.5f}\n'
            f'DONOR {donor.instance_id} / {donor.class_name} / {donor.port}: '
            f'B {dv["B_prediction"]}, original {dv["original_prediction"]}, swapped {dv["swapped_prediction"]}; '
            f'margin {dv["original_margin"]:+.5f} -> {dv["swapped_margin"]:+.5f}\n'
            'Both keep their own B/flags; exchange the complete pol/class/sigma-kappa block. '
            'GT is shown only AFTER label-free pairing.\n'
            'Green: recorded OBB. Images are not local attribution or causal occlusion maps.', fontsize=11)
        path = HERE / 'panels' / f'matched_{outcome}.png'
        fig.savefig(path, dpi=100)
        plt.close(fig)
        metadata['panel'] = str(path)
        rows.append(metadata)
    pd.DataFrame(rows).to_csv(HERE / 'review_cases.csv', index=False)
    return rows, missing


def main():
    started = time.perf_counter()
    p = pd.read_csv(HERE / 'predictions.csv', dtype={'mmsi': str, 'donor_mmsi': str})
    pooled, paired, ci, matching, composition, unmatched, strata, unique_val = [], [], [], [], [], [], [], []
    for (role, seed), g in p.groupby(['role', 'seed']):
        meta = dict(role=role, seed=seed)
        if role == 'test':
            assert len(g) == 4744 and g.instance_id.is_unique
        matching.append(dict(**meta, scope='pooled', fold=-1, **matching_stats(g)))
        for fold, h in g.groupby('fold'):
            matching.append(dict(**meta, scope='fold', fold=fold, **matching_stats(h)))
        for (fold, reason), h in g[~g.matched].groupby(['fold', 'unmatched_reason']):
            unmatched.append(dict(**meta, fold=fold, reason=reason, n=len(h), mmsi=h.mmsi.nunique(),
                                  classes=','.join(map(str, sorted(h.class_id.unique())))))
        for unit in ['class_id', 'base_prediction', 'mmsi', 'pass_group', 'port', 'valid_vh']:
            for name, h in g.groupby(unit):
                # One-sided composition subsets need not contain both members of a pair.
                info = matching_stats(h)
                info.pop('pairs')
                composition.append(dict(**meta, unit=unit, name=name, **info))
        for scope, h in [('all', g), ('matched', g[g.matched])]:
            for variant in VARIANTS:
                pooled.append(dict(**meta, scope=scope, variant=variant,
                                   unique_observations=h.instance_id.nunique(),
                                   **parent.assessment(h.class_id.to_numpy(), score(h, variant))))
            for left, right in COMPARISONS:
                paired.append(dict(**meta, scope=scope, **compare(h, left, right)))
            for unit in ['fold', 'class_id', 'pass_group', 'port', 'valid_vh']:
                for name, subgroup in h.groupby(unit):
                    for left, right in COMPARISONS:
                        strata.append(dict(**meta, scope=scope, unit=unit, name=name,
                                           **compare(subgroup, left, right)))
        if role == 'validation':
            h = g.sort_values(['fold', 'instance_id']).drop_duplicates('instance_id')
            for left, right in COMPARISONS:
                unique_val.append(dict(**meta, unique_observations=len(h), **compare(h, left, right)))
        for left, right in COMPARISONS:
            ci.extend(dict(**meta, scope='all', **r) for r in intervals(g, left, right))
    for name, rows in [('pooled', pooled), ('paired_metrics', paired), ('paired_intervals', ci),
                       ('matching_summary', matching), ('composition', composition),
                       ('unmatched_summary', unmatched), ('stratified_metrics', strata),
                       ('validation_unique_metrics', unique_val)]:
        pd.DataFrame(rows).to_csv(HERE / f'{name}.csv', index=False)
    class_intervals(p)
    finish(p)


def finish(p):
    started = time.perf_counter()
    cases, missing = render(p)
    summary = dict(render_seconds=time.perf_counter()-started, panels=[r['panel'] for r in cases],
                   images_viewed=False, missing_outcomes=missing, train_steps=0,
                   CI_scope='fixed heads and pairs; matching/selection uncertainty excluded; donor coupling can cross blocks')
    (HERE / 'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pd.read_csv(HERE / 'paired_metrics.csv').query("scope=='all'").to_string(index=False), flush=True)
    print(pd.read_csv(HERE / 'matching_summary.csv').query("scope=='pooled'").to_string(index=False), flush=True)
    print(pd.DataFrame(cases)[['outcome', 'query_instance_id', 'donor_instance_id', 'distance']].to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
