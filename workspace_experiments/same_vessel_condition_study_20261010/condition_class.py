"""Anchored real-observation contrasts, without refitting or outcome-based matching."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE/'condition_class_followup'
FACTORS = ('incidence', 'pass', 'relative_direction', 'port')
CLASSES = ('Bulk carrier', 'Fishing', 'General cargo', 'Product/chemical tanker',
           'Container', 'Crude tanker', 'Tug/towing', 'Offshore supply')
METRICS = ('r50_relation_change', 'r18_relation_change', 'r50_relation_loss',
           'r50_damage', 'r50_correct', 'r18_correct', 'dual_correct',
           'rescue', 'harm', 'net', 'rescued_damage', 'necessary_rescue',
           'necessary_harm', 'relation_rescue', 'gap_days', 'delta_incidence')


def factor_masks(p):
    # Unmatched rows include missing pass metadata, not just intermediate angles.
    known = p.group.ne('unmatched')
    base = known & p.same_port & p.same_platform
    near = p.delta_incidence.le(2)
    control = base & p.same_pass & near
    return {
        'incidence': (base & p.same_pass & p.delta_incidence.ge(5), control),
        'pass': (base & ~p.same_pass & near, control),
        'relative_direction': (control & p.delta_relative_direction.ge(25),
                               control & p.delta_relative_direction.le(10)),
        'port': (p.group.eq('DP_near'), control),
    }


def match_anchors(p, exposed, control, max_gap_difference=30.):
    cols = ['pair_id', 'a', 'gap_bin', 'gap_days']
    candidates = p.loc[exposed, cols].merge(p.loc[control, cols],
        on=['a', 'gap_bin'], suffixes=('_exposed', '_control'))
    candidates['gap_difference'] = abs(candidates.gap_days_exposed-candidates.gap_days_control)
    candidates = candidates.loc[candidates.gap_difference <= max_gap_difference]
    candidates = candidates.sort_values(['gap_difference', 'pair_id_exposed', 'pair_id_control'])
    used_exposed, used_control, records = set(), set(), []
    for row in candidates.itertuples(index=False):
        if row.pair_id_exposed in used_exposed or row.pair_id_control in used_control:
            continue
        used_exposed.add(row.pair_id_exposed)
        used_control.add(row.pair_id_control)
        records.append(dict(pair_id_exposed=row.pair_id_exposed,
            pair_id_control=row.pair_id_control, a=row.a,
            gap_difference=row.gap_difference))
    return pd.DataFrame(records, columns=['pair_id_exposed', 'pair_id_control', 'a', 'gap_difference'])


def add_outcomes(p, obs):
    p = p.copy()
    with np.load(BASE/'evidence_followup/score_parts.npz') as z:
        np.testing.assert_array_equal(z['manifest_rows'], obs.manifest_row)
        removed = (z['part50']+z['bias']).argmax(1)
    y = p.class_id.to_numpy()
    for mode in ('r50', 'r18'):
        p[mode+'_relation_change'] = p[mode+'_relation_margin_b']-p[mode+'_relation_margin_a']
    p['r50_relation_loss'] = 100.*((p.r50_relation_margin_a > 0) & (p.r50_relation_margin_b <= 0))
    p['r50_damage'] = 100.*((p.r50_pred_a == y) & (p.r50_pred_b != y))
    for mode in ('r50', 'r18', 'dual'):
        p[mode+'_correct'] = 100.*(p[mode+'_pred_b'] == y)
    rescue = (p.r50_pred_b != y) & (p.dual_pred_b == y)
    p['rescue'] = 100.*rescue
    p['harm'] = 100.*((p.r50_pred_b == y) & (p.dual_pred_b != y))
    p['net'] = p.rescue-p.harm
    p['rescued_damage'] = 100.*(rescue & (p.r50_pred_a == y))
    p['necessary_rescue'] = 100.*(rescue & (removed[p.b.to_numpy()] != y))
    p['necessary_harm'] = 100.*(p.harm.gt(0) & (removed[p.b.to_numpy()] == y))
    p['relation_rescue'] = 100.*(rescue & (p.r50_relation_margin_b <= 0) & (p.r18_relation_margin_b > 0))
    return p.set_index('pair_id', drop=False)


def summarize(g):
    ship = g.groupby('mmsi')[list(METRICS)].mean()
    result = dict(pairs=len(g), ships=len(ship), anchors=g.a.nunique(),
                  target_observations=g.b.nunique(), ports=g.port_b.nunique())
    result.update(ship.mean().to_dict())
    result.update(rescue_records=int(g.rescue.gt(0).sum()),
                  harm_records=int(g.harm.gt(0).sum()),
                  necessary_rescue_records=int(g.necessary_rescue.gt(0).sum()),
                  necessary_harm_records=int(g.necessary_harm.gt(0).sum()))
    return result


def attach_matched(m, p, obs):
    if m.empty:
        return m
    e, c = p.loc[m.pair_id_exposed].reset_index(drop=True), p.loc[m.pair_id_control].reset_index(drop=True)
    if not ((e.a == c.a) & (e.mmsi == c.mmsi) & (e.gap_bin == c.gap_bin)).all():
        raise ValueError('Anchored pairing contract violated')
    m = m.copy()
    for field in ('mmsi', 'class_id', 'fold', 'port_a'):
        m[field] = e[field].to_numpy()
    for field in (*METRICS, 'b', 'port_b', 'delta_relative_direction'):
        m['exposed_'+field] = e[field].to_numpy()
        m['control_'+field] = c[field].to_numpy()
    for field in METRICS:
        m['contrast_'+field] = m['exposed_'+field]-m['control_'+field]
    by_id = obs.set_index('obs_id')
    for role in ('exposed', 'control'):
        m[role+'_product'] = by_id.loc[m[role+'_b'], 'product'].to_numpy()
    m['anchor_product'] = by_id.loc[m.a, 'product'].to_numpy()
    return m


def summarize_matched(m):
    rows, ship_tables = [], []
    numeric = [c for c in m if c.startswith(('exposed_', 'control_', 'contrast_'))
               and c.split('_', 1)[1] in METRICS]
    for (factor, cls), g in m.groupby(['factor', 'class_id']):
        by_ship = g.groupby('mmsi')[numeric].mean()
        ship_tables.append(by_ship.reset_index().assign(factor=factor, class_id=cls))
        result = dict(factor=factor, class_id=cls, triplets=len(g), ships=len(by_ship),
                      anchors=g.a.nunique(), anchor_ports=g.port_a.nunique(),
                      anchor_products=g.anchor_product.nunique(),
                      exposed_products=g.exposed_product.nunique(),
                      control_products=g.control_product.nunique(),
                      median_gap_difference=g.gap_difference.median(), supported=len(by_ship) >= 5)
        result.update(by_ship.mean().to_dict())
        for col in ('contrast_r50_relation_change', 'contrast_r50_relation_loss',
                    'contrast_r50_damage', 'contrast_net', 'exposed_net'):
            v = by_ship[col].to_numpy()
            if len(v) >= 5:
                rng = np.random.default_rng(20261010)
                means = v[rng.integers(len(v), size=(2000, len(v)))].mean(1)
                lo, hi = np.percentile(means, [2.5, 97.5])
            else:
                lo, hi = np.nan, np.nan
            result[col+'_lo'], result[col+'_hi'] = lo, hi
        rows.append(result)
    return pd.DataFrame(rows), pd.concat(ship_tables, ignore_index=True)


def plot_results(summary):
    titles = ('Change in R50 correct-to-wrong rate', 'Dual minus R50 on changed condition',
              'Change in dual advantage')
    fields = ('contrast_r50_damage', 'exposed_net', 'contrast_net')
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.5), constrained_layout=True)
    for ax, field, title in zip(axes, fields, titles):
        grid = np.full((8, 4), np.nan)
        for r in summary.itertuples():
            if r.supported:
                grid[r.class_id, FACTORS.index(r.factor)] = getattr(r, field)
        im = ax.imshow(grid, cmap='RdBu_r', vmin=-40, vmax=40, aspect='auto')
        ax.set_xticks(range(4), ['Incidence', 'Pass', 'Rel. direction', 'Port'], rotation=30, ha='right')
        ax.set_yticks(range(8), CLASSES)
        ax.set_title(title, fontsize=11, pad=12)
        for r in summary.itertuples():
            x, y = FACTORS.index(r.factor), r.class_id
            text = f'{getattr(r, field):+.1f}\nn={r.ships}' if r.supported else f'n={r.ships}\nsparse'
            ax.text(x, y, text, ha='center', va='center', fontsize=9,
                    color='white' if r.supported and abs(getattr(r, field)) > 24 else 'black')
        fig.colorbar(im, ax=ax, shrink=.7, label='Percentage points')
    fig.suptitle('Same vessel + same earlier observation + matched time bin + lag difference <=30 days\n'
                 'Ship-weighted diagnostic contrasts; n = vessels; blank = unavailable, sparse = fewer than 5 vessels', fontsize=12)
    fig.savefig(OUT/'condition_class_matrix.png', dpi=160)
    fig.savefig(OUT/'condition_class_matrix.pdf')
    plt.close(fig)


def plot_cases(full, p, obs):
    from PIL import Image
    from textwrap import fill
    criteria = [
        ('Rescue with current R18 required', full.factor.eq('pass') & full.class_id.eq(3)
         & full.exposed_necessary_rescue.gt(0)),
        ('R50 correct, fusion harms fishing', full.factor.eq('incidence') & full.class_id.eq(1)
         & full.exposed_harm.gt(0)),
        ('Fusion rescue without required R18', full.factor.eq('incidence') & full.class_id.eq(3)
         & full.exposed_rescue.gt(0) & full.exposed_necessary_rescue.eq(0)),
    ]
    selected = []
    for label, mask in criteria:
        g = full.loc[mask].sort_values(['pair_id_exposed', 'pair_id_control'])
        if len(g):
            selected.append((label, g.iloc[0]))
    fig, axes = plt.subplots(len(selected), 4, figsize=(15, 4*len(selected)),
                             gridspec_kw={'width_ratios': [1, 1, 1, 1.3]}, layout='constrained')
    by_id = obs.set_index('obs_id')
    with np.load(BASE/'FROZEN_PREDICTIONS.npz') as z:
        scores = z['scores']
    with np.load(BASE/'evidence_followup/score_parts.npz') as z:
        removed_scores = z['part50']+z['bias']
    records = []
    for i, (label, row) in enumerate(selected):
        for j, (role, oi) in enumerate([('Anchor', row.a), ('Near condition', row.control_b),
                                        ('Changed condition', row.exposed_b)]):
            ob = by_id.loc[int(oi)]
            with Image.open(ob.image_relpath_vh) as im:
                axes[i, j].imshow(np.asarray(im), cmap='gray', interpolation='nearest')
            axes[i, j].axis('off')
            pred = scores[int(oi)].argmax(1)
            axes[i, j].set_title(f'{role}: {ob.port}\n{str(ob.acq_time)[:10]} | inc={ob.incidence_deg:.1f}\n'
                                f'{ob.pass_name} | R50/R18/dual={pred[0]}/{pred[1]}/{pred[2]}', fontsize=10)
        accounting = p.loc[int(row.pair_id_exposed)]
        removed_pred = int(removed_scores[int(row.exposed_b)].argmax())
        txt = (f'{label}\n{CLASSES[int(row.class_id)]} (class {int(row.class_id)})\n'
               f'Factor: {row.factor}\nMMSI: {int(row.mmsi)}\n'
               f'Gap control/change: {row.control_gap_days:.1f}/{row.exposed_gap_days:.1f} d\n\n'
               f'True vs rival {int(accounting.rival)}, changed observation:\n'
               f'Original R50 margin: {accounting.single_margin_b_fixed_rival:+.3f}\n'
               f'R50 reweight + bias: {accounting.old_route_and_intercept_term:+.3f}\n'
               f'Added R18 term: {accounting.r18_added_term:+.3f}\n'
               f'Dual margin: {accounting.dual_margin_b_fixed_rival:+.3f}\n\n'
               f'R18-removed prediction: {removed_pred}\n'
               'The displayed margin is against one rival; removal is evaluated over all 8 classes.\n\n'
               'Illustrative selection by first pair ID within the stated outcome category; not a prevalence estimate.')
        axes[i, 3].axis('off')
        axes[i, 3].text(0, 1, '\n'.join(fill(line, 39) for line in txt.splitlines()),
                        va='top', fontsize=10, transform=axes[i, 3].transAxes)
        records.append(dict(selection=label, **row.to_dict(), rival=int(accounting.rival),
            single_margin=accounting.single_margin_b_fixed_rival,
            r18_removed_prediction=removed_pred,
            old_route_and_intercept=accounting.old_route_and_intercept_term,
            r18_added_term=accounting.r18_added_term, dual_margin=accounting.dual_margin_b_fixed_rival))
    fig.savefig(OUT/'anchored_examples.png', dpi=150)
    plt.close(fig)
    pd.DataFrame(records).to_csv(OUT/'illustrative_cases.csv', index=False)


def run():
    OUT.mkdir(exist_ok=True)
    protocol = dict(date='2026-10-10', analysis='observational anchored condition contrasts',
        thresholds=dict(incidence_control_max=2, incidence_exposed_min=5,
                        relative_control_max=10, relative_exposed_min=25),
        matching='same earlier observation, same vessel, same gap_bin; nearest gap first, greedy 1:1 pair IDs',
        max_gap_difference_days=30, exposure_and_control_both_after_anchor=True,
        selection_uses_outcomes=False, class_support_minimum_vessels=5,
        aggregation='triplet means within each vessel, equal vessel weight within class',
        bootstrap='2000 vessel resamples, descriptive 95% intervals; scene dependence and multiple comparisons uncorrected',
        diagnostic_only=True, refit=False,
        limitations=['Observed conditions only; no randomized physical intervention',
                     'Relative direction from image-derived body axis, not verified heading',
                     'Frozen diagnostic R50/R18, not current full-model LOPO',
                     '64 references describe class relations; not actual classifier training influence',
                     'Removing R18 at fixed head is not a retrained single-backbone comparison'])
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    obs = pd.read_csv(BASE/'observations.csv')
    p = pd.read_csv(BASE/'evidence_followup/pair_accounting.csv')
    masks = factor_masks(p)
    matched, coverage = [], []
    for factor, (exposed, control) in masks.items():
        m = match_anchors(p, exposed, control).assign(factor=factor)
        m.to_csv(OUT/f'metadata_matches_{factor}.csv', index=False)
        matched.append(m)
        coverage.append(dict(factor=factor, exposed_pairs=int(exposed.sum()),
                             control_pairs=int(control.sum()), matched_triplets=len(m)))
    # Outcomes are attached only after all metadata-only matching is complete.
    p = add_outcomes(p, obs)
    tables = []
    for factor, (exposed, control) in masks.items():
        for role, mask in [('exposed', exposed), ('control', control)]:
            for cls, g in p.loc[mask.to_numpy()].groupby('class_id'):
                tables.append(dict(factor=factor, role=role, class_id=cls, **summarize(g)))
    descriptive = pd.DataFrame(tables)
    descriptive.to_csv(OUT/'descriptive_condition_class.csv', index=False)
    unique_targets = []
    for factor, (exposed, _) in masks.items():
        for cls, g in p.loc[exposed.to_numpy()].drop_duplicates('b').groupby('class_id'):
            fields = ['r50_correct', 'r18_correct', 'dual_correct', 'rescue', 'harm', 'net',
                      'necessary_rescue', 'necessary_harm']
            values = g.groupby('mmsi')[fields].mean().mean().to_dict()
            unique_targets.append(dict(factor=factor, class_id=cls, target_observations=len(g),
                                       ships=g.mmsi.nunique(), **values))
    pd.DataFrame(unique_targets).to_csv(OUT/'unique_target_outcomes.csv', index=False)
    full = pd.concat([attach_matched(m, p, obs) for m in matched if not m.empty], ignore_index=True)
    full.to_csv(OUT/'matched_triplets.csv', index=False)
    summary, ships = summarize_matched(full)
    summary.to_csv(OUT/'matched_condition_class.csv', index=False)
    ships.to_csv(OUT/'matched_ship_contrasts.csv', index=False)
    # Disjoint port summaries expose concentration instead of turning scenes into independent samples.
    port = full.groupby(['factor', 'class_id', 'port_a', 'mmsi'])[
        ['contrast_r50_relation_change', 'contrast_r50_damage', 'exposed_net', 'contrast_net']].mean()
    port.groupby(['factor', 'class_id', 'port_a']).agg(['mean', 'count']).to_csv(OUT/'matched_port_breakdown.csv')
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv', index=False)
    plot_results(summary)
    plot_cases(full, p, obs)
    print(pd.DataFrame(coverage).to_string(index=False), flush=True)
    cols = ['factor', 'class_id', 'triplets', 'ships', 'anchor_ports', 'contrast_r50_relation_change',
            'contrast_r50_damage', 'exposed_rescue', 'exposed_harm', 'exposed_net', 'contrast_net']
    print(summary.loc[summary.supported, cols].round(3).to_string(index=False), flush=True)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(matched_triplets=len(full),
        unique_ships=int(full.mmsi.nunique()), supported_cells=int(summary.supported.sum()),
        no_training=True), indent=2), encoding='utf-8')


if __name__ == '__main__':
    run()
