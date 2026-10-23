"""Prepare four seed17 evaluation cases from frozen competitive-matching caches.

Run only after source_metrics.csv, predictions_17.npz and scores.npz exist.
No solver/head replay, training, feature extraction, or support reselection.
"""
import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
P = HERE.parent
UP = P.parent / 'regional_utility_pilot/upstream'
SEED = 17
ARMS = ('full', 'fixed', 'uncoupled', 'coupled')
LOCAL_ARMS = ARMS[1:]
REGIONS = ('body', 'side', 'end')
SLICES = (slice(0, 8), slice(8, 16), slice(16, 20))
COLORS = ('#24a878', '#2c84c6', '#d39222')


def select_cases(table, predictions, idx):
    import numpy as np

    evaluation = np.flatnonzero(table.role.eq('evaluation').to_numpy())
    if len(evaluation) != 485:
        raise ValueError(f'Expected evaluation485; got {len(evaluation)}')
    lookup = {int(full): local for local, full in enumerate(idx)}
    if not set(evaluation).issubset(lookup):
        raise ValueError('scores.idx does not cover every evaluation row')
    y = table.class_id.to_numpy(int)
    fixed = predictions['fixed'].argmax(1)
    coupled = predictions['coupled'].argmax(1)
    a, b = coupled[evaluation] == y[evaluation], fixed[evaluation] == y[evaluation]
    cases = []
    for outcome, mask in [('rescued', a & ~b), ('harmed', ~a & b),
                          ('CC', a & b), ('WW', ~a & ~b)]:
        ids = evaluation[mask]
        case = dict(outcome=outcome, group_n=int(len(ids)), selected=bool(len(ids)))
        if len(ids):
            i = int(ids[np.argmin(table.iloc[ids].obs_id.to_numpy())])
            true = int(y[i])
            strongest_wrong = int(np.argmax(np.where(
                np.arange(8) == true, -np.inf, predictions['coupled'][i])))
            rival = int(fixed[i]) if outcome == 'rescued' else strongest_wrong
            reason = ('fixed_actual_error' if outcome == 'rescued' else
                      'coupled_highest_nontrue_not_prediction' if outcome == 'CC' else
                      'coupled_actual_error')
            case.update(table_index=i, cache_index=lookup[i], obs_id=int(table.iloc[i].obs_id),
                        true_class=true, fixed_prediction=int(fixed[i]),
                        coupled_prediction=int(coupled[i]), rival_class=rival,
                        rival_reason=reason, coupled_highest_nontrue=strongest_wrong)
        cases.append(case)
    return cases


def load_inputs(prediction_path, score_path, metrics_path):
    import numpy as np
    import pandas as pd

    table = pd.read_csv(UP / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
    if len(table) != 3387:
        raise ValueError(f'Expected full3387 metadata; got {len(table)}')
    metrics = pd.read_csv(metrics_path)
    with np.load(prediction_path, allow_pickle=False) as z:
        predictions = {arm: z[arm] for arm in ARMS}
    with np.load(score_path, allow_pickle=False) as z:
        cache = {key: z[key] for key in ('idx', 'refs', 'coupled_weights', 'baseline_mass', *LOCAL_ARMS)}
    idx, refs = cache['idx'], cache['refs']
    if idx.shape != (1741,) or idx.dtype.kind not in 'iu':
        raise ValueError('scores.idx must be 1741 integer full-table positions')
    if len(np.unique(idx)) != len(idx) or np.any((idx < 0) | (idx >= len(table))):
        raise ValueError('scores.idx contains repeated or out-of-table positions')
    expected = dict(refs=(1741, 8, 2), coupled_weights=(1741, 8, 2, 3, 8),
                    baseline_mass=(1741, 2, 3, 8),
                    **{arm: (1741, 8, 2, 3) for arm in LOCAL_ARMS})
    for key, shape in expected.items():
        if cache[key].shape != shape:
            raise ValueError(f'{key}: expected {shape}, got {cache[key].shape}')
    if refs.dtype.kind not in 'iu' or np.any((refs < 0) | (refs >= len(table))):
        raise ValueError('refs must be integer positions in the full metadata table')
    evaluation = table.role.eq('evaluation').to_numpy()
    for arm, values in predictions.items():
        if values.shape != (len(table), 8) or not np.isfinite(values[evaluation]).all():
            raise ValueError(f'{arm}: expected full3387 x 8, finite on evaluation')
    for key in ('baseline_mass', 'coupled_weights'):
        # End has four real slots. Padding must not be drawn as evidence.
        if not np.allclose(cache[key][..., 2, 4:], 0, atol=1e-8):
            raise ValueError(f'{key}: nonzero end-region right padding')
    with np.load(P / 'tokens.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['instance_ids'].astype(str), table.instance_id.to_numpy())
        tokens = {key: z[key] for key in ('valid', 'mass', 'geometry')}
    return table, predictions, cache, tokens, metrics


def evidence_tables(table, predictions, cache, cases):
    import numpy as np

    margins, local, weights = [], [], []
    for case in cases:
        i, j = case['table_index'], case['cache_index']
        true, rival = case['true_class'], case['rival_class']
        for arm, values in predictions.items():
            own_rival = int(np.argmax(np.where(np.arange(8) == true, -np.inf, values[i])))
            margins.append(dict(obs_id=case['obs_id'], outcome=case['outcome'], arm=arm,
                                prediction=int(values[i].argmax()), true_class=true,
                                shown_rival=rival, true_score=float(values[i, true]),
                                rival_score=float(values[i, rival]),
                                fixed_rival_margin=float(values[i, true] - values[i, rival]),
                                own_rival=own_rival,
                                own_top1_margin=float(values[i, true] - values[i, own_rival])))
        for c in range(8):
            for p, pol in enumerate(('VV', 'VH')):
                for r, region in enumerate(REGIONS):
                    for arm in LOCAL_ARMS:
                        local.append(dict(obs_id=case['obs_id'], candidate_class=c, pol=pol,
                                          region=region, arm=arm, score=float(cache[arm][j, c, p, r]),
                                          true_minus_shown_rival=float(cache[arm][j, true, p, r]
                                                                      - cache[arm][j, rival, p, r])))
                    for slot in range(4 if r == 2 else 8):
                        a = float(cache['baseline_mass'][j, p, r, slot])
                        w = float(cache['coupled_weights'][j, c, p, r, slot])
                        weights.append(dict(obs_id=case['obs_id'], candidate_class=c, pol=pol,
                                            region=region, slot=slot, baseline_mass=a,
                                            coupled_weight=w, delta=w-a))
    return margins, local, weights


def render_case(case, table, predictions, cache, tokens, snapshots, out, geo, slots, plt):
    import numpy as np

    i, j = case['table_index'], case['cache_index']
    true, rival = case['true_class'], case['rival_class']
    refs = cache['refs'][j]
    views = [(i, 'QUERY', None)]
    views += [(int(ref), f'TRUE REF {k+1}', true) for k, ref in enumerate(refs[true])]
    views += [(int(ref), f'RIVAL REF {k+1}', rival) for k, ref in enumerate(refs[rival])]
    records = []
    fig, axes = plt.subplots(2, 5, figsize=(23, 12))
    fig.subplots_adjust(left=.025, right=.985, top=.83, bottom=.18, wspace=.18, hspace=.38)
    for p, pol in enumerate(('vv', 'vh')):
        for column, (ix, label, reference_class) in enumerate(views):
            row = table.iloc[ix]
            if reference_class is not None and int(row.class_id) != reference_class:
                raise ValueError(f'Wrong reference class at full-table row {ix}')
            if reference_class is not None and row.role != 'strategy_fit':
                raise ValueError(f'Reference row {ix} is not in the source strategy_fit bank')
            pix, footprint, geometry, place = geo.native(table, snapshots, ix, p)
            valid = bool(tokens['valid'][ix, p])
            masks = slots(geometry, footprint)
            if valid:
                if geometry is None:
                    raise ValueError(f'Valid cached tokens but missing geometry: {ix}/{pol}')
                # A mismatch would make the displayed regions describe a different input.
                np.testing.assert_allclose(geometry, tokens['geometry'][ix, p], atol=2e-5, rtol=1e-6)
                np.testing.assert_allclose(masks.sum((-1, -2)), tokens['mass'][ix, p],
                                           atol=2e-4, rtol=2e-5)
            ax = axes[p, column]
            ax.imshow(pix, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set(xlim=(-.5, 159.5), ylim=(159.5, -.5), aspect='equal',
                   xticks=[0, 80, 159], yticks=[0, 80, 159])
            if valid:
                for sl, color in zip(SLICES, COLORS):
                    field = masks[sl].sum(0)
                    if field.min() < .5 < field.max():
                        ax.contour(field, levels=[.5], colors=[color], linewidths=.8)
            else:
                ax.text(80, 80, 'NO VALID LOCAL TOKENS', ha='center', color='red', fontsize=8,
                        bbox=dict(facecolor='white', alpha=.85))
            ax.set_title(f'{pol.upper()} {label} | obs {int(row.obs_id)}\n'
                         + textwrap.fill(str(row.class_name), 27) + '\n'
                         + textwrap.fill(str(row.port), 27) + f' | valid={int(valid)}', fontsize=9)
            snapshot = snapshots[int(row.obs_id)]['images'][pol]
            records.append(dict(query_obs=case['obs_id'], outcome=case['outcome'], column=column,
                                panel=label, pol=pol, table_index=ix, obs_id=int(row.obs_id),
                                instance_id=str(row.instance_id), mmsi=str(row.mmsi),
                                class_id=int(row.class_id), role=str(row.role), port=str(row.port),
                                pass_group=str(row.pass_group), path=str(row['image_'+pol]),
                                native_w=int(snapshot['w']), native_h=int(snapshot['h']),
                                placement=place, valid_local_tokens=valid,
                                geometry=None if geometry is None else geometry.tolist(),
                                mask_slot_mass=masks.sum((-1, -2)).tolist(),
                                cached_token_mass=tokens['mass'][ix, p].tolist()))
    fig.suptitle(f"SOURCE evaluation485 | {case['outcome']} coupled vs fixed | seed17 | obs {case['obs_id']}\n"
                 f"true={true}; shown rival={rival} ({case['rival_reason']}); group={case['group_n']}/485\n"
                 'Selection: minimum obs_id within each outcome; actual cached two references per shown class.',
                 fontsize=13)
    lines = [f'{arm}: pred={int(predictions[arm][i].argmax())}, '
             f'true-minus-shown-rival={predictions[arm][i, true]-predictions[arm][i, rival]:+.6f}'
             for arm in ARMS]
    fig.text(.03, .065, '\n'.join(lines), fontsize=10)
    fig.text(.03, .022, 'Native160 crop/pad; original pixel scale; fixed 0..255 display; VV/VH separate. '
             'Green=body, blue=side, amber=end.\n'
             'Contours use actual footprint and metadata. Region names are geometric; no physical or causal claim.',
             fontsize=10)
    fig.savefig(out / f"case_{case['outcome']}_{case['obs_id']}.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for p, pol in enumerate(('VV', 'VH')):
        for r, region in enumerate(REGIONS):
            n = 4 if r == 2 else 8
            x = np.arange(n)
            ax = axes[p, r]
            ax.plot(x, cache['baseline_mass'][j, p, r, :n], 'o-', color='#555555', label='baseline a')
            for c, color, label in [(true, '#16856a', 'true'), (rival, '#c94747', 'shown rival')]:
                ax.plot(x, cache['coupled_weights'][j, c, p, r, :n], 'o-', color=color,
                        label=f'coupled w, {label}={c}')
            ax.set(title=f'{pol} {region}', xlabel='region-local slot (zero based)',
                   ylabel='cached query mass', xticks=x)
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.suptitle(f"obs {case['obs_id']} | candidate-specific query weights | seed17\n"
                 'body/side: longitudinal bin then transverse half; end: longitudinal sign then transverse half',
                 fontsize=12)
    fig.tight_layout(rect=(0, .05, 1, .90))
    fig.text(.025, .02, 'End right padding omitted. Query masses are not transport plans, '
             'per-reference contributions, or additive final-logit attribution.', fontsize=10)
    fig.savefig(out / f"weights_{case['outcome']}_{case['obs_id']}.png", dpi=140)
    plt.close(fig)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=HERE / 'predictions_17.npz')
    parser.add_argument('--scores', type=Path, default=HERE / 'scores.npz')
    parser.add_argument('--metrics', type=Path, default=HERE / 'source_metrics.csv')
    args = parser.parse_args()
    required = [args.predictions, args.scores, args.metrics, UP / 'table.csv', P / 'tokens.npz']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error('Inputs not ready; no analysis started:\n' + '\n'.join(missing))

    import numpy as np
    import pandas as pd

    table, predictions, cache, tokens, metrics = load_inputs(args.predictions, args.scores, args.metrics)
    selections = select_cases(table, predictions, cache['idx'])
    cases = [case for case in selections if case['selected']]
    for key in (*LOCAL_ARMS, 'baseline_mass', 'coupled_weights'):
        if not np.isfinite(cache[key][[case['cache_index'] for case in cases]]).all():
            raise ValueError(f'Nonfinite selected-case evidence: {key}')
    sys.path.insert(0, str(P))
    from extract_tokens import geo, slots

    snapshot_path = geo.OLD / 'local_condition_module/view_snapshot.json'
    wanted = {case['table_index'] for case in cases}
    for case in cases:
        wanted.update(cache['refs'][case['cache_index'], [case['true_class'], case['rival_class']]].ravel().tolist())
    obs_ids = set(table.iloc[sorted(wanted)].obs_id)
    snapshots = {int(s['obs_id']): s for s in json.loads(snapshot_path.read_text(encoding='utf-8'))
                 if int(s['obs_id']) in obs_ids}
    out = HERE / 'analysis'
    out.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(out / '.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pd.DataFrame(selections).to_csv(out / 'case_selection.csv', index=False)
    margins, local, weights = evidence_tables(table, predictions, cache, cases)
    for filename, rows in [('case_margins.csv', margins), ('local_scores.csv', local),
                           ('query_weights.csv', weights)]:
        pd.DataFrame(rows).to_csv(out / filename, index=False)
    records = []
    for case in cases:
        records.extend(render_case(case, table, predictions, cache, tokens, snapshots, out, geo, slots, plt))
    (out / 'image_records.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    receipt = dict(seed=SEED, table_n=len(table), cached_n=len(cache['idx']), evaluation_n=485,
                   cases=selections, inputs=[str(path.resolve()) for path in required],
                   snapshot_path=str(snapshot_path), source_metrics_rows=len(metrics),
                   no_training=True, no_solver_replay=True, no_label_changes=True,
                   visual_interpretation='pending human review; no inferred physical mechanism')
    (out / 'analysis_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    report = ['# Seed17 Competitive Matching Cases', '',
              'Scope: source evaluation485, reused development split. Coupled versus fixed.',
              'Cases are minimum obs_id within each outcome, not estimates of population prevalence.', '',
              '| Outcome | Group size | Selected obs | Shown rival |',
              '| --- | ---: | ---: | --- |']
    for case in selections:
        report.append(f"| {case['outcome']} | {case['group_n']} | {case.get('obs_id', 'none')} | "
                      f"{case.get('rival_reason', 'empty outcome; no replacement')} |")
    report += ['', '## Cached Evidence And Interpretation', '',
               'case_margins.csv distinguishes a fixed comparison rival from each arm\'s strongest wrong class.',
               'local_scores.csv and query_weights.csv preserve all eight candidates for the selected queries.',
               'image_records.json records actual query/reference identities, paths, placement and mask masses.',
               'The displayed refs come directly from scores.npz; no support reselection is performed.',
               'Local scores and query masses alone do not establish transport changes or nonlinear-head attribution.',
               '', '## Required Visual Follow-Up', '',
               'For each case, record visible VV/VH facts, true/rival support availability and evidence changes.',
               'Separate those observations from proposed causes; leave unexplained changes explicitly unresolved.',
               'No physical mechanism or visual explanation is generated automatically by this script.', '']
    (out / 'CASE_REVIEW.md').write_text('\n'.join(report), encoding='utf-8')
    print(pd.DataFrame(selections).to_string(index=False))
    print(f'Wrote {len(cases)} case panels and weight panels to {out}')


if __name__ == '__main__':
    main()
