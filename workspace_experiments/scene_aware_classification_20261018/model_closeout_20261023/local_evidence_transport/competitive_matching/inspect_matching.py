"""Inspect at most four completed seed17 cases; never optimize or train.

Run after analyze_cases.py and source evaluation finish. Reconstruct only selected
queries and their cached references, then solve inner transport at a and saved w.
Outputs: small CSVs and a receipt in analysis/matching_inspection/. No NPZ export.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UP = ROOT.parent / 'regional_utility_pilot/upstream'
SEED = 17
ARMS = ('fixed', 'coupled')
REGIONS = ('body', 'side', 'end')


def load_inputs():
    import numpy as np
    import pandas as pd

    table = pd.read_csv(UP / 'table.csv', dtype={'instance_id': str, 'mmsi': str})
    selected = pd.read_csv(HERE / 'analysis/case_selection.csv')
    selected = selected[selected.selected.astype(str).str.lower().eq('true')].copy()
    if not 1 <= len(selected) <= 4 or selected.table_index.duplicated().any():
        raise ValueError('Expected one to four distinct selected cases')
    with np.load(HERE / 'scores.npz', allow_pickle=False) as z:
        cache = {key: z[key] for key in
                 ('idx', 'refs', 'available', 'baseline_mass', 'coupled_weights', *ARMS)}
    if len(table) != 3387 or cache['idx'].shape != (1741,):
        raise ValueError('Expected completed full3387 metadata / 1741-row score cache')
    lookup = {int(full): local for local, full in enumerate(cache['idx'])}
    if len(lookup) != 1741:
        raise ValueError('Duplicate full-table positions in scores.idx')
    cases = []
    for row in selected.to_dict('records'):
        case = {key: int(row[key]) for key in
                ('table_index', 'cache_index', 'obs_id', 'true_class', 'rival_class')}
        i = case['table_index']
        if not 0 <= i < len(table) or lookup.get(i) != case['cache_index']:
            raise ValueError('Case selection full/cache index mismatch')
        if (table.iloc[i].role != 'evaluation' or int(table.iloc[i].obs_id) != case['obs_id']
                or int(table.iloc[i].class_id) != case['true_class']):
            raise ValueError('Case identity or evaluation role mismatch')
        if not 0 <= case['rival_class'] < 8 or case['true_class'] == case['rival_class']:
            raise ValueError('Shown rival must be a nontrue class')
        case.update(outcome=str(row['outcome']), rival_reason=str(row['rival_reason']))
        cases.append(case)
    with np.load(HERE / 'predictions_17.npz', allow_pickle=False) as z:
        predictions = {arm: z[arm] for arm in ('full', *ARMS)}
    transforms = json.loads((HERE / 'transforms.json').read_text(encoding='utf-8'))
    metrics = pd.read_csv(HERE / 'source_metrics.csv')
    scales, lambdas, rms_errors = {}, {}, {}
    tr = np.flatnonzero(table.iloc[cache['idx']].role.eq('utility_train').to_numpy())
    for arm in ARMS:
        chosen = metrics[(metrics.seed == SEED) & (metrics.arm == arm) & (metrics.role == 'selection')]
        if len(chosen) != 1:
            raise ValueError(f'Expected exactly one seed17 selection record for {arm}')
        lambdas[arm] = float(chosen.iloc[0].weight)
        scales[arm] = np.asarray(transforms[arm], dtype=np.float64)
        valid = cache['available'][tr]
        # Cached scalar reduction only; detect stale transforms without rerunning OT.
        rms = np.sqrt((cache[arm][tr] ** 2 * valid).sum((0, 1))
                      / np.maximum(valid.sum((0, 1)), 1))
        rms = np.maximum(rms, 1e-6)
        np.testing.assert_allclose(scales[arm], rms, rtol=1e-12, atol=1e-12)
        rms_errors[arm] = float(np.max(np.abs(scales[arm] - rms)))
    return table, cases, cache, predictions, scales, lambdas, rms_errors


def reconstruct_inputs(run, table, cases, cache):
    import numpy as np

    ids = np.array([case['table_index'] for case in cases])
    rows = np.array([case['cache_index'] for case in cases])
    refs = cache['refs'][rows]
    needed = np.unique(np.r_[ids, refs.ravel()])
    # Project just the selected queries and the actual eight-class reference set.
    with np.load(ROOT / 'tokens.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['instance_ids'].astype(str), table.instance_id.to_numpy())
        features, mass = z['layer1'][needed], z['mass'][needed]
    with np.load(ROOT / 'projection_layer1.npz', allow_pickle=False) as pr:
        x = run.unit((features - pr['mean']) @ pr['components'].T).astype(np.float32)
    x[mass <= 0] = 0
    q = np.searchsorted(needed, ids)
    mapped = np.searchsorted(needed, refs)
    np.testing.assert_array_equal(needed[mapped], refs)
    # Cached refs preserve full-table identities; mapped is only a compact read index.
    args = run.regional_inputs(x[q], mass[q], x, mass, mapped)
    return args, len(needed)


def decompose(run, model, w):
    import torch

    value, _, plan = model.values(w, return_plan=True)
    similarity = (plan * model.ss[:, None]).sum((-1, -2))
    ref = model.ref[:, None]
    kl = (plan * (plan.clamp_min(1e-30).log() - ref.clamp_min(1e-30).log())
          - plan + ref).sum((-1, -2))
    entropy = -run.EPS * kl
    torch.testing.assert_close(similarity + entropy, value, atol=2e-6, rtol=1e-6)
    nclass = model.classes
    own = torch.arange(nclass, device=w.device)
    competitors = model.available[:, None] & ~torch.eye(nclass, device=w.device, dtype=torch.bool)[None]
    rival = value.masked_fill(~competitors, -1e6).argmax(-1)

    def gap(array):
        return array[:, own, own] - array.gather(2, rival[..., None]).squeeze(-1)

    outer_kl = (w * (w.clamp_min(1e-30).log()
                    - model.a[:, None].clamp_min(1e-30).log())).sum(-1)
    terms = dict(similarity_gap=gap(similarity), entropy_kl_gap=gap(entropy),
                 Fgap=gap(value), outer_kl=outer_kl, outer_contribution=-run.PENALTY * outer_kl)
    terms['score'] = terms['Fgap'] + terms['outer_contribution']
    # No rival or numerical diagnostic is interpreted as evidence in a neutral region.
    terms = {key: torch.where(model.valid, val, 0) for key, val in terms.items()}
    terms.update(value=value, similarity=similarity, kl=kl, rival=rival, plan=plan, w=w)
    return {key: val.detach().cpu().numpy() for key, val in terms.items()}


def transport_diagnostics(case, hypothesis, regional, support_class, before, after, sim,
                          reference_ids, table):
    import numpy as np

    p, r = divmod(regional, 3)
    n = 4 if r == 2 else 8
    a = before['w'][regional, hypothesis, :n].astype(np.float64)
    w = after['w'][regional, hypothesis, :n].astype(np.float64)
    t0 = before['plan'][regional, hypothesis, support_class, :n].astype(np.float64)
    t1 = after['plan'][regional, hypothesis, support_class, :n].astype(np.float64)
    q0 = np.divide(t0, a[:, None], out=np.zeros_like(t0), where=a[:, None] > 0)
    q1 = np.divide(t1, w[:, None], out=np.zeros_like(t1), where=w[:, None] > 0)
    common = (a > 0) & (w > 0)
    if np.any((a == 0) & (w > 0)):
        raise ValueError('Saved w puts mass outside the baseline query support')
    tv = .5 * np.abs(q1 - q0).sum(-1)
    best0, best1 = int(before['rival'][regional, hypothesis]), int(after['rival'][regional, hypothesis])
    roles = [name for c, name in ((hypothesis, 'own'), (best0, 'baseline_best'),
                                  (best1, 'coupled_best')) if c == support_class]
    context = dict(obs_id=case['obs_id'], outcome=case['outcome'], hypothesis=hypothesis,
                   hypothesis_role='true' if hypothesis == case['true_class'] else 'shown_rival',
                   pol=('VV', 'VH')[p], region=REGIONS[r], support_class=support_class,
                   support_class_roles='|'.join(roles), baseline_best_rival=best0,
                   coupled_best_rival=best1, rival_changed=best0 != best1,
                   reference_1_index=int(reference_ids[0]), reference_2_index=int(reference_ids[1]),
                   reference_1_obs=int(table.iloc[int(reference_ids[0])].obs_id),
                   reference_2_obs=int(table.iloc[int(reference_ids[1])].obs_id))
    rows = []
    for slot in range(n):
        row = dict(**context, query_slot=slot, baseline_mass=float(a[slot]), new_mass=float(w[slot]),
                   status='compared' if common[slot] else 'zero_query_mass' if a[slot] == 0 else 'new_mass_zero')
        if common[slot]:
            row.update(tv_including_null=float(tv[slot]),
                       real_support_half_l1=float(.5 * abs(q1[slot, :-1] - q0[slot, :-1]).sum()),
                       baseline_row_sum=float(q0[slot].sum()), new_row_sum=float(q1[slot].sum()),
                       baseline_null=float(q0[slot, -1]), new_null=float(q1[slot, -1]))
            for k in range(2):
                row[f'baseline_ref{k+1}_prob'] = float(q0[slot, k*n:(k+1)*n].sum())
                row[f'new_ref{k+1}_prob'] = float(q1[slot, k*n:(k+1)*n].sum())
        rows.append(row)
    # Compare the same support class at both w values, including the null column.
    # T1 - T0 = (w-a) Q0 + (T1-w Q0). The latter isolates changed row conditionals.
    reweight = (w - a)[:, None] * q0
    redistribution = t1 - w[:, None] * q0
    closure = float(abs(t1 - t0 - reweight - redistribution).max())
    np.testing.assert_allclose(t1 - t0, reweight + redistribution, atol=1e-12, rtol=1e-10)
    score_matrix = np.pad(sim[regional, support_class, :n].astype(np.float64), ((0, 0), (0, 1)))
    summary = dict(**context, active=True, compared_rows=int(common.sum()),
                   baseline_mass_excluded=float(a[~common].sum()), new_mass_excluded=float(w[~common].sum()),
                   query_mass_tv=float(.5 * abs(w-a).sum()),
                   conditional_tv_a_weighted=float((a[common] * tv[common]).sum()),
                   conditional_tv_w_weighted=float((w[common] * tv[common]).sum()),
                   reallocation_half_l1=float(.5 * abs(redistribution).sum()),
                   actual_similarity_delta=float(((t1-t0) * score_matrix).sum()),
                   query_reweight_similarity_delta=float((reweight * score_matrix).sum()),
                   conditional_redistribution_similarity_delta=float((redistribution * score_matrix).sum()),
                   plan_change_closure_max_abs=closure)
    if common.any():
        summary.update(conditional_tv_max=float(tv[common].max()),
                       conditional_tv_a_mean=float((a[common] * tv[common]).sum() / a[common].sum()),
                       conditional_tv_w_mean=float((w[common] * tv[common]).sum() / w[common].sum()),
                       conditional_row_sum_error=float(max(abs(q0[common].sum(-1)-1).max(),
                                                           abs(q1[common].sum(-1)-1).max())))
    return rows, summary


def fusion_decomposition(case, cache, predictions, scales, lambdas, states):
    import numpy as np

    i, j = case['table_index'], case['cache_index']
    true, rival = case['true_class'], case['rival_class']
    valid = cache['available'][j]
    counts = np.maximum(valid.sum((1, 2)), 1)
    contributions, coefficients, replay_parts = {}, {}, {}
    base = predictions['full'][i]
    base_margin = float(base[true]) - float(base[rival])
    totals = {}
    for arm in ARMS:
        values = cache[arm][j] / scales[arm][None]
        combined = (values * valid).sum((1, 2)) / counts
        raw = combined - combined.mean(-1, keepdims=True)
        reconstructed = base.copy()
        reconstructed += lambdas[arm] * raw
        # Reproduce the actual float32 in-place fusion, not just its top1 decision.
        np.testing.assert_array_equal(reconstructed, predictions[arm][i])
        coef = lambdas[arm] * valid / scales[arm][None] / counts[:, None, None]
        coefficients[arm] = coef
        contributions[arm] = (coef[true] * cache[arm][j, true]
                              - coef[rival] * cache[arm][j, rival])
        replay_parts[arm] = {}
        for key in ('similarity_gap', 'entropy_kl_gap', 'outer_contribution'):
            term = states[arm][key].reshape(2, 3, 8).transpose(2, 0, 1)
            replay_parts[arm][key] = coef[true] * term[true] - coef[rival] * term[rival]
        actual = float(predictions[arm][i, true]) - float(predictions[arm][i, rival])
        totals[arm] = dict(actual_margin=actual, base_margin=base_margin,
                           region_contribution_sum=float(contributions[arm].sum()),
                           float32_rounding_residual=actual-base_margin-float(contributions[arm].sum()))
    rows = []
    for p, pol in enumerate(('VV', 'VH')):
        for r, region in enumerate(REGIONS):
            row = dict(obs_id=case['obs_id'], outcome=case['outcome'], true_class=true,
                       shown_rival=rival, pol=pol, region=region,
                       true_active=bool(valid[true, p, r]), rival_active=bool(valid[rival, p, r]),
                       true_region_count=int(valid[true].sum()), rival_region_count=int(valid[rival].sum()))
            for arm in ARMS:
                row[f'{arm}_lambda'] = lambdas[arm]
                row[f'{arm}_rms'] = float(scales[arm][p, r])
                row[f'{arm}_margin_contribution'] = float(contributions[arm][p, r])
                for key, term in replay_parts[arm].items():
                    row[f'{arm}_{key}_contribution'] = float(term[p, r])
                row[f'{arm}_replay_residual'] = float(contributions[arm][p, r]
                                                      - sum(v[p, r] for v in replay_parts[arm].values()))
            delta = float(contributions['coupled'][p, r] - contributions['fixed'][p, r])
            # Exact algebraic split using coupled fusion coefficients as the anchor.
            score_change = sum(sign * coefficients['coupled'][c, p, r]
                               * (float(cache['coupled'][j, c, p, r]) - float(cache['fixed'][j, c, p, r]))
                               for c, sign in ((true, 1), (rival, -1)))
            calibration_change = sum(sign * (coefficients['coupled'][c, p, r]
                                             - coefficients['fixed'][c, p, r]) * cache['fixed'][j, c, p, r]
                                     for c, sign in ((true, 1), (rival, -1)))
            np.testing.assert_allclose(delta, score_change+calibration_change, atol=1e-10, rtol=1e-10)
            row.update(margin_delta_coupled_minus_fixed=delta,
                       local_score_change_at_coupled_coeff=float(score_change),
                       rms_lambda_change_on_fixed_score=float(calibration_change))
            rows.append(row)
    delta = totals['coupled']['actual_margin'] - totals['fixed']['actual_margin']
    summed = sum(row['margin_delta_coupled_minus_fixed'] for row in rows)
    rounding = (totals['coupled']['float32_rounding_residual']
                - totals['fixed']['float32_rounding_residual'])
    np.testing.assert_allclose(delta, summed+rounding, atol=1e-10, rtol=1e-10)
    summary = dict(obs_id=case['obs_id'], true_class=true, shown_rival=rival, arms=totals,
                   actual_margin_delta=delta, regional_delta_sum=summed,
                   rounding_delta=rounding, closure_abs=float(abs(delta-summed-rounding)),
                   saved_float32_predictions_exact=True)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    args = parser.parse_args()
    required = [HERE / name for name in ('analysis/case_selection.csv', 'scores.npz',
                'source_metrics.csv', 'transforms.json', 'predictions_17.npz')]
    required += [UP / 'table.csv', ROOT / 'tokens.npz', ROOT / 'projection_layer1.npz']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error('Inputs not ready; inspection not started:\n' + '\n'.join(missing))
    import numpy as np
    import pandas as pd
    import torch

    spec = importlib.util.spec_from_file_location('competitive_matching_run', HERE / 'run.py')
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    table, cases, cache, predictions, scales, lambdas, rms_errors = load_inputs()
    regional, projected_n = reconstruct_inputs(run, table, cases, cache)
    term_rows, tv_rows, tv_summaries, region_rows, receipts = [], [], [], [], []
    with torch.inference_mode():
        for position, case in enumerate(cases):
            j = case['cache_index']
            take = slice(position*6, (position+1)*6)
            sim, a, b = [torch.tensor(v[take], device=args.device, dtype=torch.float32) for v in regional[:3]]
            available = torch.tensor(regional[3][take], device=args.device, dtype=torch.bool)
            model = run.Matching(sim, a, b, available)
            np.testing.assert_allclose(a.cpu().numpy().reshape(2, 3, 8), cache['baseline_mass'][j],
                                       atol=1e-7, rtol=1e-6)
            valid = model.valid.cpu().numpy()
            np.testing.assert_array_equal(valid.reshape(2, 3, 8).transpose(2, 0, 1), cache['available'][j])
            new_w = cache['coupled_weights'][j].transpose(1, 2, 0, 3).reshape(6, 8, 8)
            states = dict(fixed=decompose(run, model, a[:, None].expand(-1, 8, -1)),
                          coupled=decompose(run, model, torch.tensor(new_w, device=args.device, dtype=torch.float32)))
            checks = {}
            for arm, state in states.items():
                replay = state['score'].reshape(2, 3, 8).transpose(2, 0, 1)
                np.testing.assert_allclose(replay, cache[arm][j], atol=5e-5, rtol=1e-5)
                checks[arm+'_score_max_abs'] = float(abs(replay-cache[arm][j]).max())
            for h in (case['true_class'], case['rival_class']):
                for regional_index in range(6):
                    p, r = divmod(regional_index, 3)
                    active = bool(valid[regional_index, h])
                    reason = ('active' if active else 'query_region_missing' if not bool(a[regional_index].sum() > 0)
                              else 'candidate_support_missing' if not bool(available[regional_index, h])
                              else 'fewer_than_two_available_classes')
                    context = dict(obs_id=case['obs_id'], outcome=case['outcome'], hypothesis=h,
                                   hypothesis_role='true' if h == case['true_class'] else 'shown_rival',
                                   pol=('VV', 'VH')[p], region=REGIONS[r], active=active, status=reason)
                    for arm, state in states.items():
                        row = dict(**context, arm=arm, cached_score=float(cache[arm][j, h, p, r]))
                        for key in ('similarity_gap', 'entropy_kl_gap', 'Fgap', 'outer_kl', 'outer_contribution', 'score'):
                            row[key] = float(state[key][regional_index, h])
                        if active:
                            competitor = int(state['rival'][regional_index, h])
                            row['best_competitor'] = competitor
                            for label, c in (('own', h), ('competitor', competitor)):
                                for key in ('value', 'similarity', 'kl'):
                                    row[f'{label}_{key}'] = float(state[key][regional_index, h, c])
                        term_rows.append(row)
                    if not active:
                        tv_summaries.append(context)
                        continue
                    # Retain both best competitors if they switch; never compare different columns/classes.
                    support_classes = sorted({h, *(int(states[arm]['rival'][regional_index, h]) for arm in ARMS)})
                    for c in support_classes:
                        rows, summary = transport_diagnostics(case, h, regional_index, c,
                            states['fixed'], states['coupled'], regional[0][take], cache['refs'][j, c], table)
                        tv_rows.extend(rows)
                        tv_summaries.append(summary)
            rows, fusion = fusion_decomposition(case, cache, predictions, scales, lambdas, states)
            region_rows.extend(rows)
            receipts.append(dict(case=case, checks=checks, maximum_inner_error=float(model.max_error), fusion=fusion))
            print(f"Inspected obs {case['obs_id']}: fixed/coupled cache errors {checks}", flush=True)

    out = HERE / 'analysis/matching_inspection'
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [('objective_terms.csv', term_rows), ('conditional_tv_rows.csv', tv_rows),
                       ('conditional_tv_summary.csv', tv_summaries), ('region_margin_contributions.csv', region_rows)]:
        pd.DataFrame(rows).to_csv(out / name, index=False)
    receipt = dict(seed=SEED, device=args.device, case_count=len(cases), projected_unique_rows=projected_n,
                   eps=run.EPS, penalty=run.PENALTY, input_paths=[str(path) for path in required],
                   implementation=str(HERE / 'run.py'), cases=receipts,
                   utility_rms=scales, selected_lambdas=lambdas, utility_rms_max_abs_errors=rms_errors,
                   no_training=True, no_outer_optimization=True, no_full_cohort_transport=True,
                   no_npz_export=True, no_upload=True,
                   interpretation=dict(
                       neutral='Invalid regions contribute zero and are excluded from TV and fusion denominators.',
                       Fgap='own F minus best available nonself F at the same hypothesis w; outer KL is separate.',
                       kl='Generalized KL uses fixed R, includes both dustbins and all entries; entropy term is -eps*KL.',
                       tv='Half L1 of T1/w minus T0/a over all 17 columns, including support null; no row renormalization.',
                       padding='body8/side8/end4 real query rows; end support refs occupy columns0:4 and4:8; null=16.',
                       support_comparison='Own class and union of baseline/new best competitors, each held fixed across w.',
                       zero_mass='Rows with zero denominator have undefined TV, excluded with excluded mass reported.',
                       plan_split='T1-T0=(w-a)*Q0+(T1-w*Q0); real-query rows only; null query is excluded from TV.',
                       linear_fusion='Class mean centering cancels in true-minus-shown-rival; each class uses its own valid-region count.',
                       margin_delta='Regional cached-score contributions plus explicit float32 rounding delta exactly reconstruct saved margins.',
                       attribution='Local-score change uses coupled coefficients; RMS/lambda change uses fixed local scores. This is algebra, not a causal experiment.'))
    receipt['utility_rms'] = {arm: value.tolist() for arm, value in scales.items()}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Wrote matching inspection for {len(cases)} cases to {out}', flush=True)


if __name__ == '__main__':
    main()
