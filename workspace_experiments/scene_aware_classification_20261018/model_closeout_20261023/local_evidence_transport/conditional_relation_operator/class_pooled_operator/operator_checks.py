"""Independent NumPy checks and source-only Gram diagnostics; no training."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import itertools
import json
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RELATIVE_TOLERANCE = 1e-6
FLOOR = 1e-8


def centered(a):
    a = np.asarray(a, dtype=np.float64)
    return a - a.mean(axis=0)


def pooled_operator(qb, qn, references):
    """Equal-reference moments, then one inverse; inputs are unpadded edges."""
    qb, qn = centered(qb), centered(qn)
    refs = [(centered(b), centered(n)) for b, n in references if len(b) >= 2]
    if len(qb) < 2 or not refs:
        return dict(valid=False, T=0., V=0., S=0.)
    grams = [b.T @ b / len(b) for b, _ in refs]
    g = np.mean(grams, axis=0)
    c = np.mean([b.T @ n / len(b) for b, n in refs], axis=0)
    lam = max(float(np.mean([np.trace(gr) / len(b)
                             for gr, (b, _) in zip(grams, refs)])), FLOOR)
    regularized = g + lam * np.eye(g.shape[0])
    f = np.linalg.solve(regularized, c)
    h = qb @ f
    aa = [np.linalg.solve(regularized, b.T).T for b, _ in refs]
    aa = [qb @ a.T / (len(refs) * len(b)) for a, (b, _) in zip(aa, refs)]
    t = float((2 * np.sum(qn * h) - np.sum(h * h)) / len(qb))
    v = float(sum(np.sum(n * n) / (len(b) - 1) *
                  (np.sum(a * a) - np.sum(a.sum(axis=1)**2) / len(b))
                  for a, (b, n) in zip(aa, refs)) / len(qb))
    return dict(valid=True, G=g, C=c, lam=lam, F=f, H=h, A=aa, T=t, V=v, S=t+v)


def load_fixed():
    spec = importlib.util.spec_from_file_location('independent_pooled_fixed', HERE.parent / 'run.py')
    fixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed)
    return fixed


def algebra_checks(fixed):
    rng = np.random.default_rng(47)
    qb, qn = rng.normal(size=(2, 4, 5))
    refs = [(rng.normal(size=(3, 5)), rng.normal(size=(3, 5))) for _ in range(2)]
    real = pooled_operator(qb, qn, refs)
    perms = list(itertools.permutations(range(3)))
    errors, null_scores, null_predictions = [], [], []
    for p0, p1 in itertools.product(perms, repeat=2):
        shuffled = [(refs[0][0], refs[0][1][list(p0)]),
                    (refs[1][0], refs[1][1][list(p1)])]
        out = pooled_operator(qb, qn, shuffled)
        errors.append(float(np.sum((centered(qn)-out['H'])**2)/len(qb)))
        null_scores.append(out['S'])
        null_predictions.append(out['H'])
    real_error = float(np.sum((centered(qn)-real['H'])**2)/len(qb))
    residuals = dict(
        exhaustive_risk_difference=abs(float(np.mean(errors)-real_error-real['S'])),
        exhaustive_mean_excess=abs(float(np.mean(null_scores))),
        exhaustive_mean_prediction=float(np.max(np.abs(np.mean(null_predictions, axis=0)))),
        exhaustive_prediction_energy=abs(float(np.mean([np.sum(h*h)/len(qb)
                                                        for h in null_predictions])-real['V'])),
        a_reconstruction=float(np.max(np.abs(sum(a @ centered(n)
                                  for a, (_, n) in zip(real['A'], refs))-real['H']))))
    reverse = pooled_operator(qb[::-1], qn[::-1], [(b[::-1], n[::-1]) for b, n in refs])
    signs = np.array([1., -1., 1., -1., -1.])
    signed = pooled_operator(qb*signs, qn*signs, [(b*signs, n*signs) for b, n in refs])
    shifted = pooled_operator(qb+5, qn-2, [(b+7, n-9) for b, n in refs])
    residuals.update(axis_order_reversal=abs(reverse['S']-real['S']),
                     common_feature_sign_flip=abs(signed['S']-real['S']),
                     additive_offsets=abs(shifted['S']-real['S']))
    for m in (2, 3, 4, 8):
        b, n = rng.normal(size=(2, m, 5))
        new = pooled_operator(qb, qn, [(b, n)])
        old = fixed.transfer(qb, qn, b, n)
        residuals[f'single_reference_m{m}'] = max(
            abs(new['T']-float(old['transfer'].mean())),
            abs(new['V']-float(old['null'].mean())),
            abs(new['S']-float(old['excess'].mean())),
            float(np.max(np.abs(new['H']-old['prediction']))))
    const = pooled_operator(qb, qn, [(b, np.ones_like(n)*3) for b, n in refs])
    residuals['constant_reference_neighbors'] = max(abs(const[k]) for k in ('T', 'V', 'S'))
    zero_body = pooled_operator(qb, qn, [(np.zeros_like(b), n) for b, n in refs])
    residuals['constant_reference_body'] = abs(zero_body['S'])
    duplicate = pooled_operator(qb, qn, refs*2)
    residuals['duplicating_refs_preserves_lambda'] = abs(real['lam']-duplicate['lam'])
    residuals['duplicating_refs_preserves_transfer'] = abs(real['T']-duplicate['T'])
    residuals['independent_duplicate_null_halves'] = abs(real['V']/2-duplicate['V'])
    unequal = [(rng.normal(size=(m, 5)), rng.normal(size=(m, 5))) for m in (2, 5)]
    un = pooled_operator(qb, qn, unequal)
    residuals['unequal_m_lambda'] = abs(un['lam']-max(float(np.mean([
        np.sum(centered(b)**2)/len(b)**2 for b, _ in unequal])), FLOOR))
    residuals['unequal_m_reconstruction'] = float(np.max(np.abs(un['H']-sum(
        a @ centered(n) for a, (_, n) in zip(un['A'], unequal)))))
    one_invalid = pooled_operator(qb, qn, refs+[(qb[:1], qn[:1])])
    residuals['invalid_reference_excluded_from_k'] = abs(one_invalid['S']-real['S'])
    assert not pooled_operator(qb[:1], qn[:1], refs)['valid']
    assert not pooled_operator(qb, qn, [])['valid']
    # A deterministic related-neighborhood example detects broken pairing sensitivity.
    b = rng.normal(size=(3, 5))
    paired = pooled_operator(b, b, [(b, b), (b, b)])
    shuffled = pooled_operator(b, b, [(b, b[[1, 2, 0]]), (b, b[[2, 0, 1]])])
    assert paired['S'] > shuffled['S'] + 1e-3
    assert max(residuals.values()) < 1e-11, residuals
    # The requested covariance floor and the legacy dual floor are different units.
    tiny_b = rng.normal(size=(3, 5))*1e-5
    tiny_n = rng.normal(size=(3, 5))
    tiny_q = rng.normal(size=(4, 5))*1e-5
    tiny = pooled_operator(tiny_q, qn, [(tiny_b, tiny_n)])
    old_tiny = fixed.transfer(tiny_q, qn, tiny_b, tiny_n)
    floor_difference = abs(tiny['S']-float(old_tiny['excess'].mean()))
    assert floor_difference > 1e-8
    # A constant QUERY neighborhood alone need not have zero excess.
    const_query = pooled_operator(qb, np.zeros_like(qn), refs)
    return dict(status='passed_with_documented_legacy_floor_difference', seed=47,
                exact_permutations=36, residuals=residuals,
                paired_score=paired['S'], shuffled_score=shuffled['S'],
                query_constant_score=const_query['S'],
                legacy_floor=dict(m=3, pooled_covariance_lambda=tiny['lam'],
                                  legacy_covariance_lambda=FLOOR/3,
                                  excess_absolute_difference=floor_difference),
                null_assumption='Independent uniform row permutations within each reference; fixed B/query/masks.')


def gram_stats(g, lam):
    eigenvalues = np.maximum(np.linalg.eigvalsh(g), 0.)
    total = float(eigenvalues.sum())
    maximum = float(eigenvalues[-1])
    positive = eigenvalues[eigenvalues > 0]
    probabilities = positive/total if total else positive
    return dict(numerical_rank=int(np.sum(eigenvalues > RELATIVE_TOLERANCE*maximum)),
                effective_rank=float(np.exp(-np.sum(probabilities*np.log(probabilities)))) if total else 0.,
                participation_rank=float(total**2/np.sum(eigenvalues**2)) if total else 0.,
                regularized_condition=float((maximum+lam)/(float(eigenvalues[0])+lam)),
                trace=total, eigenvalue_max=maximum,
                rank_absolute_threshold=RELATIVE_TOLERANCE*maximum,
                regularization_lambda=float(lam),
                eigenvalues=json.dumps(eigenvalues.tolist()))


def reader_checks(edge, x, mass, bank):
    """Read the live implementation, but never call its fit/main/smoke paths."""
    import torch
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('pooled_readonly_review', HERE/'run.py')
    mainrun = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mainrun)
    bank_ids = bank.ravel()
    x, mass = x[bank_ids], mass[bank_ids]
    bank = np.arange(64).reshape(8, 8)
    b0, n0, legal0, _ = edge.edges(x, mass)
    missing = np.flatnonzero(~legal0[:, 2].any(-1))
    ids = np.unique(np.r_[0, 1, missing[:1]])
    rng = np.random.default_rng(71)
    maps = [('identity', np.eye(32), np.eye(32)),
            ('nonidentity', np.eye(32)+rng.normal(size=(32, 32))*.03,
             np.eye(32)+rng.normal(size=(32, 32))*.03)]
    errors, max_trace = {}, 0.
    for partial in (False, True):
        legal = legal0.copy()
        if partial:
            legal[0, 0, 0] = False
            legal[1, 1, :3] = False
            legal[bank[7], 3] = False
        for arm in mainrun.ARMS:
            shuffle = 101 if 'shuffle' in arm else None
            b, n, _, _ = edge.edges(x, mass, shuffle)
            reader = mainrun.Reader(x, mass, bank, arm, device='cpu', dtype=torch.float64,
                                    legal_override=legal)
            for name, mb, mn in maps:
                with torch.no_grad():
                    z, raw, valid, detail = reader.scores(ids, torch.tensor(mb), torch.tensor(mn), trace=True)
                    weight_dtype = str(detail['weight'].dtype)
                expected = np.zeros(raw.shape)
                expected_valid = np.zeros(valid.shape, bool)
                for gi in range(4):
                    for cls in range(8):
                        refs = [(b[r, gi, legal[r, gi]]@mb, n[r, gi, legal[r, gi]]@mn)
                                for r in bank[cls] if legal[r, gi].sum() >= 2]
                        for qi, index in enumerate(ids):
                            if legal[index, gi].sum() < 2 or not refs:
                                continue
                            qb, qn = b[index, gi, legal[index, gi]]@mb, n[index, gi, legal[index, gi]]@mn
                            independent = pooled_operator(qb, qn, refs)
                            max_trace = max(max_trace, abs(independent['T']-float(detail['T'][gi, qi, cls])),
                                            abs(independent['V']-float(detail['V'][gi, qi, cls])))
                            if arm == 'independent':
                                value = .5*(pooled_operator(qb, qb, [(br, br) for br, _ in refs])['T']+
                                            pooled_operator(qn, qn, [(nr, nr) for _, nr in refs])['T'])
                            elif arm == 'covariance':
                                cq = centered(qb).T@centered(qn)/len(qb)
                                value = 2*float(np.sum(cq*independent['C']))
                            else:
                                value = independent['T' if arm == 'transfer' else 'S']
                            expected[qi, cls, gi] = value
                            expected_valid[qi, cls, gi] = True
                np.testing.assert_array_equal(valid.numpy(), expected_valid)
                expected_z = np.where(expected_valid, expected-expected.sum(1, keepdims=True)/
                                      np.maximum(expected_valid.sum(1, keepdims=True), 1), 0.)
                err = max(float(np.max(np.abs(raw.numpy()-expected))),
                          float(np.max(np.abs(z.numpy()-expected_z))))
                assert err < 1e-10, (partial, arm, name, err)
                errors[f'{arm}_{name}_partial{partial}'] = err
    # Exercise Reader.scores itself over all 3! x 3! independent permutations.
    synthetic = mainrun.Reader.__new__(mainrun.Reader)
    synthetic.arm = 'excess'
    bs, ns = np.zeros((4, 65, 8, 32)), np.zeros((4, 65, 8, 32))
    counts = np.zeros((4, 65), dtype=np.int64)
    for index, width in [(0, 3), (1, 3), (64, 4)]:
        bs[0, index, :width] = centered(rng.normal(size=(width, 32)))
        ns[0, index, :width] = centered(rng.normal(size=(width, 32)))
        counts[0, index] = width
    synthetic.b, synthetic.n = torch.tensor(bs), torch.tensor(ns)
    synthetic.counts = torch.tensor(counts)
    synthetic.sb = synthetic.b[:, :64].clone()
    synthetic.sn = synthetic.n[:, :64].clone()
    synthetic.ms = synthetic.counts[:, :64]
    synthetic.divisor = torch.ones(4, dtype=torch.float64)
    identity = torch.eye(32, dtype=torch.float64)
    original = pooled_operator(bs[0, 64, :4], ns[0, 64, :4],
                               [(bs[0, r, :3], ns[0, r, :3]) for r in (0, 1)])
    scores, sse, max_null = [], [], 0.
    for p0, p1 in itertools.product(itertools.permutations(range(3)), repeat=2):
        for r, p in enumerate((p0, p1)):
            synthetic.sn[0, r, :3] = torch.tensor(ns[0, r, list(p)])
        with torch.no_grad():
            _, raw, _, trace = synthetic.scores([64], identity, identity, trace=True)
        ref = [(bs[0, r, :3], synthetic.sn[0, r, :3].numpy()) for r in (0, 1)]
        npout = pooled_operator(bs[0, 64, :4], ns[0, 64, :4], ref)
        scores.append(float(raw[0, 0, 0]))
        sse.append(float(np.sum((ns[0, 64, :4]-npout['H'])**2)/4))
        max_null = max(max_null, abs(float(trace['V'][0, 0, 0])-npout['V']),
                       abs(scores[-1]-npout['S']))
    actual_sse = float(np.sum((ns[0, 64, :4]-original['H'])**2)/4)
    mean_excess = abs(float(np.mean(scores)))
    risk_error = abs(float(np.mean(sse)-actual_sse-original['S']))
    assert max(max_trace, max_null, mean_excess, risk_error) < 1e-10
    return dict(status='passed', source_bank_query_slots=ids.tolist(), query_roles_used='sourcebank64 only',
                arm_map_mask_errors=errors, trace_T_V_max_error=max_trace,
                production_integer_count_weight_dtype=weight_dtype,
                precision_note='Live Reader with original integer counts/ms; no precision workaround or main file changes.',
                reader_36_permutations_mean_excess_error=mean_excess,
                reader_36_permutations_numpy_max_error=max_null,
                reader_36_permutations_risk_difference_error=risk_error,
                includes_nonidentity_maps=True, includes_missing_VH=True,
                includes_partial_edges_and_all_invalid_class=True,
                D_identity='sum_r gamma_r/(K*m_r)^2 * (Br.T@Br - (Br.T@1)(Br.T@1).T/m_r)')


def rank_diagnostics(fixed):
    edge = fixed.edge
    table, x, mass, bank = edge.support.load()
    fit = np.load(edge.support.UP/'indices.npz')['fit_train']
    eligible = fit[mass[fit, :, :8].sum((1, 2)) > 0]
    pool = table.iloc[eligible].drop_duplicates('mmsi').index.to_numpy()
    queries = table.loc[table.role.isin(['utility_train', 'selection', 'evaluation'])]
    for ids in (pool, bank.ravel()):
        assert table.iloc[ids].role.eq('strategy_fit').all()
        for key in ('mmsi', 'pass_group'):
            assert not table.iloc[ids][key].isin(queries[key]).any(), key
    # Restrict edge extraction and all numerical summaries to authorized source rows.
    source_ids = np.unique(np.r_[pool, bank.ravel()])
    b, n, legal, _ = edge.edges(x[source_ids], mass[source_ids])
    bflip, nflip, lflip, _ = edge.edges(x[source_ids][:, :, edge.REV], mass[source_ids][:, :, edge.REV])
    flip_error = 0.
    for gi in range(4):
        width = len(edge.BODY[gi % 2])
        np.testing.assert_array_equal(lflip[:, gi, :width], legal[:, gi, :width][:, ::-1])
        flip_error = max(flip_error, float(np.max(np.abs(bflip[:, gi, :width]-b[:, gi, :width][:, ::-1]))),
                         float(np.max(np.abs(nflip[:, gi, :width]-n[:, gi, :width][:, ::-1]))))
    assert flip_error == 0.
    cache, rows, summaries = {}, [], []
    for gi, group in enumerate(edge.GROUPS):
        for local, index in enumerate(source_ids):
            mask = legal[local, gi]
            m = int(mask.sum())
            if m < 2:
                continue
            bc, nc = centered(b[local, gi, mask]), centered(n[local, gi, mask])
            g, c = bc.T@bc/m, bc.T@nc/m
            lam = max(float(np.trace(g)/m), FLOOR)
            cache[gi, int(index)] = (g, c, m, lam)
        for scope, candidates in [('sourcebank64', bank.ravel()), ('source_fit_train_canonical', pool)]:
            for cls in range(8):
                ids = [int(i) for i in candidates if int(table.iloc[i].class_id) == cls and (gi, int(i)) in cache]
                assert ids, (scope, group, cls)
                gs = np.stack([cache[gi, i][0] for i in ids])
                cs = np.stack([cache[gi, i][1] for i in ids])
                ms = np.array([cache[gi, i][2] for i in ids])
                g = gs.mean(0)
                lam = max(float(np.mean([np.trace(gr)/m for gr, m in zip(gs, ms)])), FLOOR)
                stats = gram_stats(g, lam)
                per = []
                outside = []
                for i, gr in zip(ids, gs):
                    _, cr, m, lr = cache[gi, i]
                    sr = gram_stats(gr, lr)
                    per.append(sr)
                    ev, vec = np.linalg.eigh(gr)
                    basis = vec[:, ev > RELATIVE_TOLERANCE*max(float(ev[-1]), 0.)]
                    fraction = 1-float(np.trace(basis.T@g@basis))/stats['trace'] if stats['trace'] else 0.
                    outside.append(max(fraction, 0.))
                    row = table.iloc[i]
                    rows.append(dict(scope=scope, group=group, class_id=cls, class_name=row.class_name,
                                     record_type='per_reference', table_index=i, obs_id=int(row.obs_id),
                                     mmsi=row.mmsi, pass_group=row.pass_group, port=row.port,
                                     valid_references=1, legal_edges=m, relative_tolerance=RELATIVE_TOLERANCE,
                                     **sr, pooled_energy_outside_this_reference_span=outside[-1],
                                     crosscov_norm=float(np.linalg.norm(cr))))
                norm_mean = float(np.mean(np.linalg.norm(cs, axis=(1, 2))))
                summary = dict(scope=scope, group=group, class_id=cls, class_name=table.iloc[ids[0]].class_name,
                               record_type='pooled', valid_references=len(ids),
                               candidate_references=int(np.sum(table.iloc[candidates].class_id.to_numpy()==cls)),
                               unique_mmsi=int(table.iloc[ids].mmsi.nunique()),
                               unique_passes=int(table.iloc[ids].pass_group.nunique()),
                               legal_edges=int(ms.sum()), relative_tolerance=RELATIVE_TOLERANCE, **stats,
                               per_reference_rank_min=min(s['numerical_rank'] for s in per),
                               per_reference_rank_mean=float(np.mean([s['numerical_rank'] for s in per])),
                               per_reference_rank_max=max(s['numerical_rank'] for s in per),
                               per_reference_effective_rank_mean=float(np.mean([s['effective_rank'] for s in per])),
                               per_reference_condition_mean=float(np.mean([s['regularized_condition'] for s in per])),
                               rank_gain_over_best_reference=stats['numerical_rank']-max(s['numerical_rank'] for s in per),
                               pooled_energy_outside_reference_span_min=float(np.min(outside)),
                               pooled_energy_outside_reference_span_mean=float(np.mean(outside)),
                               crosscov_norm=float(np.linalg.norm(cs.mean(0))),
                               crosscov_norm_mean=norm_mean,
                               crosscov_direction_retention=float(np.linalg.norm(cs.mean(0))/norm_mean) if norm_mean else 0.,
                               reference_indices=json.dumps(ids))
                rows.append(summary)
                summaries.append(summary)
    pd.DataFrame(rows).to_csv(HERE/'rank_diagnostics.csv', index=False)
    floor_count = sum(item[3] == FLOOR for item in cache.values())
    return dict(status='passed', relative_tolerance=RELATIVE_TOLERANCE,
                rank_definition='count(lambda_i(G) > 1e-6 * lambda_max(G)); float64 centered Gram',
                effective_rank_definition='exp(-sum(p_i log(p_i))), p_i=lambda_i/trace(G)',
                condition_definition='(lambda_max(G)+ridge)/(lambda_min(G)+ridge)',
                source_fit_rows=len(fit), source_canonical_vessels=len(pool), original_bank_rows=bank.size,
                source_rows_analyzed=len(source_ids), query_feature_rows_analyzed=0,
                source_mmsi_and_pass_disjoint_from_queries=True,
                feature_axis_reversal_max_error=flip_error,
                per_reference_floor_trigger_count=floor_count,
                csv_rows=len(rows), pooled_rows=len(summaries), summaries=summaries,
                live_reader_checks=reader_checks(edge, x, mass, bank))


def main():
    tick = time.perf_counter()
    fixed = load_fixed()
    algebra = algebra_checks(fixed)
    print('Algebra checks:', json.dumps(algebra), flush=True)
    ranks = rank_diagnostics(fixed)
    result = dict(algebra=algebra, data=ranks, seconds=time.perf_counter()-tick,
                  scope='NumPy algebra and original sourcebank64/canonical fit_train diagnostics only; no training.')
    (HERE/'checks.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('Complete:', ranks['csv_rows'], 'rows;', result['seconds'], 'seconds', flush=True)
    print(pd.DataFrame(ranks['summaries'])[['scope', 'group', 'class_id', 'valid_references',
          'numerical_rank', 'effective_rank', 'regularized_condition', 'rank_gain_over_best_reference',
          'crosscov_direction_retention']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
