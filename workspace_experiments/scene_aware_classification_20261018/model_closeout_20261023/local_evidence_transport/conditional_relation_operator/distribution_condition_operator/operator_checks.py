"""Independent NumPy oracles; no training, bandwidth fitting, or Git actions."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOL = 1e-10


def error(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def close(a, b, label, tol=TOL):
    err = error(a, b)
    if not np.isfinite(err) or err > tol:
        raise AssertionError((label, err, tol))
    return err


def kernel(x, y, kind='rbf', width=2.):
    if kind == 'linear':
        return x @ y.T
    # Direct coordinate differences independently check the production norm trick.
    return np.exp(-np.sum((x[:, None] - y[None]) ** 2, axis=-1) / width)


def centered(block):
    return block - block.mean(0, keepdims=True) - block.mean(1, keepdims=True) + block.mean()


def grams(q, refs, kind='rbf', width=2.):
    """Compact legal tokens, with centering by explicit block means."""
    qs = np.concatenate([centered(kernel(q, s, kind, width)) for s in refs], axis=1)
    ss = np.block([[centered(kernel(s, t, kind, width)) for t in refs] for s in refs])
    return qs, ss, centered(kernel(q, q, kind, width))


def coefficients(q, refs, width, covariance=False):
    qs, ss, _ = grams(q, refs, width=width)
    sizes = np.array([len(s) for s in refs])
    weights = np.concatenate([np.full(m, 1. / (len(refs) * m)) for m in sizes])
    ridge = max(sum(np.trace(block) / (len(refs) * m * m)
                    for block, m in zip(split_diagonal(ss, sizes), sizes)), 1e-8)
    if covariance:
        return qs * weights
    # Unwhitened compact system, unlike the production padded sqrt(W) solve.
    return np.linalg.solve(ss + ridge * np.diag(1. / weights), qs.T).T


def split_diagonal(matrix, sizes):
    cuts = np.r_[0, np.cumsum(sizes)]
    return [matrix[a:b, a:b] for a, b in zip(cuts[:-1], cuts[1:])]


def risk_terms(a, qs, ss, qq, sizes):
    nq = len(a)
    transfer = float((2 * np.sum(a * qs) - np.trace(a @ ss @ a.T)) / nq)
    variance = 0.
    offset = 0
    for block, m in zip(split_diagonal(ss, sizes), sizes):
        ar = a[:, offset:offset + m]
        variance += np.trace(block) / (m - 1) * np.sum((ar - ar.mean(1, keepdims=True)) ** 2)
        offset += m
    variance = float(variance / nq)
    return dict(T=transfer, V=variance, S=transfer + variance,
                risk=float(np.trace(qq) / nq - transfer))


def finite_features(qq, qs, ss):
    """One joint eigensystem preserves query/support and cross-reference geometry."""
    joint = np.block([[qq, qs], [qs.T, ss]])
    values, vectors = np.linalg.eigh(joint)
    assert values.min() > -TOL, values.min()
    features = vectors * np.sqrt(np.maximum(values, 0))[None]
    reconstruction = close(features @ features.T, joint, 'finite Gram reconstruction')
    return features[:len(qq)], features[len(qq):], reconstruction


def synthetic_checks():
    rng = np.random.default_rng(71024)
    qb, qn = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    sb, sn = rng.normal(size=(2, 3, 3)), rng.normal(size=(2, 3, 3))
    a = coefficients(qb, sb, 3.)
    qs, ss, qq = grams(qn, sn, width=3.)
    terms = risk_terms(a, qs, ss, qq, [3, 3])
    qf, sf, gram_error = finite_features(qq, qs, ss)
    prediction = a @ sf
    real_risk = float(np.sum((qf - prediction) ** 2) / len(qf))
    feature_error = max(close(real_risk, terms['risk'], 'explicit risk'),
                        close((2 * np.sum(qf * prediction) - np.sum(prediction ** 2)) / len(qf),
                              terms['T'], 'explicit T'))
    risks, scores = [], []
    for p1, p2 in itertools.product(itertools.permutations(range(3)), repeat=2):
        p = np.r_[p1, np.array(p2) + 3]
        risks.append(float(np.sum((qf - a @ sf[p]) ** 2) / len(qf)))
        scores.append(risk_terms(a, qs[:, p], ss[np.ix_(p, p)], qq, [3, 3])['S'])
    null_error = max(close(np.mean(risks) - real_risk, terms['S'], '36-permutation excess'),
                     close(np.mean(risks), np.sum(qf ** 2) / len(qf) + terms['V'], '36-permutation null risk'),
                     close(np.mean(scores), 0., 'mean null score'))
    # Separate reference energies incorrectly discard an observable cross term.
    h0, h1 = a[:, :3] @ sf[:3], a[:, 3:] @ sf[3:]
    omitted_cross = float(2 * np.sum(h0 * h1) / len(qf))
    assert abs(omitted_cross) > 1e-6, omitted_cross
    qp, rp = [2, 0, 3, 1], [[2, 0, 1], [1, 2, 0]]
    sbp = [sb[r, rp[r]] for r in (1, 0)]
    snp = [sn[r, rp[r]] for r in (1, 0)]
    ap = coefficients(qb[qp], sbp, 3.)
    reordered = risk_terms(ap, *grams(qn[qp], snp, width=3.), [3, 3])
    invariance = max(close(reordered[k], terms[k], 'paired reordering ' + k) for k in terms)
    cq = np.repeat([[2., -1., 4.]], 4, axis=0)
    cs = [np.repeat([[v, 1., -2.]], 3, axis=0) for v in (3., -4.)]
    constant_grams = grams(cq, cs, width=3.)
    constant_terms = risk_terms(a, *constant_grams, [3, 3])
    constant_error = max([close(v, 0., 'constant RKHS output') for v in constant_grams] +
                         [close(v, 0., 'constant RKHS risk') for v in constant_terms.values()])
    assert np.max(np.abs(kernel(qn - qn.mean(0), sn[0] - sn[0].mean(0)).sum(0))) > .1
    return dict(seed=71024, reference_edges=[3, 3], permutations=len(risks),
                gram_eigendecomposition_error=gram_error, explicit_feature_error=feature_error,
                exact_expectation_error=null_error, paired_reordering_error=invariance,
                centered_constant_error=constant_error, omitted_cross_energy=omitted_cross,
                T=terms['T'], V=terms['V'], S=terms['S'], real_risk=real_risk,
                enumerated_null_risk=float(np.mean(risks)))


def variance_example():
    b = np.array([-1., -1., 1., 1.])[:, None]
    outputs = [np.array([3., -3., 1., -1.])[:, None],
               np.array([1., -1., 3., -3.])[:, None]]
    close(np.sort(outputs[0], axis=0), np.sort(outputs[1], axis=0), 'same N marginal')
    for n in outputs:
        close([n[b[:, 0] == t].mean() for t in (-1., 1.)], 0., 'conditional linear mean')
    a = coefficients(b, [b], 2.)
    tables = {}
    for kind in ('linear', 'rbf'):
        tables[kind] = np.array([[risk_terms(a, *grams(q, [s], kind, 2.), [4])['S']
                                 for s in outputs] for q in outputs])
    linear_error = close(tables['linear'], tables['linear'][0, 0], 'linear output indistinguishable')
    linear_transfer_error = max(close(risk_terms(a, *grams(q, [s], 'linear', 2.), [4])['T'],
                                      0., 'linear transfer zero') for q in outputs for s in outputs)
    margins = np.diag(tables['rbf']) - tables['rbf'][[0, 1], [1, 0]]
    assert np.all(margins > .01), margins
    return dict(B=b.ravel().tolist(), class_N=[n.ravel().tolist() for n in outputs],
                conditional_variances=[[9, 1], [1, 9]], input_width=2., output_width=2.,
                linear_score_matrix=tables['linear'].tolist(), linear_max_error=linear_error,
                linear_transfer_zero_error=linear_transfer_error,
                rbf_score_matrix=tables['rbf'].tolist(), rbf_correct_class_margins=margins.tolist(),
                interpretation='Representation capability example only; no SAR benefit or causal claim.')


def actual_reader_checks():
    import torch
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('distribution_main_under_review', HERE / 'run.py')
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    table, x, mass, bank, idx, tr, va, te = main.exp.source_data()
    _, _, legal, _ = main.exp.edge.edges(x, mass)
    utility = idx[tr]
    counts = legal[utility].sum(-1)
    capacities = np.array([len(main.exp.edge.BODY[g % 2]) for g in range(4)])
    partial = utility[((counts > 0) & (counts < capacities)).any(-1)]
    missing = utility[(counts[:, 2:] == 0).any(-1)]
    assert len(partial), 'No actual partial-mask utility sample found'
    ids = np.unique(np.r_[utility[:1], partial[:2], missing[:1]])
    rng = np.random.default_rng(614)
    mb = np.eye(32) + .03 * rng.normal(size=(32, 4)) @ rng.normal(size=(4, 32))
    mn = np.eye(32) + .03 * rng.normal(size=(32, 4)) @ rng.normal(size=(4, 32))
    transforms = [torch.tensor(m, dtype=torch.float64) for m in (mb, mn)]
    rows = []
    for override_name in ('natural', 'partial_boundary'):
        override = legal.copy()
        if override_name == 'partial_boundary':
            override[ids[0], 0, 0] = False
            override[ids[-1], 2:] = False
            override[bank[0, 0], 0, 1:4] = False
            override[bank[0, 1], 0] = False
            override[bank[0, 1], 0, 0] = legal[bank[0, 1], 0, 0]
            override[bank[7], 3] = False
        for kind in ('linear', 'rbf'):
            for arm in main.ARMS:
                reader = main.Reader(x, mass, bank, arm, device='cpu', dtype=torch.float64,
                                     output_kernel=kind, legal_override=override)
                with torch.no_grad():
                    z, raw, valid, trace = reader.scores(ids, *transforms, trace=True)
                d = {k: v.detach().cpu().numpy() for k, v in trace.items()}
                expected = np.zeros(tuple(raw.shape))
                expected_valid = np.zeros(tuple(valid.shape), dtype=bool)
                errs = dict(A=0., Lqs=0., Lss=0., Lqq=0., T=0., V=0., gamma=0., explicit_risk=0.)
                combinations = 0
                for g in range(4):
                    for c in range(8):
                        masks = override[bank[c], g]
                        active = [r for r in range(8) if masks[r].sum() >= 2]
                        positions = np.concatenate([8*r + np.flatnonzero(masks[r]) for r in active]) if active else np.array([], int)
                        sb = [d['sb'][g, c, r][masks[r]] for r in active]
                        sn = [d['sn'][g, c, r][masks[r]] for r in active]
                        for qi, sample in enumerate(ids):
                            qi_mask = override[sample, g]
                            qpos = np.flatnonzero(qi_mask)
                            qb, qn = d['qb'][g, qi, qi_mask], d['qn'][g, qi, qi_mask]
                            if not len(qpos):
                                errs['Lqq'] = max(errs['Lqq'], close(d['Lqq'][g, qi], 0., 'empty query Gram'))
                            if len(qpos) < 2 or not active:
                                close(raw[qi, c, g].item(), 0., 'invalid raw score')
                                if not active:
                                    for name in ('A', 'Lqs'):
                                        close(d[name][g, qi, c], 0., 'empty support ' + name)
                                    close(d['Lss'][g, c], 0., 'empty support Gram')
                                continue
                            expected_valid[qi, c, g] = True
                            widths = reader.width[g].cpu().numpy()
                            a = coefficients(qb, sb, widths[0], covariance=(arm == 'covariance'))
                            lqs, lss, lqq = grams(qn, sn, kind, widths[1])
                            sizes = [len(s) for s in sn]
                            terms = risk_terms(a, lqs, lss, lqq, sizes)
                            padded_a, padded_qs = np.zeros((8, 64)), np.zeros((8, 64))
                            padded_ss, padded_qq = np.zeros((64, 64)), np.zeros((8, 8))
                            padded_a[np.ix_(qpos, positions)] = a
                            padded_qs[np.ix_(qpos, positions)] = lqs
                            padded_ss[np.ix_(positions, positions)] = lss
                            padded_qq[np.ix_(qpos, qpos)] = lqq
                            for name, actual, wanted in (
                                ('A', d['A'][g, qi, c], padded_a),
                                ('Lqs', d['Lqs'][g, qi, c], padded_qs),
                                ('Lss', d['Lss'][g, c], padded_ss),
                                ('Lqq', d['Lqq'][g, qi], padded_qq),
                                ('T', d['T'][g, qi, c], terms['T']),
                                ('V', d['V'][g, qi, c], terms['V'])):
                                errs[name] = max(errs[name], close(actual, wanted, (override_name, kind, arm, name, g, c, int(sample))))
                            gamma = np.zeros(8)
                            for r, block, m in zip(active, split_diagonal(lss, sizes), sizes):
                                gamma[r] = np.trace(block) / (m - 1)
                            errs['gamma'] = max(errs['gamma'], close(d['gamma'][g, c], gamma, 'gamma'))
                            if c == 0 and qi == 0:
                                qf, sf, _ = finite_features(lqq, lqs, lss)
                                errs['explicit_risk'] = max(errs['explicit_risk'], close(
                                    np.sum((qf - a @ sf)**2) / len(qf), terms['risk'], 'actual explicit risk'))
                            if arm == 'independent':
                                an = coefficients(qn, sn, widths[1])
                                body = risk_terms(a, *grams(qb, sb, kind, widths[0]), sizes)['T']
                                near = risk_terms(an, lqs, lss, lqq, sizes)['T']
                                value = .5 * (body + near)
                            elif arm == 'covariance':
                                value = 2 * np.sum(a * lqs) / len(qpos)
                            else:
                                value = terms['T'] if arm == 'transfer' else terms['S']
                            expected[qi, c, g] = value
                            combinations += 1
                assert np.array_equal(valid.cpu().numpy(), expected_valid), (override_name, kind, arm, 'valid')
                errs['raw'] = close(raw.cpu().numpy(), expected, 'raw')
                mean = expected.sum(1, keepdims=True) / np.maximum(expected_valid.sum(1, keepdims=True), 1)
                expected_z = np.where(expected_valid, expected - mean, 0.)
                errs['z'] = close(z.cpu().numpy(), expected_z, 'class centered z')
                row = dict(mask=override_name, output_kernel=kind, arm=arm,
                           valid_query_class_groups=combinations, max_errors=errs)
                rows.append(row)
                print(json.dumps(row), flush=True)
                del reader, trace
    return dict(status='passed', device='cpu', dtype='float64', transform_seed=614,
                query_ids=ids.tolist(), query_mmsi=table.iloc[ids].mmsi.astype(str).tolist(),
                natural_query_edge_counts=legal[ids].sum(-1).tolist(),
                group_edge_capacities=capacities.tolist(),
                actual_partial_query_ids=np.intersect1d(ids, partial).tolist(),
                actual_missing_vh_ids=np.intersect1d(ids, missing).tolist(),
                reference_ids=bank.tolist(), records=rows, no_training=True, no_evaluation_scores=True,
                bandwidth_source=str(HERE.parent / 'kernel_condition_operator/bandwidth.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-only', action='store_true')
    args = parser.parse_args()
    result = dict(status='running', tolerance=TOL, python=sys.executable,
                  synthetic=synthetic_checks(), conditional_variance_example=variance_example())
    if args.synthetic_only:
        result['actual_reader'] = dict(status='not_run')
    else:
        result['actual_reader'] = actual_reader_checks()
    result['status'] = 'synthetic_passed' if args.synthetic_only else 'passed'
    (HERE / 'checks.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'actual_reader'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
