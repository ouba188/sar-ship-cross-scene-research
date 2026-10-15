"""Frozen budget study: original, learned support correction, source-chosen shrinkage."""

from full_teacher_data import BASE, ROOT, SOURCE, PORTS, write
from case_support_completion import COLS, predict, adapt_support
from unified_head_core import coefficients, score, late, class_weights
from unified_head_run import base_kernel
from evaluate_probability_layers import probability_rows, cached

import json
import numpy as np
import pandas as pd


OUT = BASE/'support_budget_20261012'
BUDGETS = (16, 64, 128)
AMOUNTS = (0., .25, .5, .75, 1.)
CONTROL = BASE/'case_completion_class_control_20261012'
TEACHER = BASE/'case_relation_headroom_20261012'


def strength(budget):
    return 8.*budget/64


def ba(truth, scores):
    correct = np.asarray(scores).argmax(1) == truth
    return 100*np.mean([correct[truth == c].mean() for c in np.unique(truth)])


def bank_scores(phi, prior, s, q, sy, model=None):
    ps = phi[s] if model is None else adapt_support(
        phi[s], predict(model, phi[s][:, COLS], np.zeros(len(s), int), 'image_history'), 1.)
    alpha = coefficients(ps@ps.T, prior[s], sy, strength(len(s)))
    return prior[q]+(phi[q]@ps.T)@alpha


def select_amount(meta):
    folds = cached(BASE/'crossfit_evidence_20261011/source_folds.npz')
    fit_ids, assignment = folds['fit_obs_ids'], folds['assignment']
    fixed = []
    for fold in range(3):
        ids = np.flatnonzero(assignment == fold)
        frame = meta.iloc[fit_ids[ids]]
        ships = np.random.default_rng(121000+fold).permutation(np.sort(frame.mmsi.unique()))
        query_ships = ships[:len(ships)//5]
        q = ids[frame.mmsi.isin(query_ships).to_numpy()]
        pool = ids[~frame.mmsi.isin(query_ships).to_numpy()]
        pool_ships = np.sort(meta.iloc[fit_ids[pool]].mmsi.unique())
        assert len(pool_ships) >= max(BUDGETS)
        for draw in range(3):
            rng = np.random.default_rng(122000+fold*10+draw)
            order = rng.permutation(pool_ships)[:max(BUDGETS)]
            s = np.array([rng.choice(pool[meta.iloc[fit_ids[pool]].mmsi.to_numpy() == ship]) for ship in order])
            assert not set(meta.iloc[fit_ids[q]].mmsi) & set(meta.iloc[fit_ids[s]].mmsi)
            fixed.append((fold, draw, s, q))
    records = []
    for context in PORTS:
        maps = cached(TEACHER/context/'maps.npz')
        np.testing.assert_array_equal(maps['obs_ids'][:len(fit_ids)], fit_ids)
        prior = cached(BASE/'case_relation_student_20261012'/context/'source_readers.npz')['fit_prior']
        phi = maps['single'][:len(fit_ids)]
        labels = meta.iloc[fit_ids].class_id.to_numpy()
        for fold, draw, full_s, q in fixed:
            for budget in BUDGETS:
                s = full_s[:budget]
                raw = bank_scores(phi, prior, s, q, labels[s])
                for a in AMOUNTS:
                    pred = prior[q]+a*(raw-prior[q])
                    records.append(dict(context=context, fold=fold, draw=draw, budget=budget,
                                        amount=a, ba=ba(labels[q], pred), n_query=len(q)))
        print('SOURCE SELECTION', context, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'source_selection.csv', index=False)
    avg = frame.groupby('amount').ba.mean().reset_index()
    choice = float(avg.sort_values(['ba', 'amount'], ascending=[False, False], kind='stable').iloc[0].amount)
    avg.to_csv(OUT/'source_selection_summary.csv', index=False)
    write(OUT/'FROZEN_CHOICE.json', dict(amount=choice, criterion='source-fold expert BA, all budgets equally weighted',
        target_labels_used=False, correction_dose=1., budgets=BUDGETS, ridge_per_example=8/64))
    print('FROZEN AMOUNT', choice, flush=True)
    return choice


def nested_source(meta, held_ids, old_supports, q):
    query_ships = set(meta.iloc[q].mmsi)
    pool = held_ids[~meta.iloc[held_ids].mmsi.isin(query_ships).to_numpy()]
    answer = []
    for draw, old_s in enumerate(old_supports):
        rng = np.random.default_rng(123000+draw)
        first = rng.permutation(old_s)
        existing = set(meta.iloc[first].mmsi)
        available = pool[~meta.iloc[pool].mmsi.isin(existing).to_numpy()]
        ships = np.sort(meta.iloc[available].mmsi.unique())
        chosen = rng.choice(ships, 64, replace=False)
        extra = [rng.choice(available[meta.iloc[available].mmsi.to_numpy() == ship]) for ship in chosen]
        order = np.r_[first, extra].astype(int)
        assert len(set(meta.iloc[order].mmsi)) == 128
        assert not set(meta.iloc[order].mmsi) & query_ships
        answer.append(order)
    return answer


def freeze_source(meta, amount):
    splits = [cached(BASE/'incidence_transport_20261011'/f'31415_source_{d}_predictions.npz') for d in range(3)]
    q_ids = np.array(sorted(set.intersection(*(set(z['query_obs_ids']) for z in splits))), dtype=int)
    cohorts = cached(BASE/'crossfit_evidence_20261011/source_folds.npz')
    # Held IDs are stored in the common map, after the original fit cohort.
    reference = cached(TEACHER/PORTS[0]/'maps.npz')
    held_ids = reference['obs_ids'][len(cohorts['fit_obs_ids']):]
    orders = nested_source(meta, held_ids, [z['support_obs_ids'] for z in splits], q_ids)
    replays = []
    for context in PORTS:
        dest = OUT/'source'/context
        dest.mkdir(parents=True, exist_ok=True)
        maps = cached(TEACHER/context/'maps.npz')
        ids, phi, prior = maps['obs_ids'], maps['single'], maps['prior']
        index = pd.Index(ids)
        q = index.get_indexer(q_ids)
        model = cached(CONTROL/context/'final_model.npz')
        y = meta.iloc[ids].class_id.to_numpy()
        for draw, order in enumerate(orders):
            for budget in BUDGETS:
                s = index.get_indexer(order[:budget])
                assert (s >= len(cohorts['fit_obs_ids'])).all() and (q >= 0).all()
                raw = bank_scores(phi, prior, s, q, y[s])
                corrected = bank_scores(phi, prior, s, q, y[s], model)
                simple = prior[q]+amount*(raw-prior[q])
                if budget == 64:
                    for arm, scores, folder, suffix in [
                        ('raw', raw, 'case_support_completion_20261012', 'raw'),
                        ('corrected', corrected, 'case_completion_class_control_20261012', 'image_class_target')]:
                        old = cached(BASE/folder/context/f'31415_{draw}_{suffix}.npz')
                        ix = pd.Index(old['query_obs_ids']).get_indexer(q_ids)
                        assert (ix >= 0).all()
                        np.testing.assert_allclose(scores, old['scores'][ix], atol=1e-10, rtol=0)
                        replays.append(dict(context=context, draw=draw, arm=arm,
                                            max_error=float(abs(scores-old['scores'][ix]).max())))
                np.savez_compressed(dest/f'{draw}_{budget}.npz', query_ids=q_ids, support_ids=order[:budget],
                    expert_raw=raw, expert_corrected=corrected, expert_simple=simple)
        print('SOURCE FROZEN', context, flush=True)
    return replays


def freeze_targets(manifest, amount):
    replays = []
    for port in PORTS:
        dest = OUT/'target'/port
        dest.mkdir(parents=True, exist_ok=True)
        folder = ROOT/'learned_frequency_residual_20261009/tasks/self24ports'/port
        data = cached(folder/'target.npz')
        emb = cached(folder/'selective_embedding.npz')
        np.testing.assert_array_equal(emb['rows'], data['rows'])
        eta = json.loads((folder/'PREPARED.json').read_text())['eta']
        old = cached(BASE/'support_completion_target_20261012'/port/'predictions.npz')
        fixed = cached(BASE/'full_teacher_integration_20261012'/port/'no_training/predictions.npz')
        for key in ('rows', 'support', 'eval_masks'):
            np.testing.assert_array_equal(data[key], old[key])
            np.testing.assert_array_equal(data[key], fixed[key])
        phi, prior = fixed['phi'], fixed['prior']
        q = np.flatnonzero(old['eval_masks'].all(0))
        pool = np.flatnonzero(~old['eval_masks'].all(0))
        model = cached(CONTROL/port/'final_model.npz')
        for draw, old_s in enumerate(old['support']):
            rng = np.random.default_rng(124000+draw)
            order = np.r_[rng.permutation(old_s), rng.choice(np.setdiff1d(pool, old_s), 64, replace=False)]
            assert len(np.unique(order)) == 128 and not np.intersect1d(order, q).size
            for budget in BUDGETS:
                s = order[:budget]
                sy = manifest.iloc[data['rows'][s]].class_id.to_numpy()
                kb = base_kernel(data, emb['embedding'].astype(float), s, eta)
                base = score(kb, data['prior'], s, sy, strength(budget))[0][q]
                raw = bank_scores(phi, prior, s, q, sy)
                corrected = bank_scores(phi, prior, s, q, sy, model)
                simple = prior[q]+amount*(raw-prior[q])
                variants = dict(base=base, expert_raw=raw, expert_corrected=corrected, expert_simple=simple,
                    fusion_raw=late(base, raw), fusion_corrected=late(base, corrected), fusion_simple=late(base, simple))
                if budget == 64:
                    for arm in ('base', 'expert_raw', 'expert_corrected', 'fusion_raw', 'fusion_corrected'):
                        expected = old[arm][draw][q]
                        np.testing.assert_allclose(variants[arm], expected, atol=1e-10, rtol=0)
                        np.testing.assert_array_equal(variants[arm].argmax(1), expected.argmax(1))
                        replays.append(dict(port=port, draw=draw, arm=arm,
                                            max_error=float(abs(variants[arm]-expected).max())))
                np.savez_compressed(dest/f'{draw}_{budget}.npz', query_ids=data['rows'][q],
                    support_ids=data['rows'][s], **variants)
        print('TARGET FROZEN', port, flush=True)
    return replays


def evaluate(meta, manifest):
    rows, classes, supports, instability, risk_parts = [], [], [], [], []
    for family, table, draws in [('source', meta, 3), ('target', manifest, 5)]:
        for context in PORTS:
            last_query = None
            for budget in BUDGETS:
                predictions = {}
                for draw in range(draws):
                    z = cached(OUT/family/context/f'{draw}_{budget}.npz')
                    q, s = z['query_ids'], z['support_ids']
                    if last_query is not None:
                        np.testing.assert_array_equal(q, last_query)
                    last_query = q
                    assert len(s) == budget and not np.intersect1d(q, s).size
                    if budget > BUDGETS[0]:
                        prev = cached(OUT/family/context/f'{draw}_{BUDGETS[BUDGETS.index(budget)-1]}.npz')
                        np.testing.assert_array_equal(s[:len(prev['support_ids'])], prev['support_ids'])
                    qm, sm = table.iloc[q], table.iloc[s]
                    truth = qm.class_id.to_numpy()
                    counts = sm.class_id.value_counts().reindex(range(8), fill_value=0)
                    info = dict(family=family, context=context, budget=budget, draw=draw)
                    supports.append(dict(**info, n_query=len(q), n_support=len(s),
                        n_support_classes=int((counts > 0).sum()), n_known_support_ships=sm.mmsi.nunique(),
                        n_missing_support_identity=int(sm.mmsi.isna().sum()),
                        query_shares_support_ship_fraction=float((qm.mmsi.notna() & qm.mmsi.isin(sm.mmsi.dropna())).mean()),
                        class_counts=json.dumps(counts.to_dict(), sort_keys=True)))
                    for arm in [k for k in z if k.startswith(('expert_', 'fusion_')) or k == 'base']:
                        baseline_arm = 'base' if arm == 'base' else arm.split('_')[0]+'_raw'
                        kind = 'probabilities' if arm.startswith('fusion') else 'logits'
                        a = probability_rows(z[arm], truth, kind)
                        b = probability_rows(z[baseline_arm], truth, kind)
                        predictions.setdefault(arm, []).append(z[arm].argmax(1))
                        cell = pd.DataFrame(dict(actual_port=qm.port.to_numpy(), class_id=truth,
                            correct=a['correct'].astype(float), nll=a['nll'], brier=a['brier'],
                            gain_brier=b['brier']-a['brier']))
                        pc = cell.groupby(['actual_port', 'class_id'])[['brier', 'gain_brier']].mean().groupby(level=0).mean().mean()
                        record = dict(**info, arm=arm, n=len(q), oa=100*a['correct'].mean(), ba=ba(truth, z[arm]),
                            delta_oa=100*(a['correct'].astype(float)-b['correct']).mean(),
                            delta_ba=ba(truth, z[arm])-ba(truth, z[baseline_arm]),
                            nll=a['nll'].mean(), brier=a['brier'].mean(),
                            delta_nll=(b['nll']-a['nll']).mean(), delta_brier=(b['brier']-a['brier']).mean(),
                            class_brier=cell.groupby('class_id').brier.mean().mean(),
                            portclass_brier=pc.brier, portclass_risk_gain=pc.gain_brier,
                            kernel_intervention=arm == 'expert_corrected')
                        rows.append(record)
                        for c in np.unique(truth):
                            mask = truth == c
                            classes.append(dict(**info, arm=arm, class_id=int(c), n_query=int(mask.sum()),
                                n_support=int(counts[c]), recall=100*a['correct'][mask].mean(),
                                delta_recall=100*(a['correct'][mask].astype(float)-b['correct'][mask]).mean()))
                        for name, mask in [('baseline_correct', b['correct']), ('baseline_wrong', ~b['correct'])]:
                            risk_parts.append(dict(**info, arm=arm, subset=name, n=int(mask.sum()),
                                delta_brier=float((b['brier'][mask]-a['brier'][mask]).mean()) if mask.any() else np.nan,
                                population_contribution=float((b['brier'][mask]-a['brier'][mask]).sum()/len(q))))
                for arm, preds in predictions.items():
                    preds = np.stack(preds)
                    instability.append(dict(family=family, context=context, budget=budget, arm=arm,
                        any_prediction_change_fraction=float(np.any(preds != preds[0], axis=0).mean())))
    frame = pd.DataFrame(rows)
    summary = []
    metrics = ['oa', 'ba', 'delta_oa', 'delta_ba', 'nll', 'brier', 'delta_nll', 'delta_brier',
               'class_brier', 'portclass_brier', 'portclass_risk_gain']
    for key, group in frame.groupby(['family', 'context', 'budget', 'arm']):
        item = dict(zip(['family', 'context', 'budget', 'arm'], key))
        item['n_draws'] = len(group)
        for metric in metrics:
            v = group[metric]
            item.update({metric+'_mean': v.mean(), metric+'_std': v.std(ddof=0),
                         metric+'_min': v.min(), metric+'_max': v.max()})
        for metric in ('oa', 'ba'):
            v = group['delta_'+metric]
            item[metric+'_wins'] = int((v > 1e-10).sum())
            item[metric+'_ties'] = int((abs(v) <= 1e-10).sum())
            item[metric+'_losses'] = int((v < -1e-10).sum())
        summary.append(item)
    summary = pd.DataFrame(summary)
    avg = summary.groupby(['family', 'budget', 'arm'])[[m+'_mean' for m in metrics]+['oa_std', 'ba_std']].mean().reset_index()
    for name, data in [('metrics', frame), ('summary', summary), ('budget_summary', avg),
                       ('classes', pd.DataFrame(classes)), ('supports', pd.DataFrame(supports)),
                       ('prediction_instability', pd.DataFrame(instability)), ('risk_parts', pd.DataFrame(risk_parts))]:
        data.to_csv(OUT/(name+'.csv'), index=False)
    print(avg[['family', 'budget', 'arm', 'oa_mean', 'ba_mean', 'delta_ba_mean',
               'portclass_risk_gain_mean', 'ba_std']].to_string(index=False), flush=True)
    return frame, summary, avg


def self_test():
    x = np.array([[1., 0., 1.], [0., 1., 1.], [2., 1., 1.], [1., 2., 1.]])
    labels = np.array([0, 0, 1, 2])
    prior_coef = np.arange(24).reshape(3, 8)/50
    prior = x@prior_coef
    w = class_weights(labels)
    response = 2*np.eye(8)[labels]-1-prior
    b = len(labels)
    primal = np.linalg.solve(x.T@(w[:, None]*x)/b+(8/64)*np.eye(3), x.T@(w[:, None]*response)/b)
    alpha = coefficients(x@x.T, prior, labels, strength(b))
    np.testing.assert_allclose(x@primal, (x@x.T)@alpha, atol=1e-12, rtol=0)
    assert strength(64) == 8 and np.isfinite(alpha).all()
    return ['normalized_loss_primal_dual_with_missing_classes', 'B64_regularizer_identity']


def main():
    OUT.mkdir(exist_ok=True)
    tests = self_test()
    meta = pd.read_csv(BASE/'observations.csv', usecols=['class_id', 'class_name', 'port', 'mmsi', 'product'])
    manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id', 'class_name', 'port', 'mmsi', 'product'])
    amount = select_amount(meta)
    checks = dict(source=freeze_source(meta, amount), target=freeze_targets(manifest, amount))
    write(OUT/'REPLAY.json', checks)
    write(OUT/'PREDICTIONS_FROZEN.json', dict(budgets=BUDGETS, targets=list(PORTS), source_contexts=list(PORTS)))
    frame, summary, avg = evaluate(meta, manifest)
    write(OUT/'COMPLETE.json', dict(status='complete', neural_fits=0, new_source_model_fits=0,
        support_coefficients_resolved=True, amount=amount, budgets=BUDGETS, target_episodes=45,
        source_held_episodes=27, B64_replay_count=sum(map(len, checks.values())), tests=tests,
        target_parameter_selection=False, probability_temperature=1., interpretation='Historical exploratory conditional budget study'))


if __name__ == '__main__':
    main()
