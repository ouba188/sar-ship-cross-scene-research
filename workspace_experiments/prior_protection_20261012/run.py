"""Freeze class-wise prior-protection baselines, then evaluate matched caches."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import time

OUT = Path(__file__).resolve().parent
MECHANISM = OUT.parent/'support_mechanism_20261012'
sys.path.insert(0, str(MECHANISM))
import run_mechanism as m
from run_mechanism import np, pd, c, BUDGET, CachedStrongFeatures, softmax, late, COLS, predict, adapt_support
from prior_rules import penalties, solve_weights

CACHE = Path('C:/Users/25791/AppData/Local/Temp/SAR_prior_protection_20261012')
DATASETS = ('f0921', 'oss2')
RULES = ('guard', 'count', 'count_common', 'confidence')
NEW_ARMS = ('prior_only', 'guard', 'count', 'count_common', 'confidence', 'initial_guard')


def freeze():
    started = time.perf_counter()
    all_penalties, replays = [], []
    for ds in DATASETS:
        directory = BUDGET/'strong_external'/ds/ds
        feature = CachedStrongFeatures(ds)
        initial = c.archive(directory/'initial.npz')
        for draw in range(5):
            for budget in c.BUDGETS:
                old = c.archive(directory/f'{draw}_{budget}.npz')
                q, s, labels = old['query_ids'], old['support_ids'], old['support_labels']
                raw, ps = feature.target(s)
                delta = predict(initial, raw[:, COLS], np.zeros(budget, int), 'image_history')
                changed = adapt_support(raw, delta, c.DOSES[budget])
                ws, lambdas = {}, {}
                counts = np.bincount(labels, minlength=8)
                for rule in ('raw',)+RULES:
                    lam = penalties(labels, ps, budget/8., rule)
                    ws[rule] = solve_weights(raw, ps, labels, lam)
                    lambdas[rule] = lam
                    for cls in range(8):
                        all_penalties.append(dict(dataset=ds, budget=budget, draw=draw,
                            rule=rule, class_id=cls, support_n=int(counts[cls]), penalty=float(lam[cls]),
                            absent=bool(counts[cls]==0)))
                ws['initial'] = solve_weights(changed, ps, labels, lambdas['raw'])
                ws['initial_guard'] = solve_weights(changed, ps, labels, lambdas['guard'])
                ws['prior_only'] = np.zeros_like(ws['raw'])
                order = ('raw', 'initial')+NEW_ARMS
                joined = np.concatenate([ws[arm] for arm in order], axis=1)
                scores = {arm: np.empty((len(q), 8)) for arm in NEW_ARMS}
                errors = dict(raw=0., initial=0.)
                for lo in range(0, len(q), 2048):
                    qi = q[lo:lo+2048]
                    qp, pq = feature.target(qi)
                    block = (qp@joined).reshape(len(qi), len(order), 8)+pq[:, None, :]
                    for j, arm in enumerate(order):
                        if arm in errors:
                            ref = old['expert_'+arm][lo:lo+len(qi)]
                            error = float(abs(block[:, j]-ref).max())
                            errors[arm] = max(errors[arm], error)
                            np.testing.assert_allclose(block[:, j], ref, atol=1e-9, rtol=1e-9)
                        else:
                            scores[arm][lo:lo+len(qi)] = block[:, j]
                dest = CACHE/ds
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest/f'{draw}_{budget}.npz', **scores, query_ids=q, support_ids=s)
                own = OUT/'weights'/ds
                own.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(own/f'{draw}_{budget}.npz', **ws, support_ids=s, support_labels=labels)
                replays.append(dict(dataset=ds, draw=draw, budget=budget,
                    raw_error=errors['raw'], initial_error=errors['initial'], absent_classes=int((counts==0).sum())))
                print('FROZEN', ds, draw, budget, errors, flush=True)
    pd.DataFrame(all_penalties).to_csv(OUT/'penalties.csv', index=False)
    pd.DataFrame(replays).to_csv(OUT/'replays.csv', index=False)
    c.write(OUT/'PREDICTIONS_FROZEN.json', dict(datasets=DATASETS, budgets=c.BUDGETS, draws=5,
        prediction_units=len(replays), new_arms=NEW_ARMS, cache=str(CACHE),
        query_labels_loaded=False, new_neural_training=0, choice_or_hyperparameter_search=False,
        implementation='Literature-inspired squared-loss SAR controls, NOT CLAP/SS-Text+ reproductions',
        runtime=sys.executable, seconds=time.perf_counter()-started))


def evaluate():
    assert c.read_json(OUT/'PREDICTIONS_FROZEN.json')['prediction_units']==30
    rows, classes, margins = [], [], []
    for ds in DATASETS:
        directory = BUDGET/'strong_external'/ds/ds
        table = pd.read_csv(c.read_json(directory/'PREDICTIONS_FROZEN.json')['target_manifest'])
        for draw in range(5):
            for budget in c.BUDGETS:
                old = c.archive(directory/f'{draw}_{budget}.npz')
                new = c.archive(CACHE/ds/f'{draw}_{budget}.npz')
                strength = c.archive(m.CACHE/ds/f'{draw}_{budget}.npz')['class_strength_match']
                np.testing.assert_array_equal(new['query_ids'], old['query_ids'])
                np.testing.assert_array_equal(new['support_ids'], old['support_ids'])
                y = table.iloc[old['query_ids']].class_id.to_numpy(int)
                counts = np.bincount(old['support_labels'], minlength=8)
                scores = {a: old['expert_'+a] for a in ('raw', 'initial', 'simple')}
                scores['class_strength_match'] = strength
                scores.update({a: new[a] for a in NEW_ARMS})
                key = dict(dataset=ds, budget=budget, draw=draw)
                raw_expert = softmax(scores['raw'], axis=1)
                for arm, score in scores.items():
                    for layer, p, bp in [('expert', softmax(score, axis=1), raw_expert),
                                         ('fusion', late(old['base'], score), old['fusion_raw'])]:
                        rows.append(dict(**key, arm=arm, layer=layer, **m.metrics(y, p, bp)))
                        pred, reference = p.argmax(1), bp.argmax(1)
                        losses = np.square(p-np.eye(8)[y]).sum(1)
                        for cls in np.unique(y):
                            mask = y==cls
                            classes.append(dict(**key, arm=arm, layer=layer, class_id=int(cls),
                                class_name=str(table.iloc[old['query_ids'][np.flatnonzero(mask)[0]]].class_name),
                                n=int(mask.sum()), support_n=int(counts[cls]), absent=bool(counts[cls]==0),
                                recall=100*float(np.mean(pred[mask]==cls)), brier=float(losses[mask].mean()),
                                rescue=int(np.sum((reference[mask]!=cls)&(pred[mask]==cls))),
                                harm=int(np.sum((reference[mask]==cls)&(pred[mask]!=cls)))))
                # Fixed competitor exposes why a frozen class score can still lose.
                wrong = scores['raw'].copy()
                at = np.arange(len(y))
                wrong[at, y] = -np.inf
                rival = wrong.argmax(1)
                for arm in ('raw', 'initial', 'guard', 'initial_guard', 'count', 'count_common', 'confidence'):
                    correction = scores[arm]-scores['prior_only']
                    for cls in np.unique(y):
                        mask = y==cls
                        margins.append(dict(**key, arm=arm, class_id=int(cls), absent=bool(counts[cls]==0),
                            true_class_correction=float(correction[mask, cls].mean()),
                            wrong_class_correction=float(correction[at[mask], rival[mask]].mean()),
                            support_margin=float((correction[at, y]-correction[at, rival])[mask].mean())))
                print('EVALUATED', ds, draw, budget, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'metrics.csv', index=False)
    identifiers = ['dataset', 'budget', 'draw', 'layer']
    values = ['oa', 'ba', 'nll', 'brier', 'macro_brier']
    pair = frame.merge(frame[frame.arm=='raw'][identifiers+values], on=identifiers,
                       suffixes=('', '_raw'), validate='many_to_one')
    for value in values:
        pair['delta_'+value] = pair[value]-pair[value+'_raw']
    pair.to_csv(OUT/'paired.csv', index=False)
    summary = pair.groupby(['dataset', 'budget', 'layer', 'arm']).agg(
        oa=('oa','mean'), ba=('ba','mean'), delta_oa=('delta_oa','mean'), delta_ba=('delta_ba','mean'),
        delta_ba_sd=('delta_ba','std'), positive_draws=('delta_ba',lambda x:int((x>1e-10).sum())),
        delta_macro_brier=('delta_macro_brier','mean'), rescue=('rescue','mean'), harm=('harm','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    cls = pd.DataFrame(classes)
    cls.to_csv(OUT/'classes.csv', index=False)
    ids = identifiers+['class_id']
    cls = cls.merge(cls[cls.arm=='raw'][ids+['recall','brier']], on=ids,
                    suffixes=('', '_raw'), validate='many_to_one')
    cls['delta_recall'] = cls.recall-cls.recall_raw
    cls['delta_brier'] = cls.brier-cls.brier_raw
    cls['delta_ba_contribution'] = cls.delta_recall/8
    cls.groupby(['dataset','budget','layer','arm','class_id','class_name']).mean(numeric_only=True).to_csv(OUT/'class_summary.csv')
    # First average classes within each draw/group, then average draws; retain denominators.
    groups = cls.groupby(identifiers+['arm','absent']).agg(
        classes=('class_id','size'), queries=('n','sum'), recall=('recall','mean'),
        delta_recall=('delta_recall','mean'), delta_brier=('delta_brier','mean'),
        contribution=('delta_ba_contribution','sum')).reset_index()
    groups.to_csv(OUT/'coverage_groups.csv', index=False)
    pd.DataFrame(margins).to_csv(OUT/'score_paths.csv', index=False)
    c.write(OUT/'COMPLETE.json', dict(status='predictions_and_metrics_complete', prediction_units=30,
        datasets=DATASETS, budgets=c.BUDGETS, new_arms=NEW_ARMS, new_neural_training=0,
        query_selection=False, prior_models_unchanged=True, all_predictions_frozen_before_evaluation=True))
    print(summary[summary.layer=='fusion'].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze','evaluate','all'])
    args = parser.parse_args()
    with c.threadpool_limits(limits=2):
        if args.stage in ('freeze','all'):
            freeze()
        if args.stage in ('evaluate','all'):
            evaluate()
