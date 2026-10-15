"""Frozen classwise anchoring expansion using existing port-local operators."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import gc
import time
import numpy as np
import pandas as pd
from scipy.special import softmax

OUT = Path(__file__).resolve().parent
BUDGET = OUT.parent/'frozen_candidate_coverage_20261012/budget'
sys.path.insert(0, str(BUDGET))
sys.path.insert(0, str(OUT.parent/'prior_protection_20261012'))
import run_coverage as c
from prior_rules import penalties, solve_weights

RULES = ('raw', 'guard', 'count', 'count_common', 'confidence')
ARMS = RULES+('initial_guard', 'prior_only')
OLD = BUDGET/'tasks/self24ports'


def folders():
    result = sorted(OLD.glob('*/PREDICTIONS_FROZEN.json'))
    assert len(result) == 23
    return [p.parent for p in result]


def freeze():
    start = time.perf_counter()
    replays, penalty_rows = [], []
    for directory in folders():
        task = directory.name
        feat = c.Features(c.TASKS/'self24ports'/task)
        index = pd.Index(feat.target_rows)
        phi, prior = feat.target(np.arange(len(index)))
        initial = c.archive(directory/'initial.npz')
        dest = OUT/'predictions'/task
        dest.mkdir(parents=True, exist_ok=True)
        for draw in range(5):
            for budget in c.BUDGETS:
                old = c.archive(directory/f'{draw}_{budget}.npz')
                s, q = index.get_indexer(old['support_ids']), index.get_indexer(old['query_ids'])
                assert (s >= 0).all() and (q >= 0).all()
                assert not np.intersect1d(s, q).size
                x, ps, sy = phi[s], prior[s], old['support_labels']
                weights, lambdas = {}, {}
                counts = np.bincount(sy, minlength=8)
                for rule in RULES:
                    lam = penalties(sy, ps, budget/8., rule)
                    weights[rule] = solve_weights(x, ps, sy, lam)
                    lambdas[rule] = lam
                    for cls in range(8):
                        penalty_rows.append(dict(task=task, draw=draw, budget=budget, rule=rule,
                            class_id=cls, support_n=int(counts[cls]), penalty=float(lam[cls])))
                delta = c.predict(initial, x[:, c.COLS], np.zeros(budget, int), 'image_history')
                changed = c.adapt_support(x, delta, c.DOSES[budget])
                weights['initial_guard'] = solve_weights(changed, ps, sy, lambdas['guard'])
                weights['initial'] = solve_weights(changed, ps, sy, lambdas['raw'])
                weights['prior_only'] = np.zeros_like(weights['raw'])
                scores = {a: prior[q]+phi[q]@w for a, w in weights.items()}
                errors = {}
                for arm in ('raw', 'initial'):
                    ref = old['expert_'+arm]
                    errors[arm+'_error'] = float(abs(scores[arm]-ref).max())
                    np.testing.assert_allclose(scores[arm], ref, atol=1e-8, rtol=1e-8)
                    np.testing.assert_array_equal(scores[arm].argmax(1), ref.argmax(1))
                    np.testing.assert_array_equal(c.late(old['base'], scores[arm]).argmax(1),
                                                  old['fusion_'+arm].argmax(1))
                replays.append(dict(task=task, draw=draw, budget=budget, n_query=len(q), **errors))
                np.savez_compressed(dest/f'{draw}_{budget}.npz',
                    **{a:scores[a] for a in ARMS},
                    query_ids=old['query_ids'], support_ids=old['support_ids'])
        print('FROZEN', task, '15 units', flush=True)
        del feat, phi, prior
        gc.collect()
    pd.DataFrame(replays).to_csv(OUT/'replays.csv', index=False)
    pd.DataFrame(penalty_rows).to_csv(OUT/'penalties.csv', index=False)
    c.write(OUT/'PREDICTIONS_FROZEN.json', dict(tasks=[p.name for p in folders()],
        prediction_units=len(replays), budgets=c.BUDGETS, draws=5, arms=ARMS,
        seconds=time.perf_counter()-start, runtime=sys.executable,
        new_neural_training=0, query_labels_used=False, new_selection=False))


def metric(y, probability):
    correct = probability.argmax(1)==y
    loss = np.square(probability-np.eye(8)[y]).sum(1)
    groups = np.unique(y)
    return dict(oa=100*correct.mean(), ba=100*np.mean([correct[y==k].mean() for k in groups]),
        brier=loss.mean(), macro_brier=np.mean([loss[y==k].mean() for k in groups]),
        nll=-np.log(np.maximum(probability[np.arange(len(y)), y], 1e-30)).mean())


def evaluate():
    assert c.read_json(OUT/'PREDICTIONS_FROZEN.json')['prediction_units']==345
    rows, classes, supports = [], [], []
    for directory in folders():
        task = directory.name
        table = pd.read_csv(c.read_json(directory/'PREDICTIONS_FROZEN.json')['target_manifest'])
        for draw in range(5):
            for budget in c.BUDGETS:
                old = c.archive(directory/f'{draw}_{budget}.npz')
                new = c.archive(OUT/'predictions'/task/f'{draw}_{budget}.npz')
                for key in ('query_ids', 'support_ids'):
                    np.testing.assert_array_equal(new[key], old[key])
                qm, sm = table.iloc[old['query_ids']], table.iloc[old['support_ids']]
                y = qm.class_id.to_numpy(int)
                np.testing.assert_array_equal(sm.class_id.to_numpy(int), old['support_labels'])
                counts = np.bincount(old['support_labels'], minlength=8)
                key = dict(task=task, budget=budget, draw=draw)
                observed = np.unique(y)
                support = dict(**key, n_query=len(y), query_classes=len(observed),
                    support_classes=int((counts>0).sum()), absent_query_classes=int((counts[observed]==0).sum()),
                    support_imbalance=float(np.std(counts)/(budget/8.)))
                if 'mmsi' in sm:
                    support['support_ships'] = sm.mmsi.nunique()
                    support['same_ship_query_fraction'] = float((qm.mmsi.notna() & qm.mmsi.isin(sm.mmsi.dropna())).mean())
                supports.append(support)
                scores = {a:new[a] for a in ARMS}
                scores.update({a:old['expert_'+a] for a in ('initial', 'simple')})
                for layer in ('expert', 'fusion'):
                    baseline = softmax(old['expert_raw'], axis=1) if layer=='expert' else old['fusion_raw']
                    bpred = baseline.argmax(1)
                    for arm, score in scores.items():
                        prob = softmax(score, axis=1) if layer=='expert' else c.late(old['base'], score)
                        rows.append(dict(**key, layer=layer, arm=arm, **metric(y, prob)))
                        pred = prob.argmax(1)
                        loss = np.square(prob-np.eye(8)[y]).sum(1)
                        for cls in observed:
                            mask = y==cls
                            rescue = 100*np.mean((bpred[mask]!=cls)&(pred[mask]==cls))
                            harm = 100*np.mean((bpred[mask]==cls)&(pred[mask]!=cls))
                            classes.append(dict(**key, layer=layer, arm=arm, class_id=int(cls),
                                class_name=str(qm.loc[mask, 'class_name'].iloc[0]) if 'class_name' in qm else str(cls),
                                n=int(mask.sum()), query_classes=len(observed), support_n=int(counts[cls]),
                                recall=100*np.mean(pred[mask]==cls), brier=loss[mask].mean(),
                                delta_recall=rescue-harm, rescue_ba=rescue/len(observed),
                                harm_ba=harm/len(observed), delta_ba_contribution=(rescue-harm)/len(observed)))
        print('EVALUATED', task, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'metrics.csv', index=False)
    ids = ['task', 'budget', 'draw', 'layer']
    values = ['oa', 'ba', 'macro_brier', 'brier', 'nll']
    pair = frame.merge(frame[frame.arm=='raw'][ids+values], on=ids, suffixes=('', '_raw'), validate='many_to_one')
    for v in values:
        pair['delta_'+v] = pair[v]-pair[v+'_raw']
    pair.to_csv(OUT/'paired.csv', index=False)
    cls = pd.DataFrame(classes)
    check = cls.groupby(ids+['arm']).delta_ba_contribution.sum()
    np.testing.assert_allclose(check.sort_index(), pair.set_index(ids+['arm']).delta_ba.sort_index(), atol=1e-10)
    cls.to_csv(OUT/'classes.csv', index=False)
    pd.DataFrame(supports).to_csv(OUT/'supports.csv', index=False)
    c.write(OUT/'EVALUATION_COMPLETE.json', dict(prediction_units=345, tasks=23, neural_training=0,
        new_rules_selected_on_target=False, all_predictions_frozen_before_metrics=True))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'evaluate', 'all'])
    args = parser.parse_args()
    with c.threadpool_limits(limits=2):
        if args.stage in ('freeze', 'all'):
            freeze()
        if args.stage in ('evaluate', 'all'):
            evaluate()
