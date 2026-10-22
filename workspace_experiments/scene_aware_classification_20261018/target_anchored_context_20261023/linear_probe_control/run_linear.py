"""Source-validated multinomial linear probes on the identical frozen features."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import sys
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from run_anchored import EXP, STRONG, View, KINDS, combine, weights, risk, metrics

CS = [.01, .1, 1., 10.]


def main():
    start = time.perf_counter()
    targets = pd.read_csv(EXP/'readout/targets.csv', dtype={'mmsi': str})
    f, deep = np.load(EXP/'readout/features.npz'), np.load(STRONG/'deep_features.npz')
    assert np.array_equal(f['instance_ids'], deep['instance_ids'])
    raw = {'stem': f['target'], 'layer3': deep['layer3']}
    y, ids = targets.class_id.to_numpy(), targets.mmsi.to_numpy()
    records, predictions, checks = [], [], []
    for cohort in ('old', 'new'):
        for fold in range(6):
            root = PARENT if cohort == 'old' else PARENT/'partition_replication'
            d = np.load(root/f'fold{fold}_inputs.npz')
            train, val, test = [d[k] for k in ('train', 'validation', 'test')]
            kind = str(d['selected_kind']) if cohort == 'old' else 'layer3'
            names = KINDS[kind]
            views = {k: View(raw[k], train, k).transform(raw[k]) for k in names}
            z = combine(views, names)
            for a, b in ((train, val), (train, test), (val, test)):
                assert not set(ids[a]) & set(ids[b])
                assert not set(targets.iloc[a].pass_group) & set(targets.iloc[b].pass_group)
            best = None
            for c in CS:
                head = LogisticRegression(C=c, solver='lbfgs', max_iter=2000, tol=1e-6, random_state=17)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', ConvergenceWarning)
                    head.fit(z[train], y[train], sample_weight=weights(y[train], ids[train]))
                converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                score = head.decision_function(z)
                val_risk = risk(y[val], score[val])
                records.append(dict(cohort=cohort, fold=fold, kind=kind, C=c, val_nll=val_risk,
                                    iterations=int(head.n_iter_.max()), converged=converged, **metrics(y[test], score[test])))
                if best is None or val_risk < best[0]:
                    best = val_risk, c, head, score, converged
            val_risk, c, head, score, converged = best
            assert converged, (cohort, fold, c)
            probability = head.predict_proba(z[test])
            error = float(np.abs(softmax(score[test], axis=1)-probability).max())
            assert error < 1e-10
            np.savez_compressed(HERE/f'{cohort}_fold{fold}_head.npz', coef=head.coef_, intercept=head.intercept_, C=c)
            checks.append(dict(cohort=cohort, fold=fold, kind=kind, selected_C=c, source_val_nll=val_risk,
                               rows_train=len(train), rows_val=len(val), rows_test=len(test), probability_replay_error=error))
            for i in test:
                predictions.append(dict(cohort=cohort, fold=fold, instance_id=targets.iloc[i].instance_id, class_id=int(y[i]),
                    prediction=int(score[i].argmax()), base_prediction=int(d['base'][i].argmax()),
                    **{f's{j}':float(score[i,j]) for j in range(3)}, **{f'b{j}':float(d['base'][i,j]) for j in range(3)}))
            print(cohort, fold, c, 'net', int((score[test].argmax(1)==y[test]).sum()-(d['base'][test].argmax(1)==y[test]).sum()), flush=True)
            pd.DataFrame(records).to_csv(HERE/'candidates.csv', index=False)
            pd.DataFrame(predictions).to_csv(HERE/'predictions.csv', index=False)
            pd.DataFrame(checks).to_csv(HERE/'verification.csv', index=False)
    rows = pd.DataFrame(predictions)
    pooled = []
    for cohort, part in rows.groupby('cohort'):
        pooled.append(dict(cohort=cohort, **metrics(part.class_id.to_numpy(), part[['s0','s1','s2']].to_numpy()),
                           net=int(part.prediction.eq(part.class_id).sum()-part.base_prediction.eq(part.class_id).sum())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv', index=False)
    rows['net'] = rows.prediction.eq(rows.class_id).astype(int)-rows.base_prediction.eq(rows.class_id).astype(int)
    rows.groupby(['cohort','class_id']).agg(n=('net','size'),net=('net','sum')).reset_index().to_csv(HERE/'class_effects.csv', index=False)
    (HERE/'summary.json').write_text(json.dumps(dict(fits=len(records), selected_heads=len(checks), seconds=time.perf_counter()-start,
        deterministic_fits=True, source_only_selection=True, no_attention_training=True), indent=2), encoding='utf-8')
    print(pd.DataFrame(pooled).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
