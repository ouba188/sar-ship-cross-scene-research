"""Fixed three-way label-use probe: no extra-target teacher is used for fitting."""

from reverse_analysis import PARENT, OUT, TASKS, SEEDS, softmax, residual, write, metric
from experiment import fit_head
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def main():
    frame = pd.read_csv(PARENT/'manifest.csv', dtype={'mmsi': str})
    destination = OUT/'support64'
    destination.mkdir(exist_ok=True)
    write(destination/'PROTOCOL.json', dict(arms=['original_residual64', 'global64_only', 'global64_residual64'],
        baseline='S prior, not S+U or S+T', source_rows=2048, distinct_target_labels=64,
        shared_source_geometry=True, supervised_query_access=False,
        new_prior='Class-balanced ridge1 fit on S + the SAME64, total weight mass2048',
        residual='Original residual ridge8 with same64; labels reused, not new labels',
        selection='All three fixed arms reported; no hyperparameter search',
        teacher_weight_access=False, neural_fits=0, new_source_head_fits=16,
        confirmation='Reused development queries, not independent confirmation'))
    metrics, events = [], []
    frozen = []
    for task in TASKS:
        for seed in SEEDS:
            folder = PARENT/task/str(seed)
            split = dict(np.load(folder/'split.npz'))
            z = dict(np.load(folder/'features.npz'))
            rows = z.pop('rows')
            loc = pd.Index(rows)
            si, qi = loc.get_indexer(split['support']), loc.get_indexer(split['query'])
            y_support = frame.iloc[split['support']].class_id.to_numpy()
            train = np.r_[split['source'], split['support']]
            newfit_positions = loc.get_indexer(train)
            fit_labels = frame.iloc[train].class_id.to_numpy()
            with np.load(folder/'frozen/S/heads.npz') as old:
                heads = dict(old)
            new, adapted, original = {}, {}, {}
            for mode, x in [('r50', z['r50']), ('dual', np.c_[z['r50'], z['r18']]/np.sqrt(2.))]:
                h = fit_head(x[newfit_positions], fit_labels, len(split['source']))
                new[mode] = x@h['coef']+h['bias']
                adapted[mode] = residual(x, new[mode], si, y_support)
                original[mode] = residual(x, x@heads[mode+'_coef']+heads[mode+'_bias'], si, y_support)
            scores = {name:.5*softmax(p['r50'][qi])+.5*softmax(p['dual'][qi])
                      for name,p in [('original_residual64', original), ('global64_only', new),
                                     ('global64_residual64', adapted)]}
            with np.load(folder/'frozen/S/predictions.npz') as old:
                np.testing.assert_array_equal(scores['original_residual64'], old['fusion'])
            dest = destination/task/str(seed)
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest/'predictions.npz', rows=split['query'], **scores)
            frozen.append((task, seed, split['query'], scores))
    write(destination/'PREDICTIONS_FROZEN.json', dict(fits=16, target_labels_per_fit=64))
    for task, seed, rows, scores in frozen:
        truth = frame.iloc[rows].class_id.to_numpy()
        baseline = scores['original_residual64'].argmax(1)
        for arm, score in scores.items():
            mm, _ = metric(truth, score)
            pred = score.argmax(1)
            rescue, harm = (baseline != truth) & (pred == truth), (baseline == truth) & (pred != truth)
            metrics.append(dict(task=task, seed=seed, arm=arm, **mm,
                rescues=int(rescue.sum()), harms=int(harm.sum())))
            for i in np.flatnonzero(pred != baseline):
                meta = frame.iloc[rows[i]]
                events.append(dict(task=task, seed=seed, arm=arm, row=int(rows[i]), mmsi=meta.mmsi,
                    class_id=int(truth[i]), before=int(baseline[i]), after=int(pred[i]),
                    event='rescue' if rescue[i] else ('harm' if harm[i] else 'wrong_to_wrong')))
    result = pd.DataFrame(metrics)
    result.to_csv(destination/'metrics.csv', index=False)
    result.groupby(['task','arm'], sort=False)[['oa','ba','rescues','harms']].mean().reset_index().to_csv(destination/'mean_metrics.csv', index=False)
    pd.DataFrame(events).to_csv(destination/'events.csv', index=False)
    print(result.groupby(['task','arm'])[['oa','ba','rescues','harms']].mean().to_string())
    write(destination/'COMPLETE.json', dict(head_fits=16, neural_fits=0, teacher_used=False))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
