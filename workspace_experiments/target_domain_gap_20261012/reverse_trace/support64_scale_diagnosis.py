"""Distinguish source-conditioned reader changes from scalar residual amplification."""

from support64_probe import *


def main():
    frame = pd.read_csv(PARENT/'manifest.csv', dtype={'mmsi': str})
    results, scales = [], []
    for task in TASKS:
        for seed in SEEDS:
            folder = PARENT/task/str(seed)
            split = dict(np.load(folder/'split.npz'))
            z = dict(np.load(folder/'features.npz'))
            rows = z.pop('rows'); loc = pd.Index(rows)
            si, qi, source = (loc.get_indexer(split[k]) for k in ('support', 'query', 'source'))
            train = np.r_[split['source'], split['support']]
            fit_pos = loc.get_indexer(train)
            y_fit = frame.iloc[train].class_id.to_numpy()
            y_support = frame.iloc[split['support']].class_id.to_numpy()
            head = dict(np.load(folder/'frozen/S/heads.npz'))
            score = {name:{} for name in ('reference', 'joint', 'scalar_only', 'non_scalar_only')}
            for mode, x in [('r50', z['r50']), ('dual', np.c_[z['r50'], z['r18']]/np.sqrt(2.))]:
                h = fit_head(x[fit_pos], y_fit, len(source))
                p0 = x@head[mode+'_coef']+head[mode+'_bias']
                p1 = x@h['coef']+h['bias']
                original = residual(x, p0, si, y_support)
                joint = residual(x, p1, si, y_support)
                base_correction = original-p0
                delta = joint-original
                r = base_correction[source]-base_correction[source].mean(1, keepdims=True)
                d = delta[source]-delta[source].mean(1, keepdims=True)
                alpha = float(np.sum(r*d)/np.sum(r*r))
                centered = base_correction-base_correction.mean(1, keepdims=True)
                scalar = alpha*centered
                score['reference'][mode] = original[qi]
                score['joint'][mode] = joint[qi]
                score['scalar_only'][mode] = (original+scalar)[qi]
                score['non_scalar_only'][mode] = (joint-scalar)[qi]
                scales.append(dict(task=task, seed=seed, mode=mode, alpha=alpha,
                    source_non_scalar_energy_ratio=float(np.sum((d-alpha*r)**2)/np.sum(d*d))))
            fused = {name:.5*softmax(p['r50'])+.5*softmax(p['dual']) for name,p in score.items()}
            with np.load(OUT/'support64'/task/str(seed)/'predictions.npz') as old:
                np.testing.assert_array_equal(fused['reference'], old['original_residual64'])
                np.testing.assert_array_equal(fused['joint'], old['global64_residual64'])
            y = frame.iloc[split['query']].class_id.to_numpy()
            for name,p in fused.items():
                mm, _ = metric(y, p)
                results.append(dict(task=task, seed=seed, intervention=name, **mm))
    r = pd.DataFrame(results)
    r.to_csv(OUT/'support64/scale_diagnosis.csv', index=False)
    pd.DataFrame(scales).to_csv(OUT/'support64/scalar_parameters.csv', index=False)
    print(r.groupby(['task','intervention'])[['oa','ba']].mean().to_string())
    write(OUT/'support64/SCALE_DIAGNOSIS.json', dict(new_neural_fits=0,
        existing_head_fit_replays=16, scale_selection='Least squares on common source2048; no query labels',
        scalar_parameters='One per head, unconstrained, no search',
        interpretation='Posthoc computational control, not target-selected deployment policy'))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
