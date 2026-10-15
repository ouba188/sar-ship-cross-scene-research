"""No new fits: localize target-reader benefit and matched tail-update failures."""

from experiment import *
from report import metric


def main():
    frame = pd.read_csv(OUT/'manifest.csv', dtype={'mmsi': str})
    y_all = frame.class_id.to_numpy()
    interventions, generalization, classes, components = [], [], [], []
    for task in TASKS:
        for seed in SEEDS:
            folder = OUT/task/str(seed)
            split = dict(np.load(folder/'split.npz'))
            z = dict(np.load(folder/'features.npz'))
            rows = z.pop('rows')
            loc = pd.Index(rows)
            qi, si = loc.get_indexer(split['query']), loc.get_indexer(split['support'])
            y = y_all[split['query']]
            features = dict(r50=z['r50'], r18=z['r18'], dual=np.c_[z['r50'], z['r18']]/np.sqrt(2.))
            scores = {}
            for mode in ('r50', 'dual'):
                x = features[mode]
                h_u = dict(np.load(folder/'frozen/S_U/heads.npz'))
                h_t = dict(np.load(folder/'frozen/S_T/heads.npz'))
                pu = x@h_u[mode+'_coef']+h_u[mode+'_bias']
                pt = x@h_t[mode+'_coef']+h_t[mode+'_bias']
                full_u = residual(x, pu, si, y_all[split['support']])
                full_t = residual(x, pt, si, y_all[split['support']])
                dq = pt[qi]-pu[qi]
                ds = (full_t[qi]-full_u[qi])-dq
                scores[mode] = dict(reference=full_u[qi], query_prior_only=full_u[qi]+dq,
                                    support_residual_only=full_u[qi]+ds, both=full_t[qi])
                rival_scores = full_u[qi].copy(); rival_scores[np.arange(len(y)), y] = -np.inf
                rival = rival_scores.argmax(1)
                prior_margin = dq[np.arange(len(y)), y]-dq[np.arange(len(y)), rival]
                residual_margin = ds[np.arange(len(y)), y]-ds[np.arange(len(y)), rival]
                pa, pb = full_u[qi].argmax(1), full_t[qi].argmax(1)
                for name, mask in [('all', np.ones(len(y), bool)),
                                   ('rescued', (pa != y) & (pb == y)),
                                   ('harmed', (pa == y) & (pb != y))]:
                    components.append(dict(task=task, seed=seed, mode=mode, cohort=name, n=int(mask.sum()),
                        prior_margin=float(prior_margin[mask].mean()) if mask.any() else np.nan,
                        support_margin=float(residual_margin[mask].mean()) if mask.any() else np.nan))
            for intervention in scores['r50']:
                fusion = .5*softmax(scores['r50'][intervention])+.5*softmax(scores['dual'][intervention])
                mm, _ = metric(y, fusion)
                interventions.append(dict(task=task, seed=seed, intervention=intervention, **mm))
                if intervention in ('reference', 'both'):
                    arm = 'S_U' if intervention == 'reference' else 'S_T'
                    with np.load(folder/'frozen'/arm/'predictions.npz') as original:
                        np.testing.assert_allclose(fusion, original['fusion'], atol=1e-6)
                        np.testing.assert_array_equal(fusion.argmax(1), original['fusion'].argmax(1))
            train = np.r_[split['source'], split['target_add']]
            with np.load(folder/'geometry.npz') as g:
                geometry = {k:g['r18_'+k] for k in ('center', 'scale', 'matrix', 'bias')}
            extra = project(np.load(folder/'updated/S_T/extra.npy'), geometry)
            updated = dict(r50=z['r50'], r18=extra, dual=np.c_[z['r50'], extra]/np.sqrt(2.))
            for stage, ff in [('frozen', features), ('updated', updated)]:
                h = dict(np.load(folder/stage/'S_T/heads.npz'))
                for mode in ('r18', 'dual'):
                    p = ff[mode]@h[mode+'_coef']+h[mode+'_bias']
                    for cohort, ids in [('fit_source', split['source']), ('fit_target', split['target_add']),
                                        ('query', split['query'])]:
                        ix = loc.get_indexer(ids)
                        mm, recall = metric(y_all[ids], p[ix])
                        generalization.append(dict(task=task, seed=seed, stage=stage, mode=mode,
                            cohort=cohort, **mm, ce=float(-np.log(np.maximum(softmax(p[ix])[np.arange(len(ids)), y_all[ids]], 1e-12)).mean())))
                        if cohort == 'query':
                            for c, value in enumerate(recall):
                                classes.append(dict(task=task, seed=seed, stage=stage, mode=mode,
                                                    class_id=c, recall=100*value))
    pd.DataFrame(interventions).to_csv(OUT/'reader_path_interventions.csv', index=False)
    pd.DataFrame(generalization).to_csv(OUT/'update_fit_query_diagnosis.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'update_query_class_diagnosis.csv', index=False)
    pd.DataFrame(components).to_csv(OUT/'reader_margin_components.csv', index=False)
    print('READER PATH INTERVENTIONS')
    print(pd.DataFrame(interventions).groupby(['task', 'intervention'])[['oa', 'ba']].mean().to_string())
    g = pd.DataFrame(generalization)
    print('FIT/QUERY DIAGNOSTICS')
    print(g.loc[g['mode'].eq('dual')].groupby(['task', 'stage', 'cohort'])[['oa', 'ba', 'ce']].mean().to_string())
    write(OUT/'DIAGNOSIS_COMPLETE.json', dict(new_neural_fits=0, new_readout_fits=0,
        reference_and_both_interventions_replay_exact_labels=True,
        computational_interventions_not_physical_causal_identification=True))


if __name__ == '__main__':
    main()
