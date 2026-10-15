"""Diagnostic only: fixed recovered-incidence source prior / support kernel 2x2."""
from incidence_recovered_experiment import (
    OUT, ROOT, TARGETS, READOUTS, FrozenInputs, load, write,
    condition_features, readouts, angle_readouts, late,
    np, pd, threadpool_limits,
)

DEST = OUT/'diagnostic_prior_kernel'
ARMS = ('matched_00', 'angle_prior_only_10', 'angle_kernel_only_01', 'angle_both_11')


def freeze():
    if (DEST/'FROZEN.json').exists():
        return
    main = load(OUT/'policies.npz')
    inputs = FrozenInputs()
    data = inputs.data
    for key in ('ids', 'support', 'support_labels', 'eval_masks'):
        np.testing.assert_array_equal(data[key], main[key])
    valid = main['recovered_valid']
    u = condition_features(main['recovered_incidence_deg'])
    prior = inputs.priors('angle_only', u)
    predictions = np.empty((5, 4, 4, len(valid)), np.int8)
    for draw, support in enumerate(data['support']):
        labels = data['support_labels'][draw]
        kb, ke = inputs.kernels(support)
        matched = readouts(kb, ke, *inputs.matched_prior, support, labels)
        prior_only = list(readouts(kb, ke, *prior, support, labels))
        for ri in (0, 1, 3):
            prior_only[ri][~valid] = matched[ri][~valid]
        prior_only[2] = late(prior_only[0], prior_only[1], ratio=.5)
        kernel_only = angle_readouts(kb, ke, *inputs.matched_prior, u, valid,
                                     support, labels, matched)
        both = angle_readouts(kb, ke, *prior, u, valid, support, labels, matched)
        for ai, result in enumerate((matched, prior_only, kernel_only, both)):
            for ri, values in enumerate(result):
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite diagnostic score')
                predictions[draw, ai, ri] = values.argmax(1)
            np.testing.assert_array_equal(predictions[draw, ai][:, ~valid],
                                          predictions[draw, 0][:, ~valid])
        np.testing.assert_array_equal(predictions[draw, 0], main['predictions'][draw, 0])
        np.testing.assert_array_equal(predictions[draw, 3], main['predictions'][draw, 2])
        print(f'DIAGNOSTIC draw{draw}: matched/recovered endpoints exact', flush=True)
    np.savez_compressed(DEST/'policies.npz', predictions=predictions,
        arms=np.array(ARMS), readouts=np.array(READOUTS), ids=data['ids'],
        support=data['support'], support_labels=data['support_labels'],
        eval_masks=data['eval_masks'], recovered_valid=valid)
    write(DEST/'FROZEN.json', dict(diagnostic_only=True, endpoints_exact=True,
        query_labels_used=False, source_refit=False, parameter_search=False,
        contrast='Matched versus recovered angle: source prior / angle kernel 2x2',
        missing_queries='Exact matched fallback in every diagnostic arm'))


def report():
    if not (DEST/'FROZEN.json').exists():
        raise RuntimeError('Freeze diagnostics before readout')
    p = load(DEST/'policies.npz')
    frame = pd.read_csv(ROOT/TARGETS['f0921']/'target_manifest.csv',
                        usecols=['sample_id', 'class_id'])
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str), p['ids'])
    truth = frame.class_id.to_numpy()
    np.testing.assert_array_equal(truth[p['support']], p['support_labels'])
    rows, classes = [], []
    for draw, ev in enumerate(p['eval_masks']):
        y = truth[ev]
        for ai, arm in enumerate(ARMS):
            for ri, readout in enumerate(READOUTS):
                pred = p['predictions'][draw, ai, ri, ev]
                recall = [100*np.mean(pred[y == c] == c) for c in range(8)]
                rows.append(dict(draw=draw, arm=arm, readout=readout,
                                 oa=100*np.mean(pred == y), ba=np.mean(recall)))
                for c, value in enumerate(recall):
                    classes.append(dict(draw=draw, arm=arm, readout=readout,
                                        class_id=c, recall=value))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(DEST/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(DEST/'class_metrics.csv', index=False)
    summary = metrics.groupby(['readout', 'arm'])[['oa', 'ba']].mean().reset_index()
    summary.to_csv(DEST/'summary.csv', index=False)
    write(DEST/'COMPLETE.json', dict(diagnostic_only=True, endpoint_replay_exact=True,
                                    source_refit=False, parameter_search=False))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        freeze()
        report()
