"""Summarize already-frozen predictions; never select a checkpoint or policy."""

from experiment import *


def metric(truth, score):
    pred = score.argmax(1)
    recalls = [float(np.mean(pred[truth == c] == c)) if np.any(truth == c) else np.nan
               for c in range(8)]
    return dict(oa=100*float(np.mean(pred == truth)), ba=100*float(np.nanmean(recalls))), recalls


def main():
    if not (OUT/'PREDICTIONS_FROZEN.json').exists():
        raise ValueError('Finish predeclared arms before querying classification results')
    frame = pd.read_csv(OUT/'manifest.csv', dtype={'mmsi': str})
    metrics, classes, contrasts, events, inventories = [], [], [], [], []
    for task in TASKS:
        for seed in SEEDS:
            folder = OUT/task/str(seed)
            split = dict(np.load(folder/'split.npz'))
            audit = audit_split(frame, split)
            inventories.append(dict(task=task, seed=seed, source_n=len(split['source']),
                extra_n=len(split['target_add']), support_n=64, query_n=len(split['query']),
                query_ships=audit['query']['ships'], query_products=audit['query']['products'],
                query_present_classes=int(np.count_nonzero(audit['query']['class_counts']))))
            y = frame.iloc[split['query']].class_id.to_numpy()
            vessels = frame.iloc[split['query']].mmsi.to_numpy()
            loaded = {}
            for stage in ('frozen', 'updated'):
                for arm in ARMS:
                    with np.load(folder/stage/arm/'predictions.npz') as z:
                        np.testing.assert_array_equal(z['rows'], split['query'])
                        pred = {k:z[k] for k in z.files if k != 'rows'}
                    loaded[(stage, arm)] = pred
                    for mode, scores in pred.items():
                        if not np.isfinite(scores).all():
                            raise ValueError('Nonfinite final predictions')
                        mm, recall = metric(y, scores)
                        ship = pd.DataFrame(dict(mmsi=vessels, label=y, correct=scores.argmax(1) == y))
                        ship = ship.groupby(['mmsi', 'label'], as_index=False).correct.mean()
                        metrics.append(dict(task=task, seed=seed, stage=stage, arm=arm, mode=mode,
                            **mm, ship_oa=100*float(ship.correct.mean()),
                            ship_ba=100*float(ship.groupby('label').correct.mean().mean()), n=len(y)))
                        for c in range(8):
                            classes.append(dict(task=task, seed=seed, stage=stage, arm=arm, mode=mode,
                                class_id=c, n=int(np.sum(y == c)), recall=100*recall[c]))
            for arm in ARMS:
                for mode in ('r50_prior', 'r50_adapted'):
                    np.testing.assert_array_equal(loaded[('frozen', arm)][mode], loaded[('updated', arm)][mode])
            pairs = []
            for stage in ('frozen', 'updated'):
                pairs += [(stage+'_quantity', (stage, 'S'), (stage, 'S_U')),
                          (stage+'_target_experience', (stage, 'S_U'), (stage, 'S_T'))]
            pairs += [('update_'+arm, ('frozen', arm), ('updated', arm)) for arm in ARMS]
            for name, a, b in pairs:
                for mode in loaded[a]:
                    aa, bb = loaded[a][mode], loaded[b][mode]
                    ma, _ = metric(y, aa); mb, _ = metric(y, bb)
                    pa, pb = aa.argmax(1), bb.argmax(1)
                    rescue, harm = (pa != y) & (pb == y), (pa == y) & (pb != y)
                    contrasts.append(dict(task=task, seed=seed, comparison=name, mode=mode,
                        delta_oa=mb['oa']-ma['oa'], delta_ba=mb['ba']-ma['ba'],
                        rescues=int(rescue.sum()), harms=int(harm.sum()),
                        rescue_ships=len(np.unique(vessels[rescue])), harm_ships=len(np.unique(vessels[harm]))))
                    if mode in ('fusion', 'dual_prior', 'dual_adapted'):
                        for i in np.flatnonzero(pa != pb):
                            meta = frame.iloc[split['query'][i]]
                            events.append(dict(task=task, seed=seed, comparison=name, mode=mode,
                                row=int(meta.row), mmsi=meta.mmsi, product=meta['product'],
                                class_id=int(y[i]), before=int(pa[i]), after=int(pb[i]),
                                event='rescue' if rescue[i] else ('harm' if harm[i] else 'wrong_to_wrong')))
    m, c = pd.DataFrame(metrics), pd.DataFrame(contrasts)
    m.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_metrics.csv', index=False)
    c.to_csv(OUT/'contrasts.csv', index=False)
    pd.DataFrame(events).to_csv(OUT/'prediction_events.csv', index=False)
    pd.DataFrame(inventories).to_csv(OUT/'cohort_counts.csv', index=False)
    mean = m.groupby(['task', 'stage', 'arm', 'mode'], sort=False)[['oa', 'ba', 'ship_oa', 'ship_ba']].mean().reset_index()
    mean.to_csv(OUT/'mean_metrics.csv', index=False)
    cm = c.groupby(['task', 'comparison', 'mode'], sort=False)[['delta_oa', 'delta_ba', 'rescues', 'harms']].mean().reset_index()
    cm.to_csv(OUT/'mean_contrasts.csv', index=False)
    fusion = mean.loc[mean['mode'].eq('fusion')]
    print('FUSION OA/BA', flush=True)
    print(fusion.to_string(index=False), flush=True)
    print('FUSION CONTRASTS', flush=True)
    print(cm.loc[cm['mode'].eq('fusion')].to_string(index=False), flush=True)
    write(OUT/'COMPLETE.json', dict(tasks=TASKS, splits=len(TASKS)*len(SEEDS),
        neural_fits=len(TASKS)*len(SEEDS)*len(ARMS), metrics=len(m),
        all_split_audits_passed=True, r50_predictions_unchanged_by_r18_training=True,
        main_thread_files_changed=False))


if __name__ == '__main__':
    main()
