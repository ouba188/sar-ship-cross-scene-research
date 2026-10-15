"""Describe cached 64-support sensitivity on original and fixed common queries."""

from full_teacher_data import BASE, SOURCE, PORTS

import json
from itertools import combinations

import numpy as np
import pandas as pd

from evaluate_probability_layers import cached, probability_rows


OUT = BASE/'layered_evaluation_20261012/support_sensitivity'
MODEL_KEYS = ['family', 'context', 'training_seed', 'arm', 'baseline_arm', 'scope']


def common_queries(episodes):
    return np.array(sorted(set.intersection(*(set(e['ids']) for e in episodes))), dtype=int)


def accuracy(scores, truth):
    pred = np.asarray(scores).argmax(1)
    correct = pred == truth
    classes = np.unique(truth)
    return 100*correct.mean(), 100*np.mean([correct[truth == c].mean() for c in classes])


def identity_overlap(query, support):
    known = query.mmsi.notna()
    hits = known & query.mmsi.isin(support.mmsi.dropna())
    return int(known.sum()), int(hits.sum())


def source_groups(meta):
    splits = [cached(BASE/'incidence_transport_20261011'/f'31415_source_{d}_predictions.npz')
              for d in range(3)]
    for context in PORTS:
        for seed in (31415, 27182):
            episodes = []
            for draw, split in enumerate(splits):
                scores = {}
                for arm in ('task', 'score', 'relation', 'joint', 'readout'):
                    folder = 'case_relation_readout_repair_20261012' if arm == 'readout' else 'case_relation_student_20261012'
                    z = cached(BASE/folder/context/str(seed)/arm/f'31415_{draw}.npz')
                    np.testing.assert_array_equal(z['query_obs_ids'], split['query_obs_ids'])
                    scores[arm] = z['scores']
                episodes.append(dict(ids=split['query_obs_ids'], support=split['support_obs_ids'],
                                     draw=draw, support_key=f'source_{draw}', scores=scores))
            yield 'source_students', context, seed, 'task', meta, episodes
        episodes = []
        for draw, split in enumerate(splits):
            scores = {}
            for arm in ('raw', 'class_center', 'image_history', 'class_image_history', 'image_class_target'):
                folder = 'case_completion_class_control_20261012' if arm == 'image_class_target' else 'case_support_completion_20261012'
                z = cached(BASE/folder/context/f'31415_{draw}_{arm}.npz')
                np.testing.assert_array_equal(z['query_obs_ids'], split['query_obs_ids'])
                np.testing.assert_array_equal(z['support_obs_ids'], split['support_obs_ids'])
                scores[arm] = z['scores']
            episodes.append(dict(ids=split['query_obs_ids'], support=split['support_obs_ids'],
                                 draw=draw, support_key=f'source_{draw}', scores=scores))
        yield 'source_completion', context, 0, 'raw', meta, episodes


def target_groups(meta):
    for port in PORTS:
        z = cached(BASE/'support_completion_target_20261012'/port/'predictions.npz')
        episodes = []
        for draw in range(5):
            mask = z['eval_masks'][draw]
            assert not mask[z['support'][draw]].any()
            episodes.append(dict(ids=z['rows'][mask], support=z['rows'][z['support'][draw]],
                                 draw=draw, support_key=f'target_{port}_{draw}',
                                 scores={arm: z[arm][draw][mask] for arm in
                                         ('base', 'expert_raw', 'expert_corrected', 'fusion_raw', 'fusion_corrected')}))
        yield 'actual_target', port, 0, None, meta, episodes


def summarize(draws):
    records = []
    for key, group in draws.groupby(MODEL_KEYS, sort=True):
        row = dict(zip(MODEL_KEYS, key))
        row.update(n_support_draws=len(group), query_n_min=int(group.n_query.min()),
                   query_n_max=int(group.n_query.max()))
        for metric in ('oa', 'ba', 'baseline_oa', 'baseline_ba', 'delta_oa', 'delta_ba',
                       'nll', 'brier', 'delta_nll', 'delta_brier'):
            v = group[metric].to_numpy(float)
            row.update({metric+'_mean': v.mean(), metric+'_std_ddof0': v.std(ddof=0),
                        metric+'_min': v.min(), metric+'_max': v.max(),
                        metric+'_q10_descriptive': np.quantile(v, .1)})
        for metric in ('oa', 'ba'):
            delta = group['delta_'+metric].to_numpy()
            row[metric+'_paired_wins'] = int((delta > 1e-10).sum())
            row[metric+'_paired_losses'] = int((delta < -1e-10).sum())
            row[metric+'_paired_ties'] = int((abs(delta) <= 1e-10).sum())
            row[metric+'_paired_win_fraction'] = float((delta > 1e-10).mean())
            row[metric+'_std_change'] = row[metric+'_std_ddof0']-row['baseline_'+metric+'_std_ddof0']
        records.append(row)
    return pd.DataFrame(records)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    columns = ['class_id', 'class_name', 'port', 'mmsi', 'product']
    source_meta = pd.read_csv(BASE/'observations.csv', usecols=columns)
    target_meta = pd.read_csv(SOURCE/'manifest.csv', usecols=columns)
    composition, support_classes, overlap, coverage = [], [], [], []
    results, class_results, flips = [], [], []
    seen, pools = {}, {}
    groups = list(source_groups(source_meta)) + list(target_groups(target_meta))
    for family, context, seed, default_baseline, meta, episodes in groups:
        common = common_queries(episodes)
        assert len(common) > 0
        qmeta = meta.iloc[common]
        common_class_n = qmeta.class_id.value_counts().to_dict()
        info = dict(family=family, context=context, training_seed=seed)
        coverage.append(dict(**info, n_draws=len(episodes), n_common_query=len(common),
                             n_union_query=len(set.union(*(set(e['ids']) for e in episodes))),
                             n_common_classes=qmeta.class_id.nunique(),
                             n_common_known_ships=qmeta.mmsi.nunique(),
                             common_class_counts=json.dumps(common_class_n, sort_keys=True)))
        predictions = {arm: [] for arm in episodes[0]['scores']}
        for e in episodes:
            support = meta.iloc[e['support']]
            assert len(support) == 64 and len(np.unique(e['support'])) == 64
            assert not np.isin(common, e['support']).any()
            counts = support.class_id.value_counts().reindex(range(8), fill_value=0)
            pool = 'source' if family.startswith('source') else 'target_'+context
            if e['support_key'] in seen:
                np.testing.assert_array_equal(np.sort(e['support']), seen[e['support_key']])
            else:
                seen[e['support_key']] = np.sort(e['support'])
                pools.setdefault(pool, []).append((e['support_key'], e['support'], support))
                composition.append(dict(support_key=e['support_key'], pool=pool, draw=e['draw'],
                    n_support=64, n_known_ships=support.mmsi.nunique(),
                    n_unknown_identity=int(support.mmsi.isna().sum()),
                    n_products=support['product'].nunique(), n_actual_ports=support.port.nunique(),
                    n_classes=int((counts > 0).sum()), min_count_all8=int(counts.min()),
                    max_count=int(counts.max()), missing_classes=','.join(map(str, counts.index[counts == 0])),
                    class_counts=json.dumps(counts.to_dict(), sort_keys=True)))
                for c, n in counts.items():
                    support_classes.append(dict(support_key=e['support_key'], pool=pool, draw=e['draw'],
                        class_id=c, class_name=meta.loc[meta.class_id == c, 'class_name'].iloc[0], n_support=n,
                        n_support_known_ships=support.loc[support.class_id == c, 'mmsi'].nunique()))
            indices = pd.Index(e['ids']).get_indexer(common)
            assert (indices >= 0).all()
            for arm, scores in e['scores'].items():
                baseline_arm = default_baseline or ('base' if arm == 'base' else arm.split('_')[0]+'_raw')
                old = e['scores'][baseline_arm]
                predictions[arm].append(scores[indices].argmax(1))
                for scope, pick in [('original_query', np.arange(len(e['ids']))), ('common_query', indices)]:
                    mm = meta.iloc[e['ids'][pick]]
                    truth = mm.class_id.to_numpy()
                    kind = 'probabilities' if arm.startswith('fusion') else 'logits'
                    new = probability_rows(scores[pick], truth, kind)
                    before = probability_rows(old[pick], truth, kind)
                    oa, ba = accuracy(scores[pick], truth)
                    boa, bba = accuracy(old[pick], truth)
                    known, shared = identity_overlap(mm, support)
                    shared_class = counts.reindex(truth).to_numpy() > 0
                    detail = dict(**info, draw=e['draw'], support_key=e['support_key'],
                                  arm=arm, baseline_arm=baseline_arm, scope=scope)
                    results.append(dict(**detail, n_query=len(mm), n_query_classes=mm.class_id.nunique(),
                        n_query_known_identity=known, n_query_support_same_ship=shared,
                        query_support_same_ship_fraction_known=shared/known if known else np.nan,
                        query_class_supported_fraction=shared_class.mean(),
                        oa=oa, ba=ba, baseline_oa=boa, baseline_ba=bba, delta_oa=oa-boa, delta_ba=ba-bba,
                        nll=new['nll'].mean(), brier=new['brier'].mean(),
                        delta_nll=(before['nll']-new['nll']).mean(),
                        delta_brier=(before['brier']-new['brier']).mean()))
                    for c in np.unique(truth):
                        use = truth == c
                        class_results.append(dict(**detail, class_id=int(c), n_query=int(use.sum()),
                            n_support=int(counts[c]), recall=100*new['correct'][use].mean(),
                            baseline_recall=100*before['correct'][use].mean(),
                            delta_recall=100*(new['correct'][use].astype(float)-before['correct'][use]).mean()))
        for arm, values in predictions.items():
            pred = np.stack(values)
            changed = np.any(pred != pred[0], axis=0)
            pairs = [np.mean(a != b) for a, b in combinations(pred, 2)]
            flips.append(dict(**info, arm=arm, n_common_query=len(common),
                              any_prediction_change_fraction=changed.mean(),
                              mean_pairwise_disagreement=np.mean(pairs)))
        print('COMPLETE', family, context, seed, 'common_queries', len(common), flush=True)
    for pool, draws in pools.items():
        for (ka, a, ma), (kb, b, mb) in combinations(draws, 2):
            aa, bb = set(a), set(b)
            sa, sb = set(ma.mmsi.dropna()), set(mb.mmsi.dropna())
            overlap.append(dict(pool=pool, support_a=ka, support_b=kb,
                                row_intersection=len(aa & bb), row_jaccard=len(aa & bb)/len(aa | bb),
                                known_ship_intersection=len(sa & sb),
                                known_ship_jaccard=len(sa & sb)/len(sa | sb) if sa | sb else np.nan))
    frame = pd.DataFrame(results)
    summary = summarize(frame)
    # Detect changed cache interpretation, not whether a candidate is positive.
    prior = pd.read_csv(BASE/'layered_evaluation_20261012/probability/per_draw.csv')
    target_prior = pd.read_csv(BASE/'layered_evaluation_20261012/probability/per_port_draw.csv')
    n_replays = 0
    for row in frame.loc[frame.scope == 'original_query'].itertuples():
        old = target_prior if row.family == 'actual_target' else prior
        match = old[(old.family == row.family) & (old.training_seed == row.training_seed)
                    & (old.draw == row.draw) & (old.arm == row.arm)]
        match = match[match.actual_port == row.context] if row.family == 'actual_target' else match[match.context == row.context]
        assert len(match) == 1
        np.testing.assert_allclose([row.oa, row.ba], match[['micro_oa', 'ba']].iloc[0], atol=1e-10, rtol=0)
        n_replays += 1
    outputs = dict(support_composition=pd.DataFrame(composition), support_class_counts=pd.DataFrame(support_classes),
                   support_overlap=pd.DataFrame(overlap), query_coverage=pd.DataFrame(coverage),
                   per_draw=frame, summary=summary, per_class_draw=pd.DataFrame(class_results),
                   prediction_instability=pd.DataFrame(flips))
    for name, data in outputs.items():
        data.to_csv(OUT/(name+'.csv'), index=False)
    receipt = dict(status='complete', unique_support_sets=len(seen), original_metric_replays=n_replays,
                   refits=0, training=0, tests=self_test(),
                   common_query='Intersection of existing query IDs across all draws within each fixed model context.',
                   support_seeds='Source three distinct draws reused across contexts and training seeds; target five per actual port.',
                   interpretation='All ranges/std/q10/win fractions are descriptive. No independent-replication or causal-composition claim.',
                   baseline='Same support IDs and query mask for both arms; base may itself be support-adapted.',
                   missing_classes='BA averages classes present in that query scope; original/common coverage is explicitly saved.')
    (OUT/'COMPLETE.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(summary.loc[(summary.scope == 'common_query') & summary.arm.isin(['fusion_corrected', 'image_class_target', 'readout']),
                      ['family', 'context', 'training_seed', 'arm', 'ba_mean', 'ba_std_ddof0', 'delta_ba_mean',
                       'delta_ba_min', 'delta_ba_max', 'ba_paired_wins', 'ba_paired_losses']].to_string(index=False))


def self_test():
    # Shuffled query IDs must refer to the same observations, not matching positions.
    eps = [dict(ids=np.array([3, 1, 2])), dict(ids=np.array([4, 3, 1]))]
    np.testing.assert_array_equal(common_queries(eps), [1, 3])
    truth = np.array([0, 0, 0, 1])
    scores = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
    np.testing.assert_allclose(accuracy(scores, truth), [75., 50.])
    q = pd.DataFrame(dict(mmsi=[1., 2., np.nan]))
    s = pd.DataFrame(dict(mmsi=[1., 1., np.nan]))
    assert identity_overlap(q, s) == (2, 1)
    fixture = []
    for new, old in [(50., 40.), (60., 60.), (40., 50.)]:
        row = dict(family='test', context='A', training_seed=0, arm='new',
                   baseline_arm='old', scope='common_query', n_query=10)
        for metric in ('oa', 'ba'):
            row[metric], row['baseline_'+metric], row['delta_'+metric] = new, old, new-old
        row.update(nll=0., brier=0., delta_nll=0., delta_brier=0.)
        fixture.append(row)
    result = summarize(pd.DataFrame(fixture)).iloc[0]
    assert (result.ba_paired_wins, result.ba_paired_ties, result.ba_paired_losses) == (1, 1, 1)
    np.testing.assert_allclose([result.ba_mean, result.delta_ba_mean, result.ba_std_ddof0],
                               [50., 0., np.sqrt(200/3)])
    return ['query_id_intersection', 'unbalanced_oa_ba', 'known_identity_overlap',
            'paired_win_tie_loss_and_std']


if __name__ == '__main__':
    self_test()
    run()
