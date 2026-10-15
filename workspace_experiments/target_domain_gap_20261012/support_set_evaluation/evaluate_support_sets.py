"""Paired support resampling with source, geometry, and queries held fixed."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment import OUT as PARENT, TASKS, SEEDS, fit_head, softmax, residual, write
from report import metric
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
DRAWS = (1201, 1202, 1203, 1204, 1205)
ARMS = ('original_residual64', 'global64_residual64')


def support_audit(frame, source, query, support):
    selected = frame.iloc[support]
    if len(support) != 64 or selected.mmsi.nunique() != 64:
        raise ValueError('Support must contain 64 distinct vessels')
    for comparison in (frame.iloc[source], frame.iloc[query]):
        if set(selected.mmsi) & set(comparison.mmsi):
            raise ValueError('Support identity overlaps source/query')
    if set(selected['product']) & set(frame.iloc[query]['product']):
        raise ValueError('Support scene overlaps query')


def summarize(m, paired):
    within = m.groupby(['task', 'split_seed', 'arm'], sort=False).agg(
        oa_mean=('oa','mean'), oa_sd=('oa','std'), oa_min=('oa','min'), oa_max=('oa','max'),
        ba_mean=('ba','mean'), ba_sd=('ba','std'), ba_min=('ba','min'), ba_max=('ba','max'))
    within.reset_index().to_csv(OUT/'within_split_summary.csv', index=False)
    rows = []
    for (task, arm), part in within.groupby(['task','arm'], sort=False):
        record = dict(task=task, arm=arm)
        for score in ('oa','ba'):
            record[score+'_mean'] = float(part[score+'_mean'].mean())
            record[score+'_support_sd'] = float(np.sqrt(np.mean(part[score+'_sd']**2)))
            record[score+'_worst'] = float(part[score+'_min'].min())
            record[score+'_best'] = float(part[score+'_max'].max())
        rows.append(record)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT/'task_summary.csv', index=False)
    gain_rows = []
    for task, part in paired.groupby('task', sort=False):
        record = dict(task=task, draws=len(part))
        for score in ('oa','ba'):
            column = 'delta_'+score
            sd = part.groupby('split_seed')[column].std()
            record[column] = float(part[column].mean())
            record[column+'_support_sd'] = float(np.sqrt(np.mean(sd**2)))
            record[column+'_worst'] = float(part[column].min())
            record[column+'_best'] = float(part[column].max())
            record[column+'_positive'] = int((part[column] > 1e-10).sum())
            record[column+'_negative'] = int((part[column] < -1e-10).sum())
            record[column+'_ties'] = int((abs(part[column]) <= 1e-10).sum())
        gain_rows.append(record)
    gains = pd.DataFrame(gain_rows)
    gains.to_csv(OUT/'paired_summary.csv', index=False)
    print('SUPPORT SENSITIVITY', flush=True)
    print(summary.to_string(index=False), flush=True)
    print('PAIRED GAIN', flush=True)
    print(gains.to_string(index=False), flush=True)


def main():
    frame = pd.read_csv(PARENT/'manifest.csv', dtype={'mmsi': str})
    label = frame.class_id.to_numpy()
    inventory, queued, baseline = [], [], []
    for task in TASKS:
        for seed in SEEDS:
            folder = PARENT/task/str(seed)
            split = dict(np.load(folder/'split.npz'))
            z = dict(np.load(folder/'features.npz'))
            rows = z.pop('rows'); loc = pd.Index(rows)
            query = split['query']; qi = loc.get_indexer(query)
            pool = np.sort(np.r_[split['support'], split['target_add']])
            if frame.iloc[pool].mmsi.nunique() != len(pool):
                raise ValueError('Candidate pool duplicates identities')
            with np.load(folder/'frozen/S/heads.npz') as saved:
                heads = dict(saved)
            features = dict(r50=z['r50'], dual=np.c_[z['r50'],z['r18']]/np.sqrt(2.))
            prior = {mode:x@heads[mode+'_coef']+heads[mode+'_bias'] for mode,x in features.items()}
            source_score = .5*softmax(prior['r50'][qi])+.5*softmax(prior['dual'][qi])
            baseline.append((task, seed, query, source_score))
            for draw in DRAWS:
                support = np.random.default_rng(draw).choice(pool, 64, replace=False)
                support_audit(frame, split['source'], query, support)
                si = loc.get_indexer(support)
                selected = np.r_[split['source'], support]
                fit_pos = loc.get_indexer(selected)
                scores = {name:{} for name in ARMS}
                for mode, x in features.items():
                    scores['original_residual64'][mode] = residual(x, prior[mode], si, label[support])[qi]
                    h = fit_head(x[fit_pos], label[selected], len(split['source']))
                    updated_prior = x@h['coef']+h['bias']
                    scores['global64_residual64'][mode] = residual(x, updated_prior, si, label[support])[qi]
                fused = {name:.5*softmax(s['r50'])+.5*softmax(s['dual']) for name,s in scores.items()}
                dest = OUT/task/str(seed)/str(draw)
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest/'predictions.npz', query_rows=query, support_rows=support, **fused)
                counts = np.bincount(label[support], minlength=8)
                inventory.append(dict(task=task, split_seed=seed, draw_seed=draw, pool_rows=len(pool),
                    classes_covered=int(np.count_nonzero(counts)),
                    missing_classes=','.join(str(i) for i in np.flatnonzero(counts == 0)),
                    **{'class_'+str(i):int(n) for i,n in enumerate(counts)}))
                queued.append((task, seed, draw, query, counts, fused))
            print('FROZEN', task, seed, flush=True)
    write(OUT/'PREDICTIONS_FROZEN.json', dict(support_draws=len(queued), head_fits=2*len(queued),
        neural_fits=0, labels_per_draw=64, selection_by_query=False))
    metrics, classes, paired, prior_metrics = [], [], [], []
    for task, seed, query, score in baseline:
        mm, _ = metric(label[query], score)
        prior_metrics.append(dict(task=task, split_seed=seed, **mm))
    for task, seed, draw, query, counts, scores in queued:
        yy = label[query]
        if len(np.unique(yy)) != 8:
            raise ValueError('Expected fixed eight-class query cohort')
        by_arm = {}
        for arm, score in scores.items():
            mm, recall = metric(yy, score)
            predicted = score.argmax(1)
            ship = pd.DataFrame(dict(ship=frame.iloc[query].mmsi.to_numpy(),
                label=yy, correct=(predicted == yy))).groupby(['ship','label'], as_index=False).correct.mean()
            metrics.append(dict(task=task, split_seed=seed, draw_seed=draw, arm=arm, **mm,
                ship_oa=100*float(ship.correct.mean()), ship_ba=100*float(ship.groupby('label').correct.mean().mean())))
            for c in range(8):
                classes.append(dict(task=task, split_seed=seed, draw_seed=draw, arm=arm, class_id=c,
                    support_count=int(counts[c]), query_count=int(np.sum(yy == c)), recall=100*recall[c]))
            by_arm[arm] = mm
        a, b = (scores[arm].argmax(1) for arm in ARMS)
        paired.append(dict(task=task, split_seed=seed, draw_seed=draw,
            delta_oa=by_arm[ARMS[1]]['oa']-by_arm[ARMS[0]]['oa'],
            delta_ba=by_arm[ARMS[1]]['ba']-by_arm[ARMS[0]]['ba'],
            rescues=int(np.sum((a != yy) & (b == yy))), harms=int(np.sum((a == yy) & (b != yy))),
            classes_covered=int(np.count_nonzero(counts))))
    m, p = pd.DataFrame(metrics), pd.DataFrame(paired)
    m.to_csv(OUT/'metrics.csv', index=False)
    p.to_csv(OUT/'paired_gains.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_recalls.csv', index=False)
    pd.DataFrame(inventory).to_csv(OUT/'support_coverage.csv', index=False)
    pd.DataFrame(prior_metrics).to_csv(OUT/'source_prior_metrics.csv', index=False)
    summarize(m, p)
    write(OUT/'COMPLETE.json', dict(support_draws=len(queued), classifier_fits=80,
        query_metrics_exclude_support=True, all_eight_query_classes_retained=True,
        only_support_changes_within_each_fixed_split=True, new_neural_fits=0))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
