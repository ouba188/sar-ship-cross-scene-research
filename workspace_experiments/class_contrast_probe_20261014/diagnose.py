"""Paired error groups and representation checks; no new model selection."""
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from run_probe import OUT, run, contrast, PAIRS, analysis


def main():
    metrics = pd.read_csv(OUT / 'metrics.csv')
    keys = ['dataset', 'study', 'cohort', 'budget', 'cell', 'draw', 'port', 'episode', 'mode']
    ref = metrics[metrics.arm == 'class_mean'][keys + analysis.VALUES]
    paired = metrics.merge(ref, on=keys, suffixes=('', '_reference'), validate='many_to_one')
    for metric in analysis.VALUES:
        paired[metric + '_delta'] = paired[metric] - paired[metric + '_reference']
    paired.to_csv(OUT / 'paired_metrics.csv', index=False)
    paired.groupby(['dataset', 'budget', 'cell', 'port', 'mode', 'arm'])[
        [m + '_delta' for m in analysis.VALUES]].mean().reset_index().to_csv(OUT / 'port_class_effects.csv', index=False)
    # Average repeated support draws first; resample scenes within each fixed port.
    primary = paired[(paired['mode'] == 'full') & (paired.arm != 'original')]
    scene = primary.groupby(['dataset', 'budget', 'cell', 'port', 'episode', 'arm'])[['oa_delta', 'ba_delta']].mean().reset_index()
    scene.to_csv(OUT / 'scene_effects.csv', index=False)
    rng = np.random.default_rng(20261014)
    intervals = []
    for keys0, g in scene.groupby(['dataset', 'budget', 'cell', 'arm']):
        for metric in ['oa_delta', 'ba_delta']:
            replicates = []
            for _, pg in g.groupby('port'):
                values = pg[metric].to_numpy()
                replicates.append(rng.choice(values, size=(5000, len(values)), replace=True).mean(1))
            samples = np.mean(replicates, axis=0)
            intervals.append(dict(zip(['dataset', 'budget', 'cell', 'arm'], keys0), metric=metric,
                                  low=float(np.quantile(samples, .025)), high=float(np.quantile(samples, .975)),
                                  scenes=len(g), positive_scenes=int((g[metric] > 1e-10).sum()),
                                  negative_scenes=int((g[metric] < -1e-10).sum()),
                                  scope='Conditional on these ports/scenes and reused development data, not independent confirmation'))
    pd.DataFrame(intervals).to_csv(OUT / 'scene_bootstrap.csv', index=False)
    events = []
    metadata = {'ports': run.old.source_metadata()}
    for dataset in ('f0921', 'oss2'):
        metadata[dataset] = pd.read_csv(run.old.OLD / f'{dataset}_metadata.csv', dtype={'identity': str})
    for path in sorted((OUT / 'predictions').glob('*.npz')):
        dataset, rest = path.stem.split('_', 1)
        episode, budget, draw, cell = rest.split('_', 3)
        f = run.load_npz(path)
        meta = metadata[dataset]
        q, s = f['query_rows'], f['support_rows']
        y = meta.iloc[q].class_id.to_numpy(int)
        sy = meta.iloc[s].class_id.to_numpy(int)
        pred = f['full'].argmax(2)
        reference = pred[list(f['arms']).index('class_mean')]
        for ai, arm in enumerate(f['arms']):
            if arm in ('original', 'class_mean'):
                continue
            for i in np.flatnonzero((pred[ai] == y) != (reference == y)):
                events.append(dict(dataset=dataset, episode=int(episode), budget=int(budget), draw=int(draw),
                                   cell=cell, arm=arm, row=int(q[i]), ship=str(meta.iloc[q[i]].identity),
                                   port=str(meta.iloc[q[i]]['port']) if 'port' in meta else dataset,
                                   class_id=int(y[i]), old_pred=int(reference[i]), new_pred=int(pred[ai, i]),
                                   rescued=bool(pred[ai, i] == y[i]), supports_of_true_class=int(np.sum(sy == y[i]))))
    event = pd.DataFrame(events)
    event.to_csv(OUT / 'reference_change_events.csv', index=False)
    event.groupby(['dataset', 'budget', 'cell', 'arm', 'class_id', 'rescued']).agg(
        exposures=('row', 'size'), ships=('ship', 'nunique'), scenes=('episode', 'nunique'),
        zero_support_exposures=('supports_of_true_class', lambda v: int((v == 0).sum()))
    ).reset_index().to_csv(OUT / 'reference_change_groups.csv', index=False)
    diags = []
    query_stats = []
    for dataset in ('f0921', 'oss2', 'ports'):
        with (OUT / f"{'ports' if dataset == 'ports' else 'external'}_banks.pkl").open('rb') as h:
            bank = pickle.load(h)['contrast_weighted']
        if dataset == 'ports':
            f = run.load_npz(run.EXPAND / 'encoded.npz')
        else:
            f = run.load_npz(run.old.OLD / f'{dataset}_encoded.npz')
        meta = metadata[dataset]
        paths = sorted((OUT / 'predictions').glob(f'{dataset}_*.npz'))
        rows = np.unique(np.concatenate([run.load_npz(p)['query_rows'] for p in paths]))
        ix = np.searchsorted(f['rows'], rows)
        np.testing.assert_array_equal(f['rows'][ix], rows)
        z = np.c_[f['r'][ix], f['t'][ix]]
        y = meta.iloc[rows].class_id.to_numpy(int)
        d = contrast(z, bank) / bank['pair_scale']
        target_scores = []
        for j, (a, b) in enumerate(PAIRS):
            va, vb = d[y == a, j], d[y == b, j]
            score = np.nan
            if min(len(va), len(vb)) >= 2:
                score = (va.mean() - vb.mean()) ** 2 / (.5 * (va.var() + vb.var()) + .01)
            target_scores.append(score)
            diags.append(dict(dataset=dataset, class_a=int(a), class_b=int(b), source_weight=bank['weights'][j],
                              target_separation=score, query_a=len(va), query_b=len(vb),
                              abs_scaled_over2=float(np.mean(abs(d[:, j]) > 2)),
                              tanh_variance=float(np.var(np.tanh(d[:, j]))), linear_variance=float(np.var(d[:, j]))))
        valid = np.isfinite(target_scores)
        corr = spearmanr(bank['weights'][valid], np.array(target_scores)[valid]).statistic
        query_stats.append(dict(dataset=dataset, query_observations=len(rows), query_ships=meta.iloc[rows].identity.nunique(),
                                class_counts=np.bincount(y, minlength=8).tolist(), saturation=float(np.mean(abs(d) > 2)),
                                source_weight_target_separation_spearman=float(corr), assessable_pairs=int(valid.sum()),
                                note='Target-label diagnostic only; estimates confounded by query composition, not used in fit or deployment'))
    pd.DataFrame(diags).to_csv(OUT / 'target_pair_diagnostics.csv', index=False)
    (OUT / 'DATA_DIAGNOSTICS.json').write_text(json.dumps(query_stats, indent=2), encoding='utf-8')
    print(json.dumps(query_stats, indent=2), flush=True)
    print(paired[(paired['mode'] == 'full') & (paired.budget == 64) & (paired.cell == 'other_acquisition')]
          .groupby(['dataset', 'port', 'arm'])[['oa_delta', 'ba_delta']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
