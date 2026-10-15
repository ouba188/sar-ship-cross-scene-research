"""Separate external version with the current matched post-tail source head."""

import run_coverage as c
from run_coverage import np, pd, OUT, ROOT, STUDY, TASKS, BUDGETS, DOSES
import argparse
import time
from compact_fusion_source import _expert_inputs
from unified_head_core import coefficients, late
from case_support_completion import COLS, predict, adapt_support


class CachedStrongFeatures:
    def __init__(self, dataset):
        self.directory = OUT/'strong_external'/dataset/dataset
        if not (self.directory/'FEATURES_READY.json').exists():
            raise ValueError('Strong external feature cache incomplete')
        receipt = c.read_json(self.directory/'FEATURES_READY.json')
        cache = c.Path(receipt.get('cache_directory', str(self.directory)))
        self.phi = np.load(cache/'phi.npy', mmap_mode='r')
        self.prior = np.load(cache/'prior.npy', mmap_mode='r')
        self.target_rows = np.arange(len(self.phi))

    def target(self, ix):
        return self.phi[ix], self.prior[ix]


def run(dataset):
    dest = OUT/'strong_external'/dataset/dataset
    if (dest/'PREDICTIONS_FROZEN.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    folder = TASKS/dataset
    info, data = c.read_json(folder/'PREPARED.json'), c.archive(folder/'target.npz')
    embedding = c.archive(folder/'selective_embedding.npz')
    np.testing.assert_array_equal(embedding['rows'], data['rows'])
    f = c.Features(folder)
    refdir = ROOT/'backbone_ablation_20261010/targets'/dataset/dataset
    heads = c.archive(refdir/'coefficients.npz')
    reference = c.archive(refdir/'predictions.npz')
    for key in ('rows', 'support'):
        np.testing.assert_array_equal(data[key], reference[key])
    n = len(data['rows'])
    if not (dest/'FEATURES_READY.json').exists():
        cache = (c.Path('C:/Users/25791/AppData/Local/Temp/SAR_frozen_candidate_coverage_20261012/budget/f0921')
                 if dataset == 'f0921' else dest)
        cache.mkdir(parents=True, exist_ok=True)
        width = f.target(np.arange(1))[0].shape[1]
        phi = np.lib.format.open_memmap(cache/'phi.npy', mode='w+', dtype='float64', shape=(n, width))
        prior = np.lib.format.open_memmap(cache/'prior.npy', mode='w+', dtype='float64', shape=(n, 8))
        # Preserve the reference head's dataset-specific FP32 PCA batch layout.
        xe_oss = _expert_inputs(f.raw_target, f.extra_target, f.model) if dataset == 'oss2' else None
        for lo in range(0, n, 2048):
            ix = np.arange(lo, min(n, lo+2048))
            phi[ix] = f.target(ix)[0]
            x = xe_oss[ix] if xe_oss is not None else _expert_inputs(f.raw_target[ix], f.extra_target[ix], f.model)
            prior[ix] = (x@heads['dual_coef']+heads['dual_intercept']).astype(np.float32).astype(float)
            if lo % 20480 == 0:
                print('STRONG FEATURES', dataset, min(n, lo+2048), '/', n, flush=True)
        phi.flush()
        prior.flush()
        c.write(dest/'FEATURES_READY.json', dict(rows=n, phi_width=width,
            cache_directory=str(cache),
            head=str(refdir/'coefficients.npz'), key='dual_coef/dual_intercept',
            source='Matched post-tail source head; unchanged expert geometry',
            runtime=c.sys.executable, seconds=time.perf_counter()-started))
    feature = CachedStrongFeatures(dataset)
    meta, folds = pd.read_csv(STUDY/'observations.csv'), c.archive(STUDY/'crossfit_evidence_20261011/source_folds.npz')
    initial = c.fit_initializer(f, dest, meta, folds)
    table = pd.read_csv(f.target_manifest)
    q = np.flatnonzero(data['eval_masks'].all(0))
    pool = np.unique(data['support'])
    choices = c.read_json(OUT/'FROZEN_PROTOCOL.json')['simple']
    modes = list(reference['modes'])
    checks = []
    for draw, old_s in enumerate(data['support']):
        rng = np.random.default_rng(124000+draw)
        order = np.r_[rng.permutation(old_s), rng.choice(np.setdiff1d(pool, old_s), 64, replace=False)]
        assert not np.intersect1d(order, q).size
        for budget in BUDGETS:
            s = order[:budget]
            sy = table.iloc[s].class_id.to_numpy(int)
            sp, ps = feature.target(s)
            delta = predict(initial, sp[:, COLS], np.zeros(budget, int), 'image_history')
            corrected = adapt_support(sp, delta, DOSES[budget])
            ae = {arm: coordinates.T@coefficients(coordinates@coordinates.T, ps, sy, budget/8.)
                  for arm, coordinates in [('raw', sp), ('initial', corrected)]}
            bss = c.base_block(data, embedding['embedding'], np.empty(0, int), s, info['eta'])
            ab = coefficients(bss, data['prior'][s], sy, budget/8.)
            values = {a: np.empty((len(q), 8)) for a in ('base', 'raw', 'initial', 'simple')}
            for lo in range(0, len(q), 2048):
                qi = q[lo:lo+2048]
                qp, pq = feature.target(qi)
                kb = c.base_block(data, embedding['embedding'], qi, s, info['eta'])[budget:]
                values['base'][lo:lo+len(qi)] = data['prior'][qi]+kb@ab
                for arm, weight in ae.items():
                    values[arm][lo:lo+len(qi)] = pq+qp@weight
                scale = 1.-choices[str(budget)]['intervention'] if choices[str(budget)]['arm'] == 'simple' else 1.
                values['simple'][lo:lo+len(qi)] = pq+scale*(values['raw'][lo:lo+len(qi)]-pq)
            saved = dict(query_ids=data['rows'][q], support_ids=data['rows'][s], support_labels=sy, base=values['base'])
            for arm in ('raw', 'initial', 'simple'):
                saved['expert_'+arm] = values[arm]
                saved['fusion_'+arm] = late(values['base'], values[arm])
            if budget == 64:
                row = dict(draw=draw)
                for layer, mode in [('expert', 'dual_refit_expert'), ('fusion', 'dual_refit_late')]:
                    actual = saved[layer+'_raw'].argmax(1)
                    expected = reference['predictions'][draw, modes.index(mode)][q]
                    changes = int((actual != expected).sum())
                    row[layer+'_class_changes'] = changes
                    np.testing.assert_array_equal(actual, expected)
                checks.append(row)
            np.savez_compressed(dest/f'{draw}_{budget}.npz', **saved)
        print('STRONG FROZEN', dataset, draw, flush=True)
    c.write(dest/'PREDICTIONS_FROZEN.json', dict(dataset=dataset, task=dataset,
        query_n=len(q), prediction_units=15, budgets=BUDGETS, original_B64_replay=checks,
        initializer_refit=True, new_shared_or_isolated_fits=0,
        target_manifest=str(f.target_manifest), query_labels_used_for_prediction=False,
        baseline='Current stronger matched post-tail dual_refit_late', prior_path=str(refdir/'coefficients.npz'),
        old_external_preserved=str(OUT/'tasks'/dataset/dataset),
        seconds=time.perf_counter()-started, runtime=c.sys.executable))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['oss2', 'f0921'], required=True)
    args = parser.parse_args()
    with c.threadpool_limits(limits=2):
        run(args.dataset)
