"""Frozen public full-classifier probe, source-selected local SAR readouts."""
from pathlib import Path
import importlib.util
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'oss1_joint_geometry_20261017'
sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location('frozen_public_training_helpers', OLD/'train.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
public = old.public
ARMS = ('uint8_spatial', 'uint8_bag', 'sigma0_spatial', 'uint8_lowpass')
ALPHAS = (0., .75, 4.)


def standardize(raw, valid, train):
    x = raw.reshape(len(raw), 18, 6).astype(float)
    source = x[train][valid[train]]
    out = x.copy()
    for offset in (0, 3):
        present = source[:, :, offset+2] > 0
        count = present.sum(0)
        mean = np.divide((source[:, :, offset]*present).sum(0), count,
                         out=np.zeros(18), where=count > 0)
        var = np.divide(((source[:, :, offset]-mean)**2*present).sum(0), count,
                        out=np.zeros(18), where=count > 0)
        out[:, :, offset] = (x[:, :, offset]-mean)/np.sqrt(var).clip(.05)
        out[:, :, offset] *= x[:, :, offset+2]
        for j in (offset+1, offset+2):
            out[:, :, j] = (x[:, :, j]-source[:, :, j].mean(0))/source[:, :, j].std(0).clip(.05)
    out[~valid] = 0
    return torch.tensor(out.reshape(len(x), -1), dtype=torch.float64)


def map_at_r(kernel, query_labels, support_labels):
    result = []
    for row, label in zip(kernel, query_labels):
        relevant = support_labels == label
        r = int(relevant.sum())
        if not r:
            result.append(np.nan)
            continue
        rel = relevant[np.argsort(-row, kind='stable')[:r]]
        result.append(float((np.cumsum(rel)/np.arange(1, r+1)*rel).sum()/r))
    return np.array(result)


def run_fold(fold, cache, episode, prefix, manifest, features):
    start = time.perf_counter()
    data = public.load_fold('oss1', fold, cache, episode, prefix, manifest,
                           np.load(old.FULL/'oss1_c4.npy'), np.load(old.FULL/'oss1_zero.npy'))
    source = np.r_[data['train'], data['val']]
    g = public.pca(data['raw']['G50'], source)
    d = public.pca(data['raw']['D18'], source)
    c = public.pca(data['c4'], source)
    base = public.design_block(c, np.arange(len(c)))
    z = public.design_block((g, d), np.arange(len(g)))
    prior_cache = {}
    selection = [old.make_task(data, base, z, e, prior_cache)
                 for e in old.make_episodes(data, 'selection')]
    if not selection:
        raise RuntimeError('No legal source selection episodes')
    reference = public.class_map(z[:, :-1], data['labels'], data['train'])
    prior_cache[tuple(sorted(source))] = (base@public.fit_head(c, data['labels'], source),
                                        z@public.fit_head((g, d), data['labels'], source), reference)
    tasks = []
    for (origin, draw), s in data['support'].items():
        task = old.make_task(data, base, z, dict(q=data['query'], s=s, prior_rows=source,
                             origin=origin, acquisition=str(data['query_frame'].acquisition.iloc[0])), prior_cache)
        task['draw'] = int(draw)
        tasks.append(task)
    previous = pd.read_csv(old.FULL/'oss1'/f'{fold}_metrics.csv')
    np.testing.assert_array_equal(features['sample_ids'][data['rows']],
                                  manifest.sample_id.to_numpy(str)[data['rows']])
    valid = torch.tensor(features['valid'][data['rows']])
    labels = torch.tensor(data['labels'], dtype=torch.long)
    records, predictions, selections, edges = [], [], [], []
    replay = []
    for task in tasks:
        baseline = task['old'].numpy()
        metric = old.scores(data['labels'][task['q']], baseline)
        prior = previous[(previous.arm == 'c4_class_fixed') & (previous.origin == task['origin']) &
                         (previous.draw == task['draw'])].iloc[0]
        error = max(abs(metric['oa']-prior.chip_oa), abs(metric['ba']-prior.chip_ba))
        if error > 1e-7:
            raise AssertionError(f'Baseline mismatch {fold}: {error}')
        replay.append(error)
    for arm in ARMS:
        raw = standardize(features[arm][data['rows']], valid.numpy(), data['train'])
        risks = {a: [] for a in ALPHAS}
        for task in selection:
            for a in ALPHAS:
                p, _ = old.predict(raw[task['ids']], valid[task['ids']], task, a)
                risks[a].append(float(old.risk(p, labels[task['q']])))
        means = {a: float(np.mean(values)) for a, values in risks.items()}
        alpha = min(means, key=lambda a: (means[a], a))
        for a, value in means.items():
            selections.append(dict(fold=fold, arm=arm, alpha=a, selected=a == alpha,
                                   source_macro_brier=value, selection_episodes=len(selection)))
        for task in tasks:
            before = task['old'].numpy()
            p0, _ = old.predict(raw[task['ids']], valid[task['ids']], task, 0.)
            if np.max(abs(p0.numpy()-before)) > 1e-7:
                raise AssertionError('Zero-dose endpoint differs')
            tic = time.perf_counter()
            p, h = old.predict(raw[task['ids']], valid[task['ids']], task, alpha)
            seconds = time.perf_counter()-tic
            p = p.numpy()
            y = data['labels'][task['q']]
            sy = data['labels'][task['s']]
            ns = task['ns']
            active = valid[task['q']].numpy() & (int(valid[task['s']].sum()) >= 2)
            k0 = task['qs'].numpy()
            k1 = k0+alpha*(h[ns:]@h[:ns].T).numpy()
            k1[~active] = k0[~active]
            coefficient0 = public.coefficients(task['ss'].numpy(), task['prior_s'].numpy(), sy, 8.)
            ss1 = task['ss'].numpy()+alpha*(h[:ns]@h[:ns].T).numpy()
            coefficient1 = public.coefficients(ss1, task['prior_s'].numpy(), sy, 8.)
            ap0, ap = map_at_r(k0, y, sy), map_at_r(k1, y, sy)
            result, baseline = old.scores(y, p), old.scores(y, before)
            result.update(fold=fold, arm=arm, origin=task['origin'], draw=task['draw'], alpha=alpha,
                          query=len(y), valid_query=int(active.sum()), support=ns,
                          baseline_oa=baseline['oa'], baseline_ba=baseline['ba'],
                          delta_oa=result['oa']-baseline['oa'], delta_ba=result['ba']-baseline['ba'],
                          delta_brier=result['brier']-baseline['brier'], delta_nll=result['nll']-baseline['nll'],
                          macro_brier=float(old.risk(torch.tensor(p), labels[task['q']])),
                          baseline_macro_brier=float(old.risk(task['old'], labels[task['q']])),
                          support_map_r=float(np.nanmean(ap)), baseline_support_map_r=float(np.nanmean(ap0)),
                          retrieval_eligible=int(np.isfinite(ap).sum()),
                          rescued=int(((before.argmax(1) != y) & (p.argmax(1) == y)).sum()),
                          harmed=int(((before.argmax(1) == y) & (p.argmax(1) != y)).sum()),
                          cached_solve_seconds=seconds)
            result['kernel_risk_gain'] = result['baseline_macro_brier']-result['macro_brier']
            records.append(result)
            for j, index in enumerate(task['q']):
                predictions.append(dict(fold=fold, arm=arm, origin=task['origin'], draw=task['draw'],
                    global_row=int(data['rows'][index]), truth=int(y[j]), baseline=int(before[j].argmax()),
                    prediction=int(p[j].argmax()), baseline_true_probability=before[j, y[j]],
                    true_probability=p[j, y[j]], baseline_ap=ap0[j], ap=ap[j],
                    support_class_count=int((sy == y[j]).sum()), **{f'p{k}': p[j, k] for k in range(8)}))
                if active[j] and ((before[j].argmax() == y[j]) != (p[j].argmax() == y[j])):
                    opponent = int(p[j].argmax() if before[j].argmax() == y[j] else before[j].argmax())
                    c0 = coefficient0[:, y[j]]-coefficient0[:, opponent]
                    c1 = coefficient1[:, y[j]]-coefficient1[:, opponent]
                    kernel_change = (k1[j]-k0[j])*c0
                    refit_change = k1[j]*(c1-c0)
                    np.testing.assert_allclose((kernel_change+refit_change).sum(), k1[j]@c1-k0[j]@c0, atol=1e-10)
                    for pos, support_index in enumerate(task['s']):
                        edges.append(dict(fold=fold, arm=arm, origin=task['origin'], draw=task['draw'],
                            query_global_row=int(data['rows'][index]), support_global_row=int(data['rows'][support_index]),
                            query_class=int(y[j]), support_class=int(sy[pos]), opponent_class=opponent,
                            baseline=int(before[j].argmax()), prediction=int(p[j].argmax()),
                            kernel_before=k0[j, pos], kernel_increment=k1[j, pos]-k0[j, pos],
                            kernel_margin_change=kernel_change[pos], support_refit_margin_change=refit_change[pos],
                            total_expert_margin_change=kernel_change[pos]+refit_change[pos]))
    return records, predictions, selections, edges, dict(fold=fold, baseline_replays=len(replay),
            max_baseline_error=max(replay), seconds=time.perf_counter()-start)


def main():
    torch.set_num_threads(1)
    folder = HERE/'classifier'
    folder.mkdir(exist_ok=True)
    protocol = dict(arms=ARMS, alpha_candidates=ALPHAS, alpha_selection='source validation macro Brier; same and cross episodes',
        standardization='legal source train only; missing response masked, coverage/presence explicit',
        task='frozen four OSS1 acquisition folds; B16; five support draws; same/cross; no new SGD training',
        integration='PSD SS and QS increment; refit support coefficients; frozen priors and equal full fusion',
        retrieval='support-set MAP@R only, absent-class queries excluded with counts',
        limits='exploratory previously used target cohorts; no independent confirmation claim; cached timing not end-to-end')
    (folder/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    features = dict(np.load(HERE/'spatial_descriptors.npz'))
    manifest = pd.read_csv(old.MANIFEST, dtype={'mmsi': str})
    all_metrics, all_predictions, all_selection, all_edges, receipts = [], [], [], [], []
    with threadpool_limits(limits=1), torch.no_grad():
        for args in old.folds():
            print('FOLD', args[0], flush=True)
            metrics, predictions, selection, edges, receipt = run_fold(*args, manifest, features)
            all_metrics.extend(metrics)
            all_predictions.extend(predictions)
            all_selection.extend(selection)
            all_edges.extend(edges)
            receipts.append(receipt)
            print(receipt, flush=True)
    pd.DataFrame(all_metrics).to_csv(folder/'metrics.csv', index=False)
    pd.DataFrame(all_predictions).to_csv(folder/'predictions.csv', index=False)
    pd.DataFrame(all_selection).to_csv(folder/'source_selection.csv', index=False)
    pd.DataFrame(all_edges).to_csv(folder/'changed_support_edges.csv', index=False)
    pd.DataFrame(receipts).to_csv(folder/'replay_and_cost.csv', index=False)
    print(pd.DataFrame(all_metrics).groupby(['origin', 'arm'])[['oa', 'ba', 'delta_oa', 'delta_ba', 'kernel_risk_gain']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
