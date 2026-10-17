"""One matched additive-kernel fit; reuse old runners with explicit strength."""
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
PREVIOUS = HERE.parent/'axis_access_margin_20261017'
sys.path.insert(0, str(PREVIOUS))
spec = importlib.util.spec_from_file_location('access_runner', PREVIOUS/'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
prior = previous.prior
KEYS = ['fold', 'seed', 'origin', 'draw', 'global_row']


@torch.no_grad()
def frozen_controls(model, data, features, tasks, seed, state, folder):
    output = previous.all_readouts(model, features, True)
    axis, common = output[0][0], output[1][0]
    variants = {
        'margin': ['margin_axis_1p5'],
        'joint': ['joint_restore_axis'],
        'additive': ['additive_axis_only'],
    }[state]
    metrics, predictions = [], []
    for task in tasks:
        ids, q, ns = task['ids'], task['q'], task['ns']
        av = features['valid'][ids]
        for arm in variants:
            if arm == 'joint_restore_axis':
                probability, _, _ = previous.predict(axis[ids], av, task, common[ids],
                    features['common_valid'][ids], axis_weight=1.)
            else:
                scale = 1.5 if arm == 'margin_axis_1p5' else 1.
                probability, _ = prior.old.predict(axis[ids], av, task, scale)
            p = probability.numpy()
            y = data['labels'][q]
            scores = prior.old.scores(y, p)
            scores['macro_brier'] = float(prior.old.risk(probability, torch.tensor(y)))
            metrics.append(dict(fold=data['fold'], seed=seed, origin=task['origin'],
                draw=task['draw'], arm=arm, query=len(q), **scores))
            for j, row in enumerate(q):
                predictions.append(dict(fold=data['fold'], seed=seed, origin=task['origin'],
                    draw=task['draw'], arm=arm, global_row=int(data['rows'][row]),
                    truth=int(y[j]), prediction=int(p[j].argmax()), axis_valid=bool(av[ns+j]),
                    **{f'p{k}': float(p[j,k]) for k in range(8)}))
    return metrics, predictions


def main():
    torch.set_num_threads(1)
    quality = dict(np.load(prior.HERE/'quality.npz'))
    manifest = pd.read_csv(prior.old.MANIFEST, dtype={'mmsi': str})
    old_predictions = pd.read_csv(PREVIOUS/'predictions.csv')
    old_predictions = old_predictions.set_index(['arm']+KEYS)
    metrics, predictions, controls, control_predictions, timings, replays = [], [], [], [], [], []
    total_start = time.perf_counter()
    with threadpool_limits(limits=1):
        for args in prior.old.folds():
            data, features, train_tasks, val_tasks, tasks, _, _ = prior.prepare_fold(args, manifest, quality)
            fold = data['fold']
            folder = HERE/fold
            folder.mkdir(exist_ok=True)
            with np.load(PREVIOUS/fold/'universal_features.npz') as cache:
                np.testing.assert_array_equal(cache['sample_ids'].astype(str), manifest.sample_id.to_numpy(str))
                for key in ('body', 'tokens', 'positions', 'valid'):
                    features['common_'+key] = torch.tensor(cache[key][data['rows']])
            features['quality'] = torch.zeros_like(features['quality'])
            labels = torch.tensor(data['labels'], dtype=torch.long)
            (folder/'data_protocol.json').write_text(json.dumps(dict(
                train=data['rows'][data['train']].tolist(), selection=data['rows'][data['val']].tolist(),
                query=data['rows'][data['query']].tolist(), source_episodes=len(train_tasks),
                axis_weight=1., common_weight=.5, margin_weight=.05, steps=200), indent=2), encoding='utf-8')
            print('FOLD', fold, 'QUERY', len(data['query']), flush=True)
            for seed in range(3):
                for arm, use_common in [('margin', False), ('joint', True), ('additive', True)]:
                    out = folder/f'{arm}_{seed}'
                    out.mkdir(exist_ok=True)
                    start = time.perf_counter()
                    if arm == 'additive':
                        model = previous.fit('joint', seed, features, train_tasks, val_tasks, labels,
                                             out, axis_weight=1.)
                        expected_schedule = np.load(PREVIOUS/fold/f'joint_{seed}'/'schedule.npz')['episode']
                        np.testing.assert_array_equal(np.load(out/'schedule.npz')['episode'], expected_schedule)
                    else:
                        model = previous.EvidenceReader('static')
                        model.load_state_dict(torch.load(PREVIOUS/fold/f'{arm}_{seed}'/'model.pt', weights_only=True))
                    elapsed = time.perf_counter()-start
                    m, p = previous.score(model, data, features, tasks, arm, seed, out, use_common,
                                          axis_weight=1. if arm == 'additive' else .5)
                    metrics.extend(m)
                    predictions.extend(p)
                    if arm != 'additive':
                        current = pd.DataFrame(p).set_index(KEYS)
                        columns = [f'p{k}' for k in range(8)]
                        reference = old_predictions.loc[arm].loc[current.index, columns]
                        error = float(np.abs(current[columns].to_numpy()-reference.to_numpy()).max())
                        assert error < 1e-10, (fold, arm, seed, error)
                        replays.append(dict(fold=fold, arm=arm, seed=seed, maximum_probability_error=error))
                    cm, cp = frozen_controls(model, data, features, tasks, seed, arm, out)
                    controls.extend(cm)
                    control_predictions.extend(cp)
                    timings.append(dict(fold=fold, seed=seed, arm=arm, seconds=elapsed,
                        trained=arm == 'additive', trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
                    print('DONE', fold, seed, arm, round(elapsed,2), flush=True)
                for filename, records in [('metrics', metrics), ('predictions', predictions),
                    ('frozen_metrics', controls), ('frozen_predictions', control_predictions),
                    ('timings', timings), ('baseline_replay', replays)]:
                    pd.DataFrame(records).to_csv(HERE/(filename+'.csv'), index=False)
    (HERE/'runtime.json').write_text(json.dumps(dict(total_seconds=time.perf_counter()-total_start), indent=2), encoding='utf-8')
    print(pd.DataFrame(metrics).groupby(['origin','arm'])[['oa','ba','macro_brier','nll']].mean().to_string(), flush=True)
    print(pd.DataFrame(controls).groupby(['origin','arm'])[['oa','ba','macro_brier','nll']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
