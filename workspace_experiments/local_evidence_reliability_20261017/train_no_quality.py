"""Matched retrained control: static reader without extra radiometric inputs."""
import json
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import train_reliability as run


def main():
    torch.set_num_threads(1)
    quality = dict(np.load(run.HERE/'quality.npz'))
    manifest = pd.read_csv(run.old.MANIFEST, dtype={'mmsi': str})
    metrics, predictions, fits = [], [], []
    with threadpool_limits(limits=1):
        for args in run.old.folds():
            data, features, fit_tasks, val_tasks, tasks, _, _ = run.prepare_fold(args, manifest, quality)
            features = {**features, 'quality': torch.zeros_like(features['quality'])}
            labels = torch.tensor(data['labels'], dtype=torch.long)
            for seed in range(3):
                folder = run.HERE/data['fold']/f'static_no_quality_{seed}'
                folder.mkdir(exist_ok=True)
                tic = time.perf_counter()
                model = run.fit('static', seed, features, fit_tasks, val_tasks, labels, folder)
                seconds = time.perf_counter()-tic
                m, p = run.score_model(model, data, features, tasks, 'static_no_quality', seed, folder)
                metrics.extend(m)
                predictions.extend(p)
                fits.append(dict(fold=data['fold'], seed=seed, arm='static_no_quality', seconds=seconds))
                print('CONTROL', data['fold'], seed, round(seconds, 2), flush=True)
    pd.DataFrame(metrics).to_csv(run.HERE/'trained_no_quality_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(run.HERE/'trained_no_quality_predictions.csv', index=False)
    pd.DataFrame(fits).to_csv(run.HERE/'trained_no_quality_fits.csv', index=False)
    (run.HERE/'RETRAINED_CONTROL.json').write_text(json.dumps(dict(
        reason='Frozen quality zeroing shows dependence, not incremental benefit; retrain same static architecture without descriptors',
        timing='added after primary experiment as labeled root analysis, not prespecified main comparison',
        unchanged='four folds, seeds, initialization, sampler, 200steps, loss, alpha=1, old full model',
        no_new_selection=True, original_results_preserved=True), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
