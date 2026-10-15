"""Fixed prior/kernel2x2 on trained adapters; no additional target fitting labels."""

import numpy as np
import pandas as pd
import torch
from channel_operator import BASE, OUT as PARENT, build_blocks, metrics, save_json
from target_response_adapter import ResponseAdapter, residual_head, phi_prior, encode

OUT = BASE/'target_response_adapter_balanced_20261010'


def run():
    torch.set_num_threads(1)
    dest = OUT/'prior_kernel_crossover'
    dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        raise ValueError('Do not overwrite completed diagnostic')
    save_json(dest/'PROTOCOL.json', dict(
        reason='Adaptation harms classes absent from64 supports despite zero residual-column updates; frozen head weights do not freeze their feature-dependent prior scores',
        design='Original/adapted prior x original/adapted support geometry; refit the identical balanced ridge8 on the same64 labels in each cell',
        training='Reuse finished ordinary/guided adapters, no new neural training, no additional labels',
        selection='All four cells reported for all9 tasks and both adapters; no target-best per-port deployment',
        boundary='Crossed operators were not the neural-training objective; diagnostics not a fully retrained proposed model'))
    obs = pd.read_csv(BASE/'observations.csv')
    maps = np.load(PARENT/'boundary_maps.npy', mmap_mode='r')
    block = build_blocks()[1]
    paths = sorted(OUT.glob('*_models.npz'))
    with np.load(paths[0]) as m:
        channel_scale = m['channel_scale']
        center, scale, weight, bias = [torch.from_numpy(m[k]) for k in
            ('source_center', 'source_scale', 'source_coef', 'source_bias')]
    original_features = encode(block, ResponseAdapter(channel_scale).eval(), maps, np.arange(len(maps)))
    with torch.no_grad():
        original_phi, original_prior = phi_prior(torch.from_numpy(original_features), center, scale, weight, bias)
    results, classes, replays = [], [], []
    for path in paths:
        tag = path.stem.removesuffix('_models')
        with np.load(path) as saved:
            model = {k: saved[k] for k in saved.files}
        with np.load(OUT/f'{tag}_split.npz') as split:
            support, query, sy, query_ids = [split[k] for k in
                ('support_map_rows', 'query_map_rows', 'support_labels', 'query_obs_ids')]
        qm = obs.iloc[query_ids]
        port = str(qm.port.iloc[0])
        draw = int(tag.rsplit('draw', 1)[1])
        qy = qm.class_id.to_numpy()
        rows = np.concatenate([support, query])
        support_counts = np.bincount(sy, minlength=8)
        base_correct = model['head_only_pred'] == qy
        for arm in ('ordinary_adapter', 'scene_guided_adapter'):
            adapter = ResponseAdapter(channel_scale).eval()
            with torch.no_grad():
                adapter.log_gain.copy_(torch.from_numpy(model[arm+'_log_gain']))
                adapter.shift.copy_(torch.from_numpy(model[arm+'_shift']))
            features = encode(block, adapter, maps, rows)
            with torch.no_grad():
                changed_phi, changed_prior = phi_prior(torch.from_numpy(features), center, scale, weight, bias)
            phis = {'F': original_phi[rows], 'A': changed_phi}
            priors = {'F': original_prior[rows], 'A': changed_prior}
            for pk in ('FF', 'FA', 'AF', 'AA'):
                prior, phi = priors[pk[0]], phis[pk[1]]
                with torch.no_grad():
                    beta, _, _ = residual_head(phi[:64], prior[:64], torch.from_numpy(sy), balanced=True)
                    score = prior[64:]+phi[64:]@beta
                pred = score.argmax(1).numpy()
                if pk in ('FF', 'AA'):
                    ref = model['head_only_pred'] if pk == 'FF' else model[arm+'_pred']
                    errors = int((ref != pred).sum())
                    replays.append(dict(task=tag, arm=arm, cell=pk, prediction_disagreements=errors))
                    if errors:
                        raise ValueError('Original crossover diagonal does not replay')
                correct = pred == qy
                results.append(dict(port=port, draw=draw, adapter=arm, cell=pk, query=len(query),
                    **metrics(pred, qy), rescue_vs_FF=int((~base_correct & correct).sum()),
                    harm_vs_FF=int((base_correct & ~correct).sum())))
                for c in np.unique(qy):
                    mask = qy == c
                    count = int(support_counts[c])
                    classes.append(dict(port=port, draw=draw, adapter=arm, cell=pk, class_id=int(c),
                        n=int(mask.sum()), support_count=count,
                        coverage_group='absent' if count == 0 else 'singleton' if count == 1 else 'multiple',
                        recall=float(100*correct[mask].mean())))
                np.savez_compressed(dest/f'{tag}_{arm}_{pk}.npz', predictions=pred, residual_coef=beta.numpy())
            print('CROSSOVER', tag, arm, flush=True)
        pd.DataFrame(results).to_csv(dest/'results.csv', index=False)
        pd.DataFrame(classes).to_csv(dest/'by_class.csv', index=False)
    frame = pd.DataFrame(results)
    ref = frame[frame.cell.eq('FF')][['port', 'draw', 'adapter', 'OA', 'BA']].rename(columns={'OA': 'base_OA', 'BA': 'base_BA'})
    frame = frame.merge(ref, on=['port', 'draw', 'adapter'])
    frame['delta_OA'] = frame.OA-frame.base_OA
    frame['delta_BA'] = frame.BA-frame.base_BA
    frame.to_csv(dest/'paired_results.csv', index=False)
    mean = frame.groupby(['adapter', 'cell'])[['OA', 'BA', 'delta_OA', 'delta_BA']].mean().reset_index()
    mean.to_csv(dest/'summary.csv', index=False)
    frame.groupby(['port', 'adapter', 'cell'])[['OA', 'BA', 'delta_OA', 'delta_BA']].mean().reset_index().to_csv(dest/'port_summary.csv', index=False)
    cc = pd.DataFrame(classes).groupby(['port', 'draw', 'adapter', 'cell', 'coverage_group']).recall.mean().reset_index()
    cc.groupby(['adapter', 'cell', 'coverage_group']).recall.mean().reset_index().to_csv(dest/'coverage_summary.csv', index=False)
    save_json(dest/'COMPLETE.json', dict(tasks=len(paths), adapters=2, cells=4, replays=replays,
        no_neural_retraining=True, extra_labels=0))
    print(mean.to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
