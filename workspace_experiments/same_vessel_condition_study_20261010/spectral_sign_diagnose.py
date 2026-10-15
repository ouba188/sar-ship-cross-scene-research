"""Fixed-weight diagnostic of signed residuals at a formerly nonnegative boundary."""

import numpy as np
import pandas as pd
import torch
from spectral_residual import BASE, OUT, OLD, ARMS, SpectralResidual, build_blocks, phi, residual_head, save_json, metrics


def encode(blocks, model, maps, residuals, rows, rectify):
    result = np.empty((len(rows), 512), np.float32)
    with torch.inference_mode():
        for lo in range(0, len(rows), 16):
            ix = rows[lo:lo+16]
            h = model(torch.from_numpy(np.array(maps[ix])), torch.from_numpy(np.array(residuals[ix])))
            if rectify:
                h = h.relu()
            result[lo:lo+len(ix)] = blocks(h).mean((2, 3)).numpy()
    return result


def run():
    torch.set_num_threads(1)
    if not (OUT/'COMPLETE.json').exists():
        raise ValueError('Wait for the fixed original experiment to finish')
    target = OUT/'sign_diagnosis'
    target.mkdir(exist_ok=True)
    maps = np.load(OUT/'prefix_maps.npy', mmap_mode='r')
    bands = {arm: maps if arm == 'full' else np.load(OUT/(arm+'_branch.npy'), mmap_mode='r') for arm in ARMS}
    obs = pd.read_csv(BASE/'observations.csv')
    blocks = build_blocks()
    records, classes, energies, replays = [], [], [], []
    for path in sorted(OUT.glob('*_models.npz')):
        tag = path.stem.removesuffix('_models')
        with np.load(OLD/f'{tag}_split.npz') as z:
            support, query, labels, ids = [z[k] for k in
                ('support_map_rows', 'query_map_rows', 'support_labels', 'query_obs_ids')]
        with np.load(OLD/f'{tag}_models.npz') as z:
            state = {k: z[k] for k in z.files}
        truth = obs.iloc[ids].class_id.to_numpy()
        port, draw = obs.iloc[ids].port.iloc[0], int(tag.rsplit('draw', 1)[1])
        with np.load(path) as z:
            saved = {k: z[k] for k in z.files}
        for arm in ARMS:
            h = np.array(maps[support])
            delta = np.array(bands[arm][support])*saved[arm+'_gate'][None, :, None, None]
            adapted = h+delta
            negative = np.minimum(adapted, 0)
            mean_norm = max(float(np.linalg.norm(h.mean((2, 3)))), 1e-12)
            energies.append(dict(task=tag, arm=arm, original_negative_fraction=float((h < 0).mean()),
                original_zero_fraction=float((h == 0).mean()), new_negative_fraction=float((adapted < 0).mean()),
                negative_energy_vs_update=float(np.sum(negative**2)/max(np.sum(delta**2), 1e-12)),
                negative_rms_vs_original=float(np.linalg.norm(negative)/max(np.linalg.norm(h), 1e-12)),
                raw_relative_channel_mean_shift=float(np.linalg.norm(delta.mean((2, 3)))/mean_norm),
                rectified_relative_channel_mean_shift=float(np.linalg.norm((np.maximum(adapted, 0)-h).mean((2, 3)))/mean_norm)))
            predictions = {'raw': saved[arm+'_pred']}
            if arm == 'full':
                if (adapted < 0).any():
                    raise ValueError('Full-band positive scaling unexpectedly became negative')
                predictions['rectified'] = predictions['raw']
            else:
                model = SpectralResidual()
                with torch.no_grad():
                    model.gate.copy_(torch.from_numpy(saved[arm+'_gate']))
                for mode, rectify in [('raw', False), ('rectified', True)]:
                    features = encode(blocks, model, maps, bands[arm], np.r_[support, query], rectify)
                    with torch.no_grad():
                        ph, pri = phi(torch.from_numpy(features), state)
                        beta, _, _ = residual_head(ph[:64], pri[:64], torch.from_numpy(labels), balanced=True)
                        score = pri[64:]+ph[64:]@beta
                    pred = score.argmax(1).numpy()
                    if mode == 'raw':
                        err = int((pred != saved[arm+'_pred']).sum())
                        if err:
                            raise ValueError('Original spectral predictions did not replay')
                        replays.append(dict(task=tag, arm=arm, prediction_disagreements=err))
                    predictions[mode] = pred
            for mode, pred in predictions.items():
                records.append(dict(port=port, draw=draw, arm=arm, mode=mode, **metrics(pred, truth)))
                for c in np.unique(truth):
                    count = int((labels == c).sum())
                    classes.append(dict(port=port, draw=draw, arm=arm, mode=mode, class_id=int(c),
                        coverage='absent' if count == 0 else 'singleton' if count == 1 else 'multiple',
                        recall=float(100*(pred[truth == c] == c).mean())))
            print('SIGN_REPLAY', tag, arm, flush=True)
    rows = pd.DataFrame(records)
    raw = rows[rows['mode'].eq('raw')][['port', 'draw', 'arm', 'OA', 'BA']].rename(columns={'OA': 'raw_OA', 'BA': 'raw_BA'})
    rows = rows.merge(raw, on=['port', 'draw', 'arm'])
    rows['delta_OA'] = rows.OA-rows.raw_OA
    rows['delta_BA'] = rows.BA-rows.raw_BA
    rows.to_csv(target/'results.csv', index=False)
    rows.groupby(['arm', 'mode'])[['OA', 'BA', 'delta_OA', 'delta_BA']].mean().reset_index().to_csv(target/'summary.csv', index=False)
    rows.groupby(['port', 'arm', 'mode'])[['OA', 'BA', 'delta_OA', 'delta_BA']].mean().reset_index().to_csv(target/'port_summary.csv', index=False)
    cl = pd.DataFrame(classes)
    cl.to_csv(target/'by_class.csv', index=False)
    cl.groupby(['port', 'draw', 'arm', 'mode', 'coverage']).recall.mean().reset_index().groupby(
        ['arm', 'mode', 'coverage']).recall.mean().reset_index().to_csv(target/'coverage_summary.csv', index=False)
    en = pd.DataFrame(energies)
    en.to_csv(target/'energy_by_task.csv', index=False)
    en.groupby('arm').mean(numeric_only=True).reset_index().to_csv(target/'energy_summary.csv', index=False)
    save_json(target/'COMPLETE.json', dict(tasks=9, original_replays=replays, new_neural_training=0,
        boundary='Post-hoc signedness diagnosis with fixed learned gates and same64-label residual-head refit; clipping also changes channel means and update size, so not an isolated physical cause; not rectification-aware retraining or target-selected deployment'))
    print(pd.read_csv(target/'summary.csv').to_string(index=False))
    print(pd.read_csv(target/'energy_summary.csv').to_string(index=False))


if __name__ == '__main__':
    run()
