"""Label-free, one-factor diagnostic; never changes frozen predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import transfer as tr


def inputs(raw, scaler):
    feat = ((raw['centered'] - scaler['mean']) / scaler['std']
            * raw['usable'][:, :, None])
    level = ((raw['means'] - scaler['level_mean']) / scaler['level_std']
             * raw['usable'])
    return np.concatenate((feat.reshape(len(feat), -1), level, raw['usable']), 1).astype(np.float32)


def main():
    root = Path(__file__).resolve().parent
    out = root / 'normalization_diagnostic'
    out.mkdir(exist_ok=True)
    summaries, channels, cases = [], [], []
    for direction in ('oss1-to-oss2', 'oss2-to-oss1'):
        protocol = tr.read_json(root / direction / 'protocol.json')
        source = Path(protocol['source_root'])
        models = Path(protocol['source_frontend']) / 'models'
        sc = tr.load_npz(models / 'neighborhood_scaler.npz')
        sr = tr.load_npz(source / 'frontend_cache/frontend_raw.npz')
        target = tr.load_npz(Path(protocol['target_cache']) / 'frontend_raw.npz')
        fit = np.isin(sr['obs_id'], sc['fit_obs_id'])
        inactive = sc['std'] <= 1.000001e-5
        x = inputs(target, sc)
        repaired = x.copy()
        repaired[:, np.flatnonzero(inactive.ravel())] = 0
        source_x = inputs(sr, sc)
        source_repaired = source_x.copy()
        source_repaired[:, np.flatnonzero(inactive.ravel())] = 0
        for p, name in enumerate(('VV', 'VH')):
            rows = fit & sr['usable'][:, p].astype(bool)
            for c in np.flatnonzero(inactive[p]):
                a = sr['centered'][rows, p, c]
                b = target['centered'][target['usable'][:, p].astype(bool), p, c]
                channels.append(dict(direction=direction, polarization=name, channel=int(c),
                    source_min=float(a.min()), source_max=float(a.max()), source_std=float(a.std()),
                    scaler_std=float(sc['std'][p, c]), target_min=float(b.min()), target_max=float(b.max())))
        for seed in (17, 29, 43):
            model = tr.frozen_model(tr.front.Neighborhood(), tr.load_checkpoint(models / f'neighborhood_seed{seed}.pt'))
            with torch.no_grad():
                before = model(torch.from_numpy(x)).numpy()
                after = model(torch.from_numpy(repaired)).numpy()
                sb = model(torch.from_numpy(source_x)).numpy()
                sa = model(torch.from_numpy(source_repaired)).numpy()
            top = np.argsort(np.abs(before).max(1))[-8:][::-1]
            for row in top:
                cases.append(dict(direction=direction, seed=seed, obs_id=int(target['obs_id'][row]),
                    input_max_before=float(np.abs(x[row]).max()), input_max_after=float(np.abs(repaired[row]).max()),
                    residual_max_before=float(np.abs(before[row]).max()), residual_max_after=float(np.abs(after[row]).max())))
            summaries.append(dict(direction=direction, seed=seed,
                inactive_VV=int(inactive[0].sum()), inactive_VH=int(inactive[1].sum()),
                input_max_before=float(np.abs(x).max()), input_max_after=float(np.abs(repaired).max()),
                residual_max_before=float(np.abs(before).max()), residual_max_after=float(np.abs(after).max()),
                rows_residual_gt100_before=int((np.abs(before).max(1)>100).sum()),
                rows_residual_gt100_after=int((np.abs(after).max(1)>100).sum()),
                source_fit_residual_max_change=float(np.abs(sb[fit]-sa[fit]).max())))
    pd.DataFrame(channels).to_csv(out / 'source_floor_channels.csv', index=False)
    pd.DataFrame(cases).to_csv(out / 'extreme_cases.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
    tr.write_json(out / 'PROTOCOL.json', dict(target_labels_used=False, trained=False,
        modified='Only standardized dimensions with source std at existing 1e-5 floor set to zero.',
        official_predictions_changed=False,
        interpretation='Computational instability diagnostic only; not an evaluated replacement method.'))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()
