"""Same pixels, different display orientation: distinguish CNN sensitivity from viewing physics."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
SOURCE = Path('E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
GEOMETRY = ROOT/'relative_observation_geometry_20261009'
OUT = GEOMETRY/'display_rotation'
PARENT = ROOT/'protected_drift_activation_20261008/ports'
REFERENCE = ROOT/'direct_incidence_comparison_20261009/ports'


def extract():
    import torch
    sys.path.insert(0, 'E:/临时会话/visual_reliable_baseline')
    from extract_full_six_spatial_features import build_encoder, read_one
    from concurrent.futures import ThreadPoolExecutor
    if (OUT/'EXTRACTED.json').exists():
        raise ValueError('Preserve completed feature extraction')
    frame = pd.read_csv(SOURCE/'manifest.csv', low_memory=False)
    meta = pd.read_csv(GEOMETRY/'relative_conditions.csv')
    np.testing.assert_array_equal(frame.instance_id, meta.instance_id)
    rows = np.flatnonzero(meta.relative_status.eq('ok').to_numpy())
    np.save(OUT/'rows.npy', rows)
    device = torch.device('cuda')
    torch.set_num_threads(2)
    encoder = build_encoder(device)
    pooled = np.empty((len(rows), 2, 2048), dtype=np.float32)
    def read(row):
        return np.stack([read_one(frame.loc[row, 'image_relpath_'+pol]) for pol in ['vv', 'vh']])
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for lo in range(0, len(rows), 32):
            batch = np.stack(list(pool.map(read, rows[lo:lo+32])))
            tensor = (torch.from_numpy(batch).to(device=device, dtype=torch.float32)/255.-.5)/.25
            for rotation in [0, 1]:
                rotated = torch.rot90(tensor, rotation, dims=(-2, -1))
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    values = encoder(rotated).flatten(2).transpose(1, 2)
                spatial = values.float().cpu().numpy().astype(np.float16).astype(np.float32)
                pooled[lo:lo+len(batch), rotation] = spatial.mean(1)
            if lo % 640 == 0:
                print(f'EXTRACT {lo+len(batch)}/{len(rows)}', flush=True)
    archived = np.asarray(np.load(SOURCE/'pooled_features.npy', mmap_mode='r')[rows], dtype=float)
    zero = pooled[:, 0].astype(float)
    cosine = np.sum(archived*zero, axis=1)/(np.linalg.norm(archived, axis=1)*np.linalg.norm(zero, axis=1))
    np.save(OUT/'pooled_zero_rot90.npy', pooled)
    check = dict(rows=len(rows), archived_replay_cosine_min=float(cosine.min()),
        archived_replay_cosine_median=float(np.median(cosine)),
        operation='torch.rot90 after original128resize; same VV/VH rotation; no interpolation during rotation',
        encoder='Existing paired R50 ImageNetV2, bfloat16 forward, spatial float16 then float32 mean as parent',
        metadata='Physical incidence/relative-viewing conditions unchanged; not a simulated SAR observation')
    (OUT/'EXTRACTED.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    print(json.dumps(check), flush=True)
    if cosine.min() < .995:
        raise ValueError('Encoder replay insufficient; do not interpret rotation evaluation')


def base_phi(bank, g, u):
    r = g['stable_rank']
    core = bank[:, :128]-.5*(bank[:, 192+r:] @ g['drift_to_xs'])
    normalized = (core-g['base_center'])/g['base_scale']/np.sqrt(128)
    return np.c_[normalized, bank[:, 192:192+r]/np.sqrt(r), u/np.sqrt(2), np.ones(len(bank))]


def replace_r50(raw, original, g):
    bank = np.array(original, dtype=float, copy=True)
    model = g['model']
    bank[:, :128] = (model['pca'].transform(raw)-model['center'])/model['scale']
    standardized = (raw-g['feature_center'])/g['feature_scale']
    r = g['stable_rank']
    bank[:, 192:192+r] = (standardized @ g['qs']-g['stable_center'])/g['stable_scale']
    bank[:, 192+r:] = standardized @ g['qd']
    return bank


def evaluate():
    from direct_incidence_conditioning import condition_features
    from residual_expert import fit_correction
    from run_conditioned_vessel_drift import SIDECAR
    from run_generic_transfer import metrics
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed perturbation probe')
    check = json.loads((OUT/'EXTRACTED.json').read_text(encoding='utf-8'))
    if check['archived_replay_cosine_min'] < .995:
        raise ValueError('Encoder replay insufficient')
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'product', 'port', 'class_id'])
    side = pd.read_csv(SIDECAR, usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    frame = frame.merge(side, on=['instance_id', 'product'], how='left', sort=False, validate='one_to_one')
    frame.loc[frame.status.ne('ok'), 'incidence_deg'] = np.nan
    meta = pd.read_csv(GEOMETRY/'relative_conditions.csv')
    frame['relative_acute_deg'] = meta.relative_acute_deg
    selected = np.load(OUT/'rows.npy')
    new_raw = np.load(OUT/'pooled_zero_rot90.npy', mmap_mode='r')
    records, observations, verification = [], [], []
    for port in sorted(frame.loc[selected, 'port'].unique()):
        parent = PARENT/port
        with (parent/'geometry.pkl').open('rb') as f:
            g = pickle.load(f)
        with (REFERENCE/port/'direct_heads.pkl').open('rb') as f:
            head = pickle.load(f)['joint_base']
        with np.load(REFERENCE/port/'policies.npz') as f:
            old = {k: f[k] for k in f.files}
        direct = list(old['mode_names']).index('direct_condition')
        tr = old['query_rows']
        bank = np.load(parent/'target_bank.npy', mmap_mode='r')
        u = condition_features(frame.loc[tr, 'incidence_deg'].to_numpy())
        phi = base_phi(bank, g, u)
        prior = np.c_[bank, u] @ head['coef']+head['intercept']
        chosen = selected[frame.loc[selected, 'port'].eq(port).to_numpy()]
        indices, raw_indices = np.searchsorted(tr, chosen), np.searchsorted(selected, chosen)
        archived_raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')[chosen]
        bank_difference = replace_r50(archived_raw, bank[indices], g)-bank[indices]
        bank_replay = float(np.max(np.abs(bank_difference)))
        bank_relative = float(np.linalg.norm(bank_difference)/np.linalg.norm(bank[indices]))
        source_score_error = float(np.max(np.abs(bank_difference @ head['coef'][:bank.shape[1]])))
        if bank_relative > 1e-4 or source_score_error > 1e-3:
            raise ValueError(f'Raw-to-bank mapping mismatch {port} {bank_relative} {source_score_error}')
        new_banks = [replace_r50(np.asarray(new_raw[raw_indices, rotation], dtype=float), bank[indices], g) for rotation in [0, 1]]
        new_phi = [base_phi(b, g, u[indices]) for b in new_banks]
        new_prior = [np.c_[b, u[indices]] @ head['coef']+head['intercept'] for b in new_banks]
        replay = 0.
        truth = frame.loc[chosen, 'class_id'].to_numpy()
        for draw, ev in enumerate(old['eval_mask']):
            support = np.searchsorted(tr, old['support_rows'][draw])
            coefficient = fit_correction(phi[support], frame.loc[old['support_rows'][draw], 'class_id'].to_numpy(), prior[support])
            replay = max(replay, float(np.max(np.abs(prior+phi @ coefficient-old['base_scores'][draw, direct]))))
            scores = [old['base_scores'][draw, direct, indices]]+[b+x @ coefficient for b, x in zip(new_prior, new_phi)]
            eligible = ev[indices]
            for mode, score in zip(['archived', 'reextracted0', 'rotated90'], scores):
                pred = score.argmax(1)
                result = metrics(truth[eligible], pred[eligible], 8)
                records.append(dict(port=port, draw=draw, mode=mode, n=int(eligible.sum()), oa=result['acc'], ba=result['ba']))
                for j in np.flatnonzero(eligible):
                    observations.append(dict(manifest_row=int(chosen[j]), port=port, draw=draw, mode=mode,
                        class_id=int(truth[j]), relative_angle=float(frame.loc[chosen[j], 'relative_acute_deg']),
                        correct=int(pred[j] == truth[j]), prediction=int(pred[j])))
        if replay > 3e-6:
            raise ValueError(f'Base replay failed {port} {replay}')
        verification.append(dict(port=port, cached_base_score_replay_error=replay, raw_to_bank_replay_error=bank_replay,
            raw_to_bank_relative_l2_error=bank_relative, raw_to_source_score_error=source_score_error))
    table, observation = pd.DataFrame(records), pd.DataFrame(observations)
    table.to_csv(OUT/'metrics.csv', index=False)
    observation.to_csv(OUT/'predictions.csv', index=False)
    pd.DataFrame(verification).to_csv(OUT/'verification.csv', index=False)
    summary = table.groupby(['port', 'mode']).mean(numeric_only=True).groupby('mode').mean()
    summary.to_csv(OUT/'summary.csv')
    observation['angle_bin'] = pd.cut(observation.relative_angle, [0, 15, 30, 45, 60, 75, 90.0001], right=False)
    paired = observation.pivot(index=['manifest_row', 'port', 'draw', 'class_id', 'relative_angle'], columns='mode', values='prediction')
    flips = float((paired.rotated90 != paired.reextracted0).mean())
    by_angle = observation.groupby(['manifest_row', 'class_id', 'angle_bin', 'mode'], observed=True).correct.mean().reset_index().groupby(
        ['class_id', 'angle_bin', 'mode'], observed=True).agg(n=('manifest_row', 'nunique'), oa=('correct', 'mean'))
    by_angle.to_csv(OUT/'class_angle_results.csv')
    result = dict(ports=len(verification), observations=len(selected),
        paired_draw_prediction_flip_fraction=flips, scope='Frozen BASE ONLY, same original64support fits; not complete routed model',
        interpretation='Pixel-array display rotation preserves acquired scattering content; changes include encoder directional sensitivity, not physical look-angle change',
        caveat='Rotation perturbation shows computational sensitivity, not how much of the natural angle gap is caused by it')
    (OUT/'COMPLETE.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(summary.to_string(), flush=True)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['extract', 'evaluate'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (extract if args.stage == 'extract' else evaluate)()
