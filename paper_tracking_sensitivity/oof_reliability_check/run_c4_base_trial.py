"""C4 pooled R50: query perturbation versus matched head/support adaptation."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

from probe_display_rotation import ROOT, SOURCE, GEOMETRY, PARENT, REFERENCE, base_phi, replace_r50

OUT = ROOT/'c4_base_trial_20261009'
BASE_MODES = ('archived', 'reextracted0', 'query_c4', 'support_c4', 'matched_c4')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def extract():
    import torch
    sys.path.insert(0, str(SOURCE.parent.parent))
    from extract_full_six_spatial_features import build_encoder, read_one
    if (OUT/'EXTRACTED.json').exists():
        raise ValueError('Preserve completed extraction')
    frame = pd.read_csv(SOURCE/'manifest.csv', low_memory=False)
    selected = np.load(GEOMETRY/'display_rotation/rows.npy')
    n = len(frame)
    zero = np.lib.format.open_memmap(OUT/'pooled_zero.npy', mode='w+', dtype='float32', shape=(n, 2048))
    average = np.lib.format.open_memmap(OUT/'pooled_c4.npy', mode='w+', dtype='float32', shape=(n, 2048))
    orbit = np.empty((len(selected), 4, 2048), dtype='float32')
    torch.set_num_threads(2)
    device = torch.device('cuda')
    encoder = build_encoder(device)
    seconds = np.zeros(4)
    started = time.perf_counter()
    def read(row):
        return np.stack([read_one(frame.loc[row, 'image_relpath_'+pol]) for pol in ('vv', 'vh')])
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for lo in range(0, n, 32):
            end = min(n, lo+32)
            batch = np.stack(list(pool.map(read, range(lo, end))))
            tensor = (torch.from_numpy(batch).to(device=device, dtype=torch.float32)/255.-.5)/.25
            local = []
            for k in range(4):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    values = encoder(torch.rot90(tensor, k, dims=(-2, -1))).flatten(2).transpose(1, 2)
                pooled = values.float().cpu().numpy().astype(np.float16).astype(np.float32).mean(1)
                seconds[k] += time.perf_counter()-tick
                local.append(pooled)
            local = np.stack(local, axis=1)
            zero[lo:end] = local[:, 0]
            average[lo:end] = local.mean(1)
            inside = np.flatnonzero((selected >= lo) & (selected < end))
            orbit[inside] = local[selected[inside]-lo]
            if lo % 1024 == 0 or end == n:
                zero.flush(); average.flush()
                progress = dict(completed=end, total=n, elapsed_seconds=time.perf_counter()-started)
                write_json(OUT/'progress.json', progress)
                print(json.dumps(progress), flush=True)
    zero.flush(); average.flush()
    np.save(OUT/'measurable_orbit.npy', orbit)
    np.save(OUT/'measurable_rows.npy', selected)
    archived = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    cosines = []
    for lo in range(0, n, 1024):
        a, b = np.asarray(archived[lo:lo+1024], dtype=float), np.asarray(zero[lo:lo+1024], dtype=float)
        cosines.extend(np.sum(a*b, axis=1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)))
    report = dict(rows=n, encoder='Original frozen paired ImageNetV2 R50',
        rotations=[0, 90, 180, 270], preprocessing='Original128bilinear, pairedVVVH, bfloat16, spatialfloat16pool',
        archived_cosine_min=float(min(cosines)), archived_cosine_median=float(np.median(cosines)),
        forward_and_transfer_seconds=seconds.tolist(), total_seconds=time.perf_counter()-started,
        encoder_pass_ratio=4, physical_metadata_unchanged=True)
    write_json(OUT/'EXTRACTED.json', report)
    if min(cosines) < .995:
        raise ValueError('Original encoder parity failed')
    print(json.dumps(report), flush=True)


def evaluate():
    from direct_incidence_conditioning import condition_features, extend_transforms
    from residual_expert import fit_correction
    from run_conditioned_vessel_drift import SIDECAR
    from run_protected_drift_activation import make_transforms, gate_features
    from protected_drift_activation import fit_source_heads
    from evidence_aware_handoff import predict_gate, route_scores
    if (OUT/'FROZEN.json').exists():
        raise ValueError('Preserve completed predictions')
    parity = json.loads((OUT/'EXTRACTED.json').read_text())
    if parity['archived_cosine_min'] < .995:
        raise ValueError('Encoder parity failed')
    write_json(OUT/'PROTOCOL.json', dict(
        base_modes=BASE_MODES, target_labels=64, draws=5, source_ridge=1, target_ridge=8,
        fixed='Original outer-source rows, geometry/PCA/Qs/Qd/scales, incidence, frozen R50 weights, support sets',
        query_c4='Only query features averaged, original source head and original support correction',
        support_c4='Average query and support, original source head',
        matched_c4='Average source/query/support; refit source base head and target correction; geometry remains original',
        full='Replace base only; original incidence expert and frozen gate; 5% handoff recomputed',
        independent_confirmation=False, parameter_search=False, main_model_unchanged=True,
        invariance='C4 BASE only; neither arbitrary-angle nor physical-view invariance; hybrid expert remains directional',
        compute='Four R50 passes on every query. Existing5% budget is handoff fraction, not total compute'))
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'product', 'port', 'class_id'])
    side = pd.read_csv(SIDECAR, usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    frame = frame.merge(side, on=['instance_id', 'product'], how='left', sort=False, validate='one_to_one')
    frame.loc[frame.status.ne('ok'), 'incidence_deg'] = np.nan
    zero = np.load(OUT/'pooled_zero.npy', mmap_mode='r')
    average = np.load(OUT/'pooled_c4.npy', mmap_mode='r')
    checks = []
    for folder in sorted(PARENT.iterdir()):
        if not (folder/'policies.npz').exists():
            continue
        port = folder.name
        output = OUT/'ports'/port
        output.mkdir(parents=True, exist_ok=True)
        with (folder/'geometry.pkl').open('rb') as f:
            g = pickle.load(f)
        with (folder/'source_models.pkl').open('rb') as f:
            gate = pickle.load(f)['gates']['joint']
        with (REFERENCE/port/'direct_heads.pkl').open('rb') as f:
            head = pickle.load(f)['joint_base']
        with np.load(REFERENCE/port/'policies.npz') as f:
            old = {k: f[k] for k in f.files}
        slot = list(old['mode_names']).index('direct_condition')
        sr, tr = np.load(folder/'source_rows.npy'), old['query_rows']
        np.testing.assert_array_equal(np.load(folder/'target_rows.npy'), tr)
        bank = np.load(folder/'target_bank.npy', mmap_mode='r')
        u = condition_features(frame.loc[tr, 'incidence_deg'].to_numpy())
        su = condition_features(frame.loc[sr, 'incidence_deg'].to_numpy())
        banks = [np.asarray(bank), replace_r50(zero[tr].astype(float), bank, g), replace_r50(average[tr].astype(float), bank, g)]
        phis = [base_phi(b, g, u) for b in banks]
        priors = [np.c_[b, u] @ head['coef']+head['intercept'] for b in banks]
        source = replace_r50(average[sr].astype(float), np.load(folder/'source_bank.npy', mmap_mode='r'), g)
        transforms = extend_transforms({'joint_base': make_transforms(g)['joint_base']})
        matched_head = fit_source_heads(np.c_[source, su], frame.loc[sr, 'class_id'].to_numpy(), transforms)['joint_base']
        with (output/'matched_head.pkl').open('wb') as f:
            pickle.dump(matched_head, f)
        matched_prior = np.c_[banks[2], u] @ matched_head['coef']+matched_head['intercept']
        bases, fulls, selections = [], [], []
        replay = 0.
        for draw, ev in enumerate(old['eval_mask']):
            support = np.searchsorted(tr, old['support_rows'][draw])
            np.testing.assert_array_equal(tr[support], old['support_rows'][draw])
            if ev[support].any():
                raise ValueError('Support entered query set')
            labels = frame.loc[old['support_rows'][draw], 'class_id'].to_numpy()
            oldcoef = fit_correction(phis[0][support], labels, priors[0][support])
            replay = max(replay, float(np.max(np.abs(priors[0]+phis[0] @ oldcoef-old['base_scores'][draw, slot]))))
            supportcoef = fit_correction(phis[2][support], labels, priors[2][support])
            matchedcoef = fit_correction(phis[2][support], labels, matched_prior[support])
            scores = [old['base_scores'][draw, slot], priors[1]+phis[1] @ oldcoef,
                priors[2]+phis[2] @ oldcoef, priors[2]+phis[2] @ supportcoef,
                matched_prior+phis[2] @ matchedcoef]
            expert = old['expert_scores'][draw, slot]
            new = route_scores(scores[-1], expert, predict_gate(gate, gate_features(scores[-1], expert)), labels, ev, tr)
            if new['selected'][~ev].any() or new['selected'].sum() > int(.05*ev.sum()):
                raise ValueError('Handoff budget error')
            bases.append(np.stack(scores))
            fulls.append(np.stack([old['predictions'][draw, slot], new['predictions']]))
            selections.append(np.stack([old['selected'][draw, slot], new['selected']]))
        if replay > 3e-6:
            raise ValueError(f'Parent replay failed {port}: {replay}')
        np.savez_compressed(output/'policies.npz', query_rows=tr, support_rows=old['support_rows'],
            eval_mask=old['eval_mask'], base_modes=np.asarray(BASE_MODES), base_scores=np.stack(bases),
            full_modes=np.asarray(['incidence', 'matched_c4_base_original_expert']),
            full_predictions=np.stack(fulls), selected=np.stack(selections))
        checks.append(dict(port=port, source_n=len(sr), target_n=len(tr), parent_score_replay_error=replay))
        print(json.dumps(checks[-1]), flush=True)
    pd.DataFrame(checks).to_csv(OUT/'verification.csv', index=False)
    write_json(OUT/'FROZEN.json', dict(ports=len(checks), query_labels_used_in_fit=0))


def report():
    from run_generic_transfer import metrics
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'port', 'class_id'])
    meta = pd.read_csv(GEOMETRY/'relative_conditions.csv')
    np.testing.assert_array_equal(frame.instance_id, meta.instance_id)
    records, observations = [], []
    for folder in sorted((OUT/'ports').iterdir()):
        with np.load(folder/'policies.npz') as f:
            p = {k: f[k] for k in f.files}
        tr = p['query_rows']
        truth = frame.loc[tr, 'class_id'].to_numpy()
        angle = meta.loc[tr, 'relative_acute_deg'].to_numpy()
        for draw, ev in enumerate(p['eval_mask']):
            for stage, modes, predictions in [('base', p['base_modes'], p['base_scores'][draw].argmax(2)),
                    ('full', p['full_modes'], p['full_predictions'][draw])]:
                ref = predictions[0]
                for mode, pred in zip(modes, predictions):
                    for subset, mask in [('all', ev), ('measurable', ev & np.isfinite(angle))]:
                        if not mask.any():
                            continue
                        score = metrics(truth[mask], pred[mask], 8)
                        records.append(dict(port=folder.name, draw=draw, stage=stage, mode=mode, subset=subset,
                            n=int(mask.sum()), oa=score['acc'], ba=score['ba'],
                            rescues=int(((ref != truth) & (pred == truth) & mask).sum()),
                            harms=int(((ref == truth) & (pred != truth) & mask).sum())))
                    for j in np.flatnonzero(ev):
                        observations.append(dict(manifest_row=int(tr[j]), port=folder.name, draw=draw, stage=stage,
                            mode=mode, class_id=int(truth[j]), relative_angle=float(angle[j]), prediction=int(pred[j]),
                            correct=int(pred[j] == truth[j]), rescue=int(ref[j] != truth[j] and pred[j] == truth[j]),
                            harm=int(ref[j] == truth[j] and pred[j] != truth[j])))
    result, observation = pd.DataFrame(records), pd.DataFrame(observations)
    result.to_csv(OUT/'metrics.csv', index=False)
    observation.to_csv(OUT/'predictions.csv', index=False)
    pm = result.groupby(['port', 'stage', 'mode', 'subset']).mean(numeric_only=True).reset_index()
    pm.to_csv(OUT/'port_means.csv', index=False)
    full8 = frame.groupby('port').class_id.nunique()
    summaries = []
    for group, mask in [('all_ports', np.ones(len(pm), dtype=bool)), ('strict8_ports', pm.port.map(full8).eq(8))]:
        summary = pm[mask].groupby(['stage', 'mode', 'subset']).agg(ports=('port', 'nunique'), oa=('oa', 'mean'), ba=('ba', 'mean')).reset_index()
        summary.insert(0, 'group', group)
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT/'summary.csv', index=False)
    instances = observation.groupby(['manifest_row', 'port', 'class_id', 'stage', 'mode']).agg(
        correct=('correct', 'mean'), rescue=('rescue', 'mean'), harm=('harm', 'mean'),
        draws=('draw', 'size'), relative_angle=('relative_angle', 'first')).reset_index()
    instances.to_csv(OUT/'instance_effects.csv', index=False)
    for group, name in [(['port', 'class_id', 'stage', 'mode'], 'port_class'), (['class_id', 'stage', 'mode'], 'class')]:
        instances.groupby(group).agg(n=('manifest_row', 'nunique'), recall=('correct', 'mean'),
            rescue_fraction=('rescue', 'mean'), harm_fraction=('harm', 'mean')).to_csv(OUT/(name+'_effects.csv'))
        measured = instances[np.isfinite(instances.relative_angle)].copy()
        measured['angle_bin'] = pd.cut(measured.relative_angle, [0,15,30,45,60,75,90.0001], right=False)
        measured.groupby(group+['angle_bin'], observed=True).agg(n=('manifest_row', 'nunique'),
            recall=('correct', 'mean'), rescue_fraction=('rescue', 'mean'), harm_fraction=('harm', 'mean')).to_csv(OUT/(name+'_angle.csv'))
    write_json(OUT/'COMPLETE.json', dict(ports=int(result.port.nunique()), same_draws=5,
        scope='Matched-base experiment with original expert/gate hybrid, not retrained invariant full model',
        no_target_outcome_selection=True))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['extract', 'evaluate', 'report'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {'extract': extract, 'evaluate': evaluate, 'report': report}[args.stage]()
