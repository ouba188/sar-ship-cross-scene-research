"""Fixed C4 base / class-mix transfer on existing matched SAR protocols."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
WS = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline')
SOURCE = WS/'artifacts/eight_class_adaptive_20260916'
OUT = ROOT/'cross_dataset_c4_20261009'
DATASETS = {
    'oss1': ('opensarship1_transfer_20261007', 'opensarship1_transfer_20261007',
             'module_completion_20261008/oss1/joint'),
    'oss2': ('opensarship2_transfer_20261007', 'opensarship2_transfer_20261007',
             'module_completion_20261008/oss2/joint'),
    'fusar': ('fusar_transfer_20261007', 'fusar_transfer_20261007', None),
    'f0921': ('f0921_detector_recrop_20261007', 'f0921_transfer_20261007',
              'joint_residual_pilot_20261008/joint'),
}
ARMS = ('reference', 'reextracted_zero', 'c4_query_only', 'c4_matched', 'fixed_class_mix')
CLASSES = ('bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
           'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'PROTOCOL.json').exists():
        return
    files = sorted((ROOT/'directional_fusion_trial_20261009/nested_geometry_multistart/ports').glob('*/fusion_models.json'))
    if len(files) != 23:
        raise ValueError('Expected all23 corrected source-fit fusion models')
    alpha = np.mean([json.loads(p.read_text())['class_mix']['alpha'] for p in files], axis=0)
    write_json(OUT/'PROTOCOL.json', dict(
        datasets=list(DATASETS), arms=ARMS, classes=CLASSES, alpha=alpha.tolist(),
        alpha_rule='Equal mean of all23 existing nested-source class_mix vectors; fixed before external readout',
        alpha_sources=[str(p) for p in files], source_ridge=1., support_ridge=8.,
        source_base='Dataset-specific purged source rows; frozen original PCA/scales; refit C4 head only',
        target='Existing full mapped cohorts; same five draws of64 global labels; 5% handoff cap',
        expert='Exact saved joint expert scores for OSS1/OSS2/F0921; original expert for FUSAR',
        scope='Transfer of C4 and fixed fusion intervention, NOT full recent24port Qs/Qd/incidence architecture',
        alpha_provenance='Previously developed24port rule; not newly fitted on dataset-purged source. Historical exposure/identity overlap not erased.',
        historical_development=True, independent_confirmation=False, target_tuning=False,
        cost='Four R50 passes for every query; 5% limits handoff count only',
        invariance='Image C4 averaging, not physical viewing-angle or causal invariance',
        fusar_input='One physical image duplicated into two channels; not VV/VH',
        f0921_encoder='Inherited expert SSL pretrained on F0921 corpus; not independent encoder validation'))


def archived_features(name):
    directory = ROOT/DATASETS[name][0]
    if name == 'f0921':
        return np.load(directory/'paired_r50.npy', mmap_mode='r')
    with np.load(directory/'features.npz') as saved:
        return saved['paired_r50']


def extract(name):
    import torch
    sys.path.insert(0, str(WS))
    from extract_full_six_spatial_features import build_encoder, read_one
    directory = ROOT/DATASETS[name][0]
    out = OUT/name
    out.mkdir(parents=True, exist_ok=True)
    if (out/'EXTRACTED.json').exists():
        print('REUSE EXTRACTION '+name, flush=True)
        return
    frame = pd.read_csv(directory/'target_manifest.csv', keep_default_na=False)
    if name == 'f0921':
        pixels = np.load(directory/'inputs128.npy', mmap_mode='r')
        np.testing.assert_array_equal(np.load(directory/'feature_rows.npy'), frame['row'].to_numpy())
        if len(pixels) != len(frame):
            raise ValueError('F0921 input order/size mismatch')
        def read(i):
            return pixels[i]
    else:
        if name == 'fusar':
            paths = [(p, p) for p in frame.image_path]
        else:
            paths = list(frame[['chip_path_vv', 'chip_path_vh']].itertuples(index=False, name=None))
        def read(i):
            return np.stack([read_one(Path(p)) for p in paths[i]])
    n = len(frame)
    zero = np.lib.format.open_memmap(out/'pooled_zero.npy', mode='w+', dtype='float32', shape=(n, 2048))
    average = np.lib.format.open_memmap(out/'pooled_c4.npy', mode='w+', dtype='float32', shape=(n, 2048))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    encoder = build_encoder(device)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for lo in range(0, n, 32):
            hi = min(n, lo+32)
            x = torch.from_numpy(np.stack(list(pool.map(read, range(lo, hi))))).to(device).float()
            x = (x/255.-.5)/.25
            orbit = []
            for k in range(4):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    spatial = encoder(torch.rot90(x, k, dims=(-2, -1))).flatten(2).transpose(1, 2)
                orbit.append(spatial.float().cpu().numpy().astype(np.float16).astype(np.float32).mean(1))
            zero[lo:hi], average[lo:hi] = orbit[0], np.mean(orbit, axis=0)
            if lo % 8192 == 0 or hi == n:
                zero.flush()
                average.flush()
                print(f'EXTRACT {name} {hi}/{n} {time.perf_counter()-started:.1f}s', flush=True)
    old = archived_features(name)
    cosine, rel = [], []
    for lo in range(0, n, 2048):
        a, b = np.asarray(old[lo:lo+2048], dtype=float), np.asarray(zero[lo:lo+2048], dtype=float)
        cosine.extend(np.sum(a*b, 1)/np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-12))
        rel.extend(np.linalg.norm(a-b, axis=1)/np.maximum(np.linalg.norm(a, axis=1), 1e-12))
    replay = dict(rows=n, cosine_min=float(np.min(cosine)), cosine_median=float(np.median(cosine)),
                  relative_l2_max=float(np.max(rel)), seconds=time.perf_counter()-started,
                  four_passes=True, original_pixels_unchanged=True)
    write_json(out/'encoder_replay.json', replay)
    if min(cosine) < .995:
        raise ValueError(f'{name}: original encoder replay failed; inspect before readout')
    np.save(out/'sample_ids.npy', frame.sample_id.to_numpy(str))
    write_json(out/'EXTRACTED.json', replay)
    print('EXTRACTED '+name+' '+json.dumps(replay), flush=True)


def base_coordinates(features, model):
    xs = np.empty((len(features), 128), dtype=np.float32)
    for lo in range(0, len(features), 2048):
        xs[lo:lo+2048] = ((model['pca'].transform(features[lo:lo+2048])-model['center'])/model['scale']).astype(np.float32)
    return xs


def base_phi(xs, model):
    normalized = (xs-model['residual_center'])/model['residual_scale']
    return np.column_stack((normalized/np.sqrt(128), np.ones(len(xs))))


def route(base, expert, gate, support_labels, ev, row_ids):
    from receiver_consistent_handoff import gate_scores
    from budgeted_residual_handoff import ranked_mask
    b, e = base.argmax(1), expert.argmax(1)
    score = gate_scores(base, expert, gate)
    observed = np.bincount(support_labels, minlength=base.shape[1]) > 0
    eligible = ev & (b != e) & observed[e] & (score > 0)
    selected = ranked_mask(score, eligible, row_ids, int(np.floor(.05*ev.sum())))
    return np.where(selected, e, b), selected


def evaluate(name):
    from sklearn.linear_model import RidgeClassifier
    from residual_expert import fit_correction
    out = OUT/name
    if (out/'FROZEN.json').exists():
        print('REUSE POLICIES '+name, flush=True)
        return
    if not (out/'EXTRACTED.json').exists():
        raise ValueError('Extraction must complete first')
    target_dir, source_dir, joint_dir = DATASETS[name]
    frame = pd.read_csv(ROOT/target_dir/'target_manifest.csv', keep_default_na=False)
    ids, y = frame.target_manifest_row.to_numpy(int), frame.class_id.to_numpy(int)
    model_path = ROOT/source_dir/'source_model.pkl'
    policy_path = ROOT/target_dir/'policies.npz'
    if name == 'f0921':
        # The joint run inherited a refitted gate, not the earlier transfer gate.
        model_path = ROOT/'joint_residual_pilot_20261008/source_model.pkl'
        policy_path = ROOT/'neural_complement_pilot_20261007/frozen_replay/policies.npz'
    with model_path.open('rb') as f:
        model = pickle.load(f)
    with np.load(policy_path) as saved:
        old = {k: saved[k] for k in ('sample_ids', 'support_rows', 'eval_masks', 'base_scores', 'expert_scores', 'predictions', 'selected')}
    np.testing.assert_array_equal(old['sample_ids'], frame.sample_id.to_numpy(str))
    np.testing.assert_array_equal(np.load(out/'sample_ids.npy'), old['sample_ids'])
    if joint_dir:
        with np.load(ROOT/joint_dir/'policies.npz') as saved:
            expert = saved['expert_scores']
            ref_pred, ref_selected = saved['predictions'], saved['selected']
            np.testing.assert_array_equal(saved['support_rows'], old['support_rows'])
            np.testing.assert_array_equal(saved['eval_mask'], old['eval_masks'])
            np.testing.assert_array_equal(saved['sample_ids'], old['sample_ids'])
    else:
        expert, ref_pred, ref_selected = old['expert_scores'], old['predictions'][:, 2], old['selected']
    with np.load(ROOT/source_dir/'source_split.npz') as saved:
        sr = saved['source_rows']
    source_y = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy(int)
    c4_source = np.load(ROOT/'c4_base_trial_20261009/pooled_c4.npy', mmap_mode='r')
    # Same source PCA/scales; only the base head sees rotation-averaged features.
    sx = base_coordinates(c4_source[sr], model)
    head = RidgeClassifier(alpha=1., class_weight='balanced').fit(sx, source_y[sr])
    np.testing.assert_array_equal(head.classes_, np.arange(8))
    with (out/'matched_head.pkl').open('wb') as f:
        pickle.dump(head, f)
    del sx, c4_source
    xs = [base_coordinates(archived_features(name), model)]
    xs.extend(base_coordinates(np.load(out/f'pooled_{key}.npy', mmap_mode='r'), model) for key in ('zero', 'c4'))
    phi = [base_phi(x, model) for x in xs]
    prior = [model['base'].decision_function(x).astype(np.float32).astype(float) for x in xs]
    matched = head.decision_function(xs[2]).astype(np.float32).astype(float)
    alpha = np.asarray(json.loads((OUT/'PROTOCOL.json').read_text())['alpha'])
    base_pred, full_pred, selections, changes = [], [], [], []
    replay_error = 0.
    for draw, support in enumerate(old['support_rows']):
        ev, sy = old['eval_masks'][draw], y[support]
        if len(support) != 64 or len(np.unique(support)) != 64 or ev[support].any():
            raise ValueError('Support/query protocol mismatch')
        coef = fit_correction(phi[0][support], sy, prior[0][support])
        replay = prior[0]+phi[0]@coef
        replay_error = max(replay_error, float(np.max(np.abs(replay-old['base_scores'][draw]))))
        c4coef = fit_correction(phi[2][support], sy, matched[support])
        c4 = matched+phi[2]@c4coef
        original = old['base_scores'][draw]
        scores = [original, prior[1]+phi[1]@coef, prior[2]+phi[2]@coef,
                  c4, c4+alpha*(original-c4)]
        bp, fp, masks = [], [], []
        for ai, score in enumerate(scores):
            prediction, selected = route(score, expert[draw], model['gate'], sy, ev, ids)
            if selected[~ev].any() or selected.sum() > int(.05*ev.sum()):
                raise ValueError('Handoff cap/support violation')
            if ai == 0:
                np.testing.assert_array_equal(prediction, ref_pred[draw])
                np.testing.assert_array_equal(selected, ref_selected[draw])
            bp.append(score.argmax(1))
            fp.append(prediction)
            masks.append(selected)
        base_pred.append(bp)
        full_pred.append(fp)
        selections.append(masks)
        changes.append(dict(draw=draw, zero_base_changed=int(np.sum((bp[1] != bp[0]) & ev)),
                            zero_full_changed=int(np.sum((fp[1] != fp[0]) & ev))))
        print(f'FROZEN {name} draw{draw}', flush=True)
    if replay_error > 1e-4:
        raise ValueError(f'Existing base replay failed: {replay_error}')
    np.savez_compressed(out/'policies.npz', arms=np.asarray(ARMS), base_predictions=np.asarray(base_pred),
        predictions=np.asarray(full_pred), selected=np.asarray(selections), support_rows=old['support_rows'],
        eval_masks=old['eval_masks'], sample_ids=old['sample_ids'], source_rows=sr)
    write_json(out/'FROZEN.json', dict(rows=len(frame), source_rows=len(sr), replay_error=replay_error,
        model_path=str(model_path), base_policy_path=str(policy_path),
        old_full_predictions_and_routing_exact=True, zero_changes=changes,
        query_labels_used_for_training=0, target_labels_per_draw=64))


def report():
    from budgeted_residual_handoff import score_policy
    metrics, classes = [], []
    for name, (directory, _, _) in DATASETS.items():
        out = OUT/name
        if not (out/'FROZEN.json').exists():
            continue
        frame = pd.read_csv(ROOT/directory/'target_manifest.csv', keep_default_na=False)
        y = frame.class_id.to_numpy(int)
        with np.load(out/'policies.npz') as saved:
            for d, ev in enumerate(saved['eval_masks']):
                for ai, arm in enumerate(ARMS):
                    for branch, key in [('base', 'base_predictions'), ('full', 'predictions')]:
                        p, ref = saved[key][d, ai], saved[key][d, 0]
                        m = score_policy(p, ref, y, ev)
                        metrics.append(dict(dataset=name, draw=d, arm=arm, branch=branch,
                            oa=100*m['acc'], ba=100*m['ba'], recover_ba_pp=100*m['recover_ba'],
                            harm_ba_pp=100*m['harm_ba'], evaluated=int(ev.sum()),
                            handoffs=int(saved['selected'][d, ai].sum()) if branch == 'full' else 0))
                        for c, recall in enumerate(m['class_recall']):
                            if recall is not None:
                                mask = ev & (y == c)
                                classes.append(dict(dataset=name, draw=d, arm=arm, branch=branch,
                                    class_id=c, class_name=CLASSES[c], n=int(mask.sum()), recall=100*recall,
                                    recovered=int(np.sum(mask & (p == y) & (ref != y))),
                                    harmed=int(np.sum(mask & (p != y) & (ref == y)))))
    df = pd.DataFrame(metrics)
    df.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_metrics.csv', index=False)
    summary = df.groupby(['dataset', 'branch', 'arm'], sort=False)[['oa', 'ba', 'recover_ba_pp', 'harm_ba_pp', 'handoffs']].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'extract', 'evaluate', 'report'), required=True)
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    prepare()
    if args.stage in ('extract', 'evaluate'):
        for dataset in args.datasets:
            (extract if args.stage == 'extract' else evaluate)(dataset)
    elif args.stage == 'report':
        report()
