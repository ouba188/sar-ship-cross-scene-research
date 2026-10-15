"""Isolated real-image cost benchmark of frozen OSS2 joint deployment.

Does not change or train any parent experiment. Head equations mirror
run_opensarship2_transfer.target_geometry and evaluate_joint_residual.views;
cached-score replay is checked before any timing is accepted.
"""

import os

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import json
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from threadpoolctl import threadpool_limits
from torchvision import transforms
from torchvision.models import resnet18

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = Path(__file__).resolve().parent
CODE = ROOT/'oof_reliability_check'
WS = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline')
TARGET = ROOT/'opensarship2_transfer_20261007'
JOINT = ROOT/'module_completion_20261008/oss2'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(WS))
from budgeted_residual_handoff import ranked_mask
from receiver_consistent_handoff import gate_scores
from residual_expert import fit_correction
from extract_full_six_spatial_features import build_encoder, read_one

MODES = ('base64', 'all_expert', 'gate_005', 'gate_100')
SAMPLE_SIZE, BATCH_SIZE, WARMUPS, REPEATS, SEED = 256, 64, 2, 7, 20261008


def write_json(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def gpu_status():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,temperature.gpu,pstate',
        '--format=csv'], text=True).strip()


class Heads:
    def __init__(self, model, stats, support_base, support_extra, base_coef, beta, observed):
        self.m, self.stats = model, stats
        self.base_coef, self.beta, self.observed = base_coef, beta, observed
        _, self.support_retained, self.support_discarded = self.expert_views(
            self.base_coordinates(support_base), support_base, support_extra)

    def base_coordinates(self, base):
        m = self.m
        return ((m['pca'].transform(base)-m['center'])/m['scale']).astype(np.float32)

    def base_geometry(self, xs):
        m = self.m
        prior = m['base'].decision_function(xs).astype(np.float32).astype(float)
        xb = (xs-m['residual_center'])/m['residual_scale']
        phi = np.column_stack((xb/np.sqrt(128), np.ones(len(xs))))
        return prior, phi

    def expert_views(self, xs, base, extra):
        m = self.m
        joint = np.column_stack((xs, m['pca_ssl'].transform(extra).astype(np.float32)))
        xe = (joint-m['expert_center'])/m['expert_scale']
        prior = m['expert'].decision_function(xe).astype(np.float32).astype(float)
        discarded = []
        for x, pca in ((base, m['pca']), (extra, m['pca_ssl'])):
            centered = np.asarray(x, dtype=float)-pca.mean_
            components = np.asarray(pca.components_, dtype=float)
            discarded.append(centered-(centered@components.T)@components)
        raw = np.column_stack(discarded)
        residual = ((raw-self.stats['discarded_center'])/
                    self.stats['discarded_scale']/np.sqrt(raw.shape[1]))
        return prior, xe/np.sqrt(192), residual

    def scores(self, base, extra, mode):
        xs = self.base_coordinates(base)
        b = e = None
        if mode != 'all_expert':
            prior, phi = self.base_geometry(xs)
            b = prior+phi@self.base_coef
        if mode != 'base64':
            prior, retained, discarded = self.expert_views(xs, base, extra)
            kernel = (.5*(retained@self.support_retained.T)+
                      .5*(discarded@self.support_discarded.T)+1.)
            e = prior+kernel@self.beta
        return b, e

    def predict(self, b, e, row_ids, mode):
        if mode == 'base64':
            return b.argmax(1), 0
        if mode == 'all_expert':
            return e.argmax(1), len(e)
        bp, ep = b.argmax(1), e.argmax(1)
        score = gate_scores(b, e, self.m['gate'])
        eligible = (bp != ep) & self.observed[ep] & (score > 0)
        cap = int(np.floor(.05*len(bp))) if mode == 'gate_005' else len(bp)
        selected = ranked_mask(score, eligible, row_ids, cap)
        return np.where(selected, ep, bp), int(selected.sum())


def main():
    if (OUT/'COMPLETE.json').exists():
        raise RuntimeError('Preserve completed benchmark; choose a new output directory to rerun.')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    initial_gpu = gpu_status()
    print(initial_gpu, flush=True)
    frame = pd.read_csv(TARGET/'target_manifest.csv', keep_default_na=False)
    with (JOINT/'source_model.pkl').open('rb') as handle:
        model = pickle.load(handle)
    with np.load(TARGET/'policies.npz') as saved:
        support = saved['support_rows'][0]
        base_coef = saved['base_correction'][0]
        observed = saved['observed_classes'][0]
        ref_base_scores = saved['base_scores'][0]
    with np.load(JOINT/'joint/policies.npz') as saved:
        np.testing.assert_array_equal(saved['support_rows'][0], support)
        beta = saved['coefficients'][0]
        ref_expert_scores = saved['expert_scores'][0]
        query = np.flatnonzero(saved['eval_mask'][0])
    rows = np.sort(np.random.default_rng(SEED).choice(query, SAMPLE_SIZE, replace=False))
    row_ids = frame.iloc[rows].target_manifest_row.to_numpy()
    with np.load(TARGET/'features.npz') as saved:
        cached_base = saved['paired_r50']
    cached_extra = np.load(JOINT/'joint/target_extra.npy')
    with np.load(JOINT/'deployment_complement.npz') as saved:
        stats = {k: saved[k] for k in ('discarded_center', 'discarded_scale')}
    heads = Heads(model, stats, cached_base[support], cached_extra[support], base_coef, beta, observed)
    b, e = heads.scores(cached_base[rows], cached_extra[rows], 'gate_005')
    np.testing.assert_allclose(b, ref_base_scores[rows], rtol=0, atol=3e-5)
    np.testing.assert_allclose(e, ref_expert_scores[rows], rtol=0, atol=3e-5)
    validation = dict(cached_base_score_max_abs=float(np.max(np.abs(b-ref_base_scores[rows]))),
                      cached_expert_score_max_abs=float(np.max(np.abs(e-ref_expert_scores[rows]))))
    protocol = dict(dataset='OpenSARShip 2.0', scope='fixed 256-query timing subset, not full-dataset timing',
        support_draw=0, support_labels=64, sampling_seed=SEED, sample_size=SAMPLE_SIZE,
        sample_local_rows=rows.tolist(), support_local_rows=support.tolist(), batch_size=BATCH_SIZE,
        warmups_per_mode=WARMUPS, repeats_per_mode=REPEATS, modes=MODES,
        mode_order='seeded shuffled complete blocks', timing='wall clock with CUDA synchronization',
        includes=['warm-cache file read', 'decode', 'preprocessing', 'H2D', 'R50',
                  'R18 for expert modes', 'D2H', 'CPU geometry and fitted heads', 'routing'],
        excludes=['checkpoint loading', 'support feature extraction', 'support fitting', 'disk result writes'],
        support_fit_timing='reported separately; cached support features only',
        gate_budget_cohort='256 sampled queries; full-dataset OA/BA is copied separately, not inferred',
        shared_R50_computed_once=True, R18_skipped_for_base64=True,
        R18_computed_for_every_query_in_all_other_modes=True,
        all_expert='R50 + R18 + expert head only; base head and gate omitted',
        base_precision='bf16 forward; spatial float16 rounding then float32 mean',
        expert_precision='float32 prefix and joint layer4; Resize(128) aspect preserved',
        cpu_threads=2, image_workers=4, tf32=False, gpu_initial=initial_gpu,
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        model_directory=str(JOINT), source_code=str(Path(__file__)), independent_confirmation=False)
    write_json('PROTOCOL.json', protocol)
    print('HEAD REPLAY PASSED', validation, flush=True)

    support_labels = frame.iloc[support].class_id.to_numpy(int)
    adaptation = []
    for i in range(WARMUPS+REPEATS):
        start = time.perf_counter()
        xs = heads.base_coordinates(cached_base[support])
        pb, phib = heads.base_geometry(xs)
        pe, retained, discarded = heads.expert_views(xs, cached_base[support], cached_extra[support])
        fitted_base = fit_correction(phib, support_labels, pb)
        kernel = .5*(retained@retained.T)+.5*(discarded@discarded.T)+1.
        _, inverse, counts = np.unique(support_labels, return_inverse=True, return_counts=True)
        root = np.sqrt(len(support)/(len(counts)*counts[inverse]))
        desired = 2*np.eye(pe.shape[1])[support_labels]-1.
        fitted_beta = root[:, None]*np.linalg.solve(
            root[:, None]*kernel*root[None, :]+8.*np.eye(len(support)),
            root[:, None]*(desired-pe))
        elapsed = time.perf_counter()-start
        if i >= WARMUPS:
            adaptation.append(elapsed)
    np.testing.assert_allclose(fitted_base, base_coef, atol=3e-5, rtol=0)
    np.testing.assert_allclose(fitted_beta, beta, atol=3e-5, rtol=0)
    write_json('support_fit.json', dict(seconds=adaptation, median_seconds=float(np.median(adaptation)),
        scope='geometry plus Base64 and expert residual fit from cached 64 support features; no encoders'))

    paired = build_encoder(torch.device('cuda'))
    net = resnet18(weights=None)
    checkpoint = torch.load(WS/'e206_out/ssl_resnet18.pt', map_location='cpu', weights_only=True)
    net.load_state_dict(checkpoint['backbone'], strict=False)
    net.layer4.load_state_dict(torch.load(JOINT/'joint/layer4.pt', map_location='cpu', weights_only=True))
    prefix = torch.nn.Sequential(*list(net.children())[:-3]).eval()
    tail = net.layer4.eval()
    del checkpoint, net
    transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
    paths = list(zip(frame.iloc[rows].chip_path_vv, frame.iloc[rows].chip_path_vh))

    def read_pair(pair, expert):
        vv, vh = pair
        base = np.stack((read_one(Path(vv)), read_one(Path(vh))))
        if not expert:
            return base, None
        with Image.open(vh) as im:
            extra = transform(im.convert('RGB'))
        return base, extra

    def extract(pool, expert):
        bases, extras = [], []
        for lo in range(0, len(paths), BATCH_SIZE):
            batch = list(pool.map(lambda pair: read_pair(pair, expert), paths[lo:lo+BATCH_SIZE]))
            pixels = torch.from_numpy(np.stack([x[0] for x in batch])).cuda().float()
            pixels = (pixels/255.-.5)/.25
            with torch.autocast('cuda', dtype=torch.bfloat16):
                spatial = paired(pixels).flatten(2).transpose(1, 2)
            bases.append(spatial.to(torch.float16).cpu().numpy().astype(np.float32).mean(1))
            del pixels, spatial
            if expert:
                grouped = {}
                result = np.empty((len(batch), 512), dtype=np.float32)
                for i, (_, tensor) in enumerate(batch):
                    grouped.setdefault(tuple(tensor.shape), []).append(i)
                for indices in grouped.values():
                    pixels = torch.stack([batch[i][1] for i in indices]).cuda()
                    result[indices] = tail(prefix(pixels)).mean((2, 3)).cpu().numpy()
                    del pixels
                extras.append(result)
        return np.concatenate(bases), np.concatenate(extras) if expert else None

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        prefix.cuda()
        tail.cuda()
        actual_base, actual_extra = extract(pool, True)
        for name, actual, cached in (('base', actual_base, cached_base[rows]),
                                     ('expert', actual_extra, cached_extra[rows])):
            rel = np.linalg.norm(actual-cached, axis=1)/np.maximum(np.linalg.norm(cached, axis=1), 1e-12)
            validation[name+'_feature_max_relative_l2'] = float(rel.max())
        actual_b, actual_e = heads.scores(actual_base, actual_extra, 'gate_005')
        validation['base_prediction_agreement'] = float(np.mean(actual_b.argmax(1) == ref_base_scores[rows].argmax(1)))
        validation['expert_prediction_agreement'] = float(np.mean(actual_e.argmax(1) == ref_expert_scores[rows].argmax(1)))
        write_json('VALIDATION.json', validation)
        if (validation['base_feature_max_relative_l2'] > .01 or
                validation['expert_feature_max_relative_l2'] > 1e-4 or
                min(validation['base_prediction_agreement'], validation['expert_prediction_agreement']) < .99):
            raise RuntimeError('Real-image replay differs materially; do not accept runtime results.')
        print('REAL-IMAGE REPLAY PASSED', validation, flush=True)
        rng = np.random.default_rng(SEED)
        for block in range(WARMUPS+REPEATS):
            for mode in rng.permutation(MODES):
                expert = mode != 'base64'
                prefix.to('cuda' if expert else 'cpu')
                tail.to('cuda' if expert else 'cpu')
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                resident = torch.cuda.memory_allocated()
                started = time.perf_counter()
                base, extra = extract(pool, expert)
                extracted = time.perf_counter()
                b, e = heads.scores(base, extra, mode)
                scored = time.perf_counter()
                prediction, selected = heads.predict(b, e, row_ids, mode)
                torch.cuda.synchronize()
                ended = time.perf_counter()
                if block >= WARMUPS:
                    record = dict(mode=mode, repeat=block-WARMUPS, rows=len(rows),
                        end_to_end_seconds=ended-started, image_encoder_seconds=extracted-started,
                        head_seconds=scored-extracted, route_seconds=ended-scored,
                        milliseconds_per_image=1000*(ended-started)/len(rows),
                        images_per_second=len(rows)/(ended-started), selected=selected,
                        resident_gpu_mib=resident/2**20,
                        peak_allocated_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
                    records.append(record)
                    pd.DataFrame(records).to_csv(OUT/'timings.csv', index=False)
                    print(f'{mode} repeat {block-WARMUPS+1}/{REPEATS}: '
                          f'{record["milliseconds_per_image"]:.3f} ms/image', flush=True)
            print(f'BLOCK {block+1}/{WARMUPS+REPEATS} DONE', flush=True)
    table = pd.DataFrame(records)
    summary = []
    for mode in MODES:
        own = table[table['mode'] == mode]
        summary.append(dict(mode=mode, **{column: float(own[column].median()) for column in
            ('end_to_end_seconds', 'image_encoder_seconds', 'head_seconds', 'route_seconds',
             'milliseconds_per_image', 'images_per_second', 'resident_gpu_mib', 'peak_allocated_gpu_mib')},
            min_ms_per_image=float(own.milliseconds_per_image.min()),
            max_ms_per_image=float(own.milliseconds_per_image.max())))
    pd.DataFrame(summary).to_csv(OUT/'runtime_summary.csv', index=False)
    full_metrics = ROOT/'handoff_budget_sweep_20261008/dataset_means.csv'
    if (full_metrics.parent/'COMPLETE.json').exists():
        all_metrics = pd.read_csv(full_metrics)
        all_metrics[all_metrics.policy.isin(['gate_000', 'gate_005', 'gate_100', 'all_expert'])].to_csv(
            OUT/'existing_full_dataset_accuracy.csv', index=False)
    write_json('COMPLETE.json', dict(dataset='oss2', timed_queries=len(rows),
        repeats=REPEATS, modes=MODES, validation=validation, gpu_final=gpu_status(),
        full_accuracy_source=str(full_metrics), full_accuracy_is_new_experiment=False,
        limitation='Only OSS2 raw-image runtime measured; do not generalize milliseconds to other datasets.'))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
