"""Real-pixel replay and bounded deployment timing of the frozen SAR pipeline."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import pickle
import threading
import time

import numpy as np
import pandas as pd
from PIL import Image
import psutil
from threadpoolctl import threadpool_limits

from prepare_learned_frequency import ROOT, SOURCE, OUT as EXP, load_npz, load_pickle
from c4_condition_descriptors import describe, read_pair, MANIFESTS
from conditional_direction_residual import normalize_direction
from unified_head_core import coefficients, late

OUT = ROOT/'completion_e2e_20261010'
WS = SOURCE.parents[1]
PORT = 'Antwerp-Bruges'
KEYS = ('elongation_proxy', 'center_excess_proxy')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def rank_against_cohort(values, references):
    result = np.full((len(values), 2), .5)
    for col, reference in enumerate(references):
        x = values[:, col]
        good = np.isfinite(x)
        result[good, col] = (np.searchsorted(reference, x[good], side='left')
                            + np.searchsorted(reference, x[good], side='right'))/(2*len(reference))
    return result


def pixel_conditions(values, state, indices):
    # Preserve ties lost solely through the original descriptor CSV round-trip.
    saved = state['descriptors_reference'][indices]
    values = np.where(np.abs(values-saved) <= 1e-12, saved, values)
    return rank_against_cohort(values, state['rank_references'])


def geometry(state, raw, c4, extra, conditions, angles):
    def discarded(x, pca):
        centered = np.asarray(x, dtype=float)-pca.mean_
        pc = np.asarray(pca.components_, dtype=float)
        return centered-(centered@pc.T)@pc

    if state['dataset'] == 'self24ports':
        from probe_display_rotation import replace_r50, base_phi
        g = state['geometry']
        m = g['model']
        bank = np.zeros((len(raw), 192+g['stable_rank']+g['drift_rank']))
        bank[:, :128] = ((m['pca'].transform(raw)-m['center'])/m['scale']).astype(np.float32)
        bank[:, 128:192] = m['pca_ssl'].transform(extra).astype(np.float32)
        standardized = (raw.astype(float)-g['feature_center'])/g['feature_scale']
        r = g['stable_rank']
        bank[:, 192:192+r] = (standardized@g['qs']-g['stable_center'])/g['stable_scale']
        bank[:, 192+r:] = standardized@g['qd']
        base = replace_r50(c4.astype(float), bank, g)
        phi = base_phi(base, g, angles)
        mean = normalize_direction((base[:, :128]-g['base_center'])/g['base_scale'])
        pb = np.c_[base, angles]@state['heads']['joint_base']['coef']+state['heads']['joint_base']['intercept']
        pe = np.c_[bank, angles]@state['heads']['joint_expert']['coef']+state['heads']['joint_expert']['intercept']
        stats = g['stats']
        comp = np.c_[discarded(raw, m['pca']), discarded(extra, m['pca_ssl'])]
        comp = (comp-stats['discarded_center'])/stats['discarded_scale']
        zd = bank[:, 192+r:]
        comp[:, :2048] -= .5*(zd@g['drift_to_complement'])
        core = bank[:, :128]-.5*(zd@g['drift_to_xs'])
        xe = (np.c_[core, bank[:, 128:192].astype(np.float32)]-m['expert_center'])/m['expert_scale']
        expert = np.c_[np.sqrt(.5)*xe/np.sqrt(192), np.sqrt(.5)*comp/np.sqrt(2560),
                       np.ones(len(raw)), .5*bank[:, 192:192+r]/np.sqrt(r), np.sqrt(.125)*angles]
    else:
        from cross_dataset_c4 import base_coordinates, base_phi
        m = state['base_model']
        x = base_coordinates(c4, m)
        phi = base_phi(x, m)
        mean = normalize_direction((x-m['residual_center'])/m['residual_scale'])
        pb = m['base'].decision_function(x).astype(np.float32).astype(float)
        m, stats = state['expert_model'], state['stats']
        xs = ((m['pca'].transform(raw)-m['center'])/m['scale']).astype(np.float32)
        xe = (np.c_[xs, m['pca_ssl'].transform(extra).astype(np.float32)]-m['expert_center'])/m['expert_scale']
        pe = m['expert'].decision_function(xe).astype(np.float32).astype(float)
        comp = np.c_[discarded(raw, m['pca']), discarded(extra, m['pca_ssl'])]
        comp = (comp-stats['discarded_center'])/stats['discarded_scale']/np.sqrt(2560)
        expert = np.c_[np.sqrt(.5)*xe/np.sqrt(192), np.sqrt(.5)*comp, np.ones(len(raw))]
    return dict(phi=phi, mean=mean, conditions=conditions, prior=pb, expert_prior=pe, expert=expert)


def kernels(query, support, eta):
    relation = np.exp(-np.sum((query['conditions'][:, None]-support['conditions'][None])**2, axis=2)/.5)
    kb = query['phi']@support['phi'].T
    kb += (eta*(query['mean']@support['mean'].T)+.25*(query['embedding']@support['embedding'].T))*relation
    return kb, query['expert']@support['expert'].T


def fit(components, labels, eta, mode=None):
    kb, ke = kernels(components, components, eta)
    pb, pe = components['prior'], components['expert_prior']
    result = {}
    if mode in (None, 'late_softmax_fixed'):
        result.update(base=coefficients(kb, pb, labels, 8.), expert=coefficients(ke, pe, labels, 8.))
    if mode in (None, 'unified_fixed'):
        result['unified'] = coefficients(.5*(kb+ke), .5*(pb+pe), labels, 8.)
    return result


def predict(query, support, alpha, eta, mode):
    kb, ke = kernels(query, support, eta)
    pb, pe = query['prior'], query['expert_prior']
    if mode == 'late_softmax_fixed':
        return late(pb+kb@alpha['base'], pe+ke@alpha['expert'])
    return .5*(pb+pe)+.5*(kb+ke)@alpha['unified']


def prepare(dataset):
    from unified_head_expert import ExpertContext, _conditions
    task = PORT if dataset == 'self24ports' else dataset
    folder = EXP/'tasks'/dataset/task if dataset == 'self24ports' else EXP/'tasks'/dataset
    info = json.loads((folder/'PREPARED.json').read_text())
    data = load_npz(folder/'target.npz')
    available = np.flatnonzero(data['eval_masks'][0])
    query = available[np.linspace(0, len(available)-1, 16, dtype=int)]
    selected = np.r_[data['support'][0], query]
    rows = data['rows'][selected]
    context = ExpertContext(folder)
    pe, ke = context.target_components(data, data['support'][0])
    state = dict(dataset=dataset, task=task, eta=info['eta'], ids=data['ids'][selected],
                 selected=selected, rows=rows, labels=data['support_labels'][0],
                 frequency_checkpoint=folder/'selective.pt')
    columns = ['image_relpath_vv', 'image_relpath_vh'] if dataset == 'self24ports' else ['chip_path_vv', 'chip_path_vh']
    frame = pd.read_csv(MANIFESTS[dataset], usecols=columns, keep_default_na=False) if dataset != 'f0921' else None
    if dataset == 'self24ports':
        state['geometry'] = context.geometry
        state['heads'] = load_pickle(ROOT/'direct_incidence_comparison_20261009/ports'/task/'direct_heads.pkl')
        state['tail_checkpoint'] = ROOT/'module_completion_20261008/ports_projection'/task/'joint/layer4.pt'
        state['paths'] = list(frame.iloc[rows][['image_relpath_vv', 'image_relpath_vh']].itertuples(index=False, name=None))
        state['angles'] = _conditions()[rows]
        raw = np.asarray(np.load(SOURCE/'pooled_features.npy', mmap_mode='r')[rows])
        c4 = np.asarray(np.load(ROOT/'c4_base_trial_20261009/pooled_c4.npy', mmap_mode='r')[rows])
        extra = np.asarray(np.load(ROOT/'module_completion_20261008/ports_projection'/task/'matched_joint/joint_extra.npy', mmap_mode='r')[selected])
    else:
        state['base_model'] = load_pickle(Path(info['model_path']))
        state['expert_model'], state['stats'] = context.model, context.stats
        state['tail_checkpoint'] = context.joint/'layer4.pt'
        state['angles'] = np.zeros((len(rows), 2))
        if dataset == 'f0921':
            state['pixel_path'] = MANIFESTS[dataset].parent/'inputs128.npy'
            state['paths'] = None
            raw = np.asarray(np.load(MANIFESTS[dataset].parent/'paired_r50.npy', mmap_mode='r')[rows])
        else:
            state['paths'] = list(frame.iloc[rows][['chip_path_vv', 'chip_path_vh']].itertuples(index=False, name=None))
            raw = load_npz(MANIFESTS[dataset].parent/'features.npz')['paired_r50'][rows]
        c4 = np.asarray(np.load(ROOT/'cross_dataset_c4_20261009'/dataset/'pooled_c4.npy', mmap_mode='r')[rows])
        extra = np.asarray(np.load(context.joint/'target_extra.npy', mmap_mode='r')[rows])
    descriptors = pd.read_csv(ROOT/'c4_condition_diagnosis_20261009/descriptors'/f'{dataset}.csv', usecols=['row_id', *KEYS]).set_index('row_id')
    cohort = descriptors.loc[data['rows']]
    state['rank_references'] = [np.sort(cohort[k].to_numpy()[np.isfinite(cohort[k])]) for k in KEYS]
    state['descriptors_reference'] = descriptors.loc[rows, list(KEYS)].to_numpy()
    ranks = rank_against_cohort(state['descriptors_reference'], state['rank_references'])
    np.testing.assert_allclose(ranks, data['conditions'][selected], atol=1e-12)
    components = geometry(state, raw, c4, extra, ranks, state['angles'])
    components['embedding'] = load_npz(folder/'selective_embedding.npz')['embedding'][selected].astype(float)
    checks = {}
    for key in ('phi', 'mean', 'prior', 'conditions'):
        checks[key] = float(np.max(np.abs(components[key]-data[key][selected])))
        np.testing.assert_allclose(components[key], data[key][selected], atol=3e-5, rtol=1e-6)
    support = {k: v[:64] for k, v in components.items()}
    kb, calculated_ke = kernels(components, support, state['eta'])
    checks['expert_prior'] = float(np.max(abs(components['expert_prior']-pe[selected])))
    checks['expert_kernel'] = float(np.max(abs(calculated_ke-ke[selected])))
    np.testing.assert_allclose(components['expert_prior'], pe[selected], atol=3e-5, rtol=1e-6)
    np.testing.assert_allclose(calculated_ke, ke[selected], atol=3e-5, rtol=1e-6)
    alpha = fit(support, state['labels'], state['eta'])
    frozen = load_npz(ROOT/'unified_head_20261010/targets'/dataset/task/'policies.npz')
    predictions = {}
    for mode in ('late_softmax_fixed', 'unified_fixed'):
        expected = frozen['predictions'][0, list(frozen['modes']).index(mode), selected]
        actual = predict(components, support, alpha, state['eta'], mode).argmax(1)
        np.testing.assert_array_equal(actual, expected)
        predictions[mode] = expected
    state['reference'] = dict(raw=raw, c4=c4, extra=extra, components=components,
                              predictions=predictions, support=selected[:64].tolist(), query=query.tolist())
    destination = OUT/dataset
    destination.mkdir(parents=True, exist_ok=True)
    with (destination/'prepared.pkl').open('wb') as stream:
        pickle.dump(state, stream)
    write(destination/'CPU_REPLAY.json', dict(dataset=dataset, task=task, rows=rows.tolist(),
        ids=state['ids'].tolist(), query_indices=query.tolist(), support_draw=0, support_count=64,
        cached_geometry_max_errors=checks, cached_readouts_exact=True,
        rule='16 equally spaced eligible query indices, fixed before pixel replay; no query labels loaded'))
    print('PREPARED '+dataset+' '+json.dumps(checks), flush=True)


class Pipeline:
    def __init__(self, state):
        import torch
        from torch import nn
        from torchvision.models import resnet50, resnet18, ResNet50_Weights
        from torchvision import transforms
        from learned_frequency_model import FrequencyResidual
        self.state = state
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.pixels = np.load(state['pixel_path'], mmap_mode='r') if state['dataset'] == 'f0921' else None
        self.transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
        weights = ResNet50_Weights.IMAGENET1K_V2
        checkpoint = Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/', 1)[-1]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        net = resnet50(weights=None)
        net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        paired = nn.Conv2d(2, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            paired.weight.copy_(net.conv1.weight.mean(1, keepdim=True).repeat(1, 2, 1, 1)*1.5)
        net.conv1 = paired
        self.r50_prefix = nn.Sequential(*list(net.children())[:6]).cuda().eval()
        self.r50_tail = nn.Sequential(net.layer3, net.layer4, nn.AdaptiveAvgPool2d((2, 2))).cuda().eval()
        net = resnet18(weights=None)
        ck = torch.load(WS/'e206_out/ssl_resnet18.pt', map_location='cpu', weights_only=True)
        net.load_state_dict(ck['backbone'], strict=False)
        self.ssl_prefix = nn.Sequential(*list(net.children())[:-3]).cuda().eval()
        net.layer4.load_state_dict(torch.load(state['tail_checkpoint'], map_location='cpu', weights_only=True))
        self.ssl_tail = net.layer4.cuda().eval()
        self.frequency = FrequencyResidual('selective').cuda().eval()
        ck = torch.load(state['frequency_checkpoint'], map_location='cpu', weights_only=True)
        self.frequency.load_state_dict(ck['state'])

    def read(self, index):
        if self.pixels is not None:
            pair = np.array(self.pixels[self.state['rows'][index]], copy=True)
            ssl = None
        else:
            pair = read_pair(self.state['paths'][index])[0]
            ssl = None
            if self.state['dataset'] == 'oss2':
                with Image.open(self.state['paths'][index][1]) as image:
                    ssl = self.transform(image.convert('RGB'))
        descriptor = describe(pair)
        return pair, ssl, [descriptor[k] for k in KEYS]

    def components(self, indices, keep_features=False):
        import torch
        items = list(self.pool.map(self.read, indices))
        conditions = pixel_conditions(np.array([a[2] for a in items]), self.state, indices)
        pixels = torch.from_numpy(np.stack([a[0] for a in items])).cuda().float()
        x = (pixels/255.-.5)/.25
        orbit = []
        with torch.inference_mode():
            for rotation in range(4):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    maps = self.r50_prefix(torch.rot90(x, rotation, dims=(-2, -1)))
                    spatial = self.r50_tail(maps).flatten(2).transpose(1, 2)
                orbit.append(spatial.float().cpu().numpy().astype(np.float16).astype(np.float32).mean(1))
                if rotation == 0:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        embedding, _ = self.frequency(maps.half().float(), torch.as_tensor(conditions, dtype=torch.float32, device='cuda'))
                    z = embedding.float().cpu().numpy().astype(float)
            if self.state['dataset'] != 'oss2':
                prefix = self.ssl_prefix(pixels[:, 1:2].repeat(1, 3, 1, 1)/255.)
                extra = self.ssl_tail(prefix.half().float()).mean((2, 3)).cpu().numpy()
            else:
                extra = np.empty((len(items), 512), np.float32)
                groups = {}
                for index, (_, tensor, _) in enumerate(items):
                    groups.setdefault(tuple(tensor.shape), []).append(index)
                for group in groups.values():
                    inputs = torch.stack([items[j][1] for j in group]).cuda()
                    extra[group] = self.ssl_tail(self.ssl_prefix(inputs)).mean((2, 3)).cpu().numpy()
        raw, c4 = orbit[0], np.mean(orbit, axis=0)
        result = geometry(self.state, raw, c4, extra, conditions, self.state['angles'][indices])
        result['embedding'] = z
        return (result, dict(raw=raw, c4=c4, extra=extra)) if keep_features else result

    def close(self):
        self.pool.shutdown()


class RssSampler:
    def __init__(self):
        self.event = threading.Event()
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.event.wait(.002):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.event.set()
        self.thread.join()


def relative_error(actual, expected):
    return dict(max_abs=float(np.max(abs(actual-expected))),
                relative_l2_max=float(np.max(np.linalg.norm(actual-expected, axis=1)/
                                            np.maximum(np.linalg.norm(expected, axis=1), 1e-12))))


def measure(dataset, repeats):
    import torch
    destination = OUT/dataset
    started = time.perf_counter()
    with (destination/'prepared.pkl').open('rb') as stream:
        state = pickle.load(stream)
    reference = state.pop('reference')
    cached = geometry(state, reference['raw'], reference['c4'], reference['extra'],
                      reference['components']['conditions'], state['angles'])
    for key, value in cached.items():
        np.testing.assert_allclose(value, reference['components'][key], atol=3e-5, rtol=1e-6)
    pipe = Pipeline(state)
    torch.cuda.synchronize()
    load_ms = 1000*(time.perf_counter()-started)
    support_indices = np.arange(64)
    started = time.perf_counter()
    component_batches, feature_batches = [], []
    for start in range(0, 80, 16):
        c, f = pipe.components(np.arange(start, start+16), keep_features=True)
        component_batches.append(c)
        feature_batches.append(f)
    components = {k: np.concatenate([c[k] for c in component_batches]) for k in component_batches[0]}
    features = {k: np.concatenate([f[k] for f in feature_batches]) for k in feature_batches[0]}
    parity = {k: relative_error(v, reference[k]) for k, v in features.items()}
    for k in ('phi', 'mean', 'prior', 'expert_prior', 'expert', 'conditions', 'embedding'):
        parity[k] = relative_error(components[k], reference['components'][k])
    support = {k: v[:64] for k, v in components.items()}
    alpha = fit(support, state['labels'], state['eta'])
    reference_support = {k: v[:64] for k, v in reference['components'].items()}
    reference_alpha = fit(reference_support, state['labels'], state['eta'])
    reference_scores = {mode: predict(reference['components'], reference_support, reference_alpha, state['eta'], mode)
                        for mode in ('late_softmax_fixed', 'unified_fixed')}
    agreement = {}
    for mode in ('late_softmax_fixed', 'unified_fixed'):
        prediction = predict(components, support, alpha, state['eta'], mode).argmax(1)
        expected = reference['predictions'][mode]
        actual_scores = predict(components, support, alpha, state['eta'], mode)
        agreement[mode] = dict(support_equal=int(np.sum(prediction[:64] == expected[:64])),
                               query_equal=int(np.sum(prediction[64:] == expected[64:])), query_count=16,
                               score_max_abs=float(np.max(abs(actual_scores-reference_scores[mode]))),
                               changed_query_ids=state['ids'][64:][prediction[64:] != expected[64:]].tolist())
    write(destination/'PIXEL_REPLAY.json', dict(features=parity, predictions=agreement,
        selection='Original draw0 supports plus predetermined16 queries; not selected for successful replay',
        replay_ms=1000*(time.perf_counter()-started)))
    print('PIXEL REPLAY '+dataset+' '+json.dumps(agreement), flush=True)
    if parity['raw']['relative_l2_max'] > .02 or parity['c4']['relative_l2_max'] > .02 or parity['extra']['relative_l2_max'] > .002:
        raise ValueError('Pixel recipe differs from frozen model; inspect PIXEL_REPLAY before timing')
    if parity['conditions']['max_abs'] > 1e-6:
        raise ValueError('Current pixel-derived conditions differ from frozen cohort; inspect before timing')
    timings = []
    setup_times = {mode: [] for mode in ('late_softmax_fixed', 'unified_fixed')}
    setup_memory = {}
    for mode in setup_times:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            chunks = [pipe.components(support_indices[i:i+16]) for i in range(0, 64, 16)]
            support = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
            fit(support, state['labels'], state['eta'], mode)
            torch.cuda.synchronize()
            setup_times[mode].append(1000*(time.perf_counter()-start))
        setup_memory[mode] = dict(gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                  gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    alpha = fit(support, state['labels'], state['eta'])
    fit_times = {mode: [] for mode in ('late_softmax_fixed', 'unified_fixed')}
    kb, ke = kernels(support, support, state['eta'])
    for repeat in range(20):
        start = time.perf_counter()
        coefficients(kb, support['prior'], state['labels'], 8.)
        coefficients(ke, support['expert_prior'], state['labels'], 8.)
        fit_times['late_softmax_fixed'].append(1000*(time.perf_counter()-start))
        start = time.perf_counter()
        coefficients(.5*(kb+ke), .5*(support['prior']+support['expert_prior']), state['labels'], 8.)
        fit_times['unified_fixed'].append(1000*(time.perf_counter()-start))
    for batch in (1, 16):
        for mode in ('late_softmax_fixed', 'unified_fixed'):
            indices = np.arange(64, 64+batch)
            for _ in range(2):
                c = pipe.components(indices)
                predict(c, support, alpha, state['eta'], mode)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            times = []
            with RssSampler() as memory:
                for repeat in range(repeats):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    c = pipe.components(indices)
                    scores = predict(c, support, alpha, state['eta'], mode)
                    torch.cuda.synchronize()
                    times.append(1000*(time.perf_counter()-start))
            timings.append(dict(batch=batch, mode=mode, median_batch_ms=float(np.median(times)),
                p25_batch_ms=float(np.percentile(times, 25)), p75_batch_ms=float(np.percentile(times, 75)),
                median_ms_per_image=float(np.median(times)/batch), repeats=repeats,
                gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                sampled_process_peak_rss_mib=memory.peak/2**20, batch_ms=times,
                frozen_prediction_equal=int(np.sum(scores.argmax(1) == reference['predictions'][mode][indices])),
                frozen_prediction_count=len(indices), frozen_score_max_abs=float(np.max(abs(scores-reference_scores[mode][indices])))))
            print('MEASURE '+dataset+' '+json.dumps(timings[-1]), flush=True)
    receipt = dict(dataset=dataset, task=state['task'], device=torch.cuda.get_device_name(),
        torch_version=torch.__version__, input_count=16, labels=64, support_draw=0,
        model_loading_ms=load_ms, model_loading_scope='Diagnostic initialization: state/checkpoint load, cached-geometry parity check, CPU model construction and CUDA transfer; not pure production loading. Python import and CUDA context creation excluded; cache warmth uncontrolled',
        support64_actual_pixels_and_readout_fit_ms=setup_times,
        support64_median_ms={m: float(np.median(v)) for m, v in setup_times.items()},
        support64_memory=setup_memory,
        coefficient_only_fit_median_ms={m: float(np.median(v)) for m, v in fit_times.items()},
        timings=timings, image_io_workers=4, blas_threads=2, torch_threads=2,
        timing_scope='Query image/pixel read, preprocessing, pixel proxies/CDF lookup, H2D, actual R50 C4/shared direction-zero layer2, frequency adapter, trained SSL18, D2H, CPU geometry, kernels against64 supports and selected readout',
        f0921_boundary='Existing actual uint8 VV/VH128 chip mmap, not feature cache. Original SAR-scene window/crop and initial128resize excluded.',
        other_input_boundary='Real VV/VH PNG decode and PIL L bilinear128resize. OSS2 SSL also uses native VH RGB Resize128 aspect ratio retained.',
        cohort_boundary='Target cohort descriptor CDF precomputed and resident; pixel descriptor recomputed per query. Acquisition angles and source geometry resident. Not strict unseen streaming or raw-scene end-to-end.',
        condition_ties='Recomputed descriptors within1e-12 of the same known cohort row are snapped to its serialized value to preserve existing exact ties. Measured raw descriptor disagreement was at most4.44e-16.',
        offline_exclusions=['source training and geometry fitting', 'whole-cohort descriptor acquisition/sort', 'metadata discovery/download', 'original scene crop', 'Python import/CUDA context initialization'],
        filesystem='Warm repeated image/pixel reads after explicit replay and2warmups; no OS cache eviction',
        memory='CUDA allocator peak allocated/reserved for this single task process; RSS sampled every2ms, includes resident model/state and replay references, not a guaranteed OS peak',
        precision='R50/frequency bf16 autocast, map float16 boundary; R18 float32; prefix float16 boundary only for24ports/F; CPU kernels float64; TF32 disabled',
        actual_weights=dict(r50='torch ImageNet1K V2 cached checkpoint, paired conv conversion',
            ssl=str(WS/'e206_out/ssl_resnet18.pt'), tail=str(state['tail_checkpoint']), frequency=str(state['frequency_checkpoint'])),
        replay=agreement, no_query_truth_read=True)
    write(destination/'MEASURED.json', receipt)
    pipe.close()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'pixel-conditions', 'measure'), required=True)
    parser.add_argument('--datasets', nargs='+', default=['self24ports', 'f0921', 'oss2'])
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    if args.stage == 'measure':
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.init()
    with threadpool_limits(limits=2):
        for dataset in args.datasets:
            if args.stage == 'prepare':
                prepare(dataset)
            elif args.stage == 'pixel-conditions':
                with (OUT/dataset/'prepared.pkl').open('rb') as stream:
                    state = pickle.load(stream)
                if dataset == 'f0921':
                    pixels = np.load(state['pixel_path'], mmap_mode='r')
                    pairs = [np.asarray(pixels[row]) for row in state['rows']]
                else:
                    pairs = [read_pair(paths)[0] for paths in state['paths']]
                values = np.array([[describe(pair)[k] for k in KEYS] for pair in pairs])
                actual = pixel_conditions(values, state, np.arange(len(values)))
                expected = state['reference']['components']['conditions']
                difference = np.abs(actual-expected)
                descriptor_error = float(np.nanmax(np.abs(values-state['descriptors_reference'])))
                write(OUT/dataset/'PIXEL_CONDITIONS.json', dict(max_abs=float(difference.max()), descriptor_max_abs=descriptor_error,
                    changed=int(np.sum(difference > 1e-12)), actual=actual.tolist(), expected=expected.tolist()))
                print(dataset+' pixel condition difference '+str(difference.max())+' descriptor error '+str(descriptor_error), flush=True)
            else:
                measure(dataset, args.repeats)


if __name__ == '__main__':
    main()
