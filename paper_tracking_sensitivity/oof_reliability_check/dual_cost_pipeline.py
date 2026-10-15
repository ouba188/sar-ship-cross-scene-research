"""Matched single/dual-backbone actual-pixel timing, without source refitting."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
import json
from pathlib import Path
import pickle
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from completion_e2e_pipeline import (ROOT, OUT as PREVIOUS, Pipeline, read_pair, describe,
    KEYS, pixel_conditions, normalize_direction, fit, predict, write)

OUT = ROOT/'dual_cost_20261010'
ARMS = ('r50', 'dual')


def components_from_features(state, raw, c4, extra, conditions, angles, arm):
    def discarded(x, pca):
        centered = np.asarray(x, float)-pca.mean_
        pc = np.asarray(pca.components_, float)
        return centered-(centered@pc.T)@pc

    if state['dataset'] == 'self24ports':
        from probe_display_rotation import replace_r50, base_phi
        g = state['geometry']
        m, stats, r = g['model'], g['stats'], g['stable_rank']
        bank = np.zeros((len(raw), 192+r+g['drift_rank']))
        bank[:, :128] = ((m['pca'].transform(raw)-m['center'])/m['scale']).astype(np.float32)
        if arm == 'dual':
            bank[:, 128:192] = m['pca_ssl'].transform(extra).astype(np.float32)
        standardized = (raw.astype(float)-g['feature_center'])/g['feature_scale']
        bank[:, 192:192+r] = (standardized@g['qs']-g['stable_center'])/g['stable_scale']
        zd = standardized@g['qd']
        bank[:, 192+r:] = zd
        base = replace_r50(c4.astype(float), bank, g)
        phi = base_phi(base, g, angles)
        mean = normalize_direction((base[:, :128]-g['base_center'])/g['base_scale'])
        head = state['heads']['joint_base']
        pb = np.c_[base, angles]@head['coef']+head['intercept']
        fitted = state['matched_heads'][arm]
        augmented = np.c_[bank, angles]
        keep = np.r_[np.arange(128), np.arange(192, augmented.shape[1])] if arm == 'r50' else np.arange(augmented.shape[1])
        pe = augmented[:, keep]@fitted['coef'][keep]+fitted['intercept']
        core = bank[:, :128]-.5*zd@g['drift_to_xs']
        width = 192 if arm == 'dual' else 128
        retained = np.c_[core, bank[:, 128:192].astype(np.float32)] if arm == 'dual' else core
        retained = (retained-m['expert_center'][:width])/m['expert_scale'][:width]
        comp = discarded(raw, m['pca'])
        if arm == 'dual':
            comp = np.c_[comp, discarded(extra, m['pca_ssl'])]
        comp = (comp-stats['discarded_center'][:comp.shape[1]])/stats['discarded_scale'][:comp.shape[1]]
        comp[:, :2048] -= .5*zd@g['drift_to_complement']
        common = np.c_[np.ones(len(raw)), .5*bank[:, 192:192+r]/np.sqrt(r), np.sqrt(.125)*angles]
    else:
        from cross_dataset_c4 import base_coordinates, base_phi
        bm = state['base_model']
        x = base_coordinates(c4, bm)
        phi = base_phi(x, bm)
        mean = normalize_direction((x-bm['residual_center'])/bm['residual_scale'])
        pb = bm['base'].decision_function(x).astype(np.float32).astype(float)
        m, stats = state['expert_model'], state['stats']
        xs = ((m['pca'].transform(raw)-m['center'])/m['scale']).astype(np.float32)
        retained = np.c_[xs, m['pca_ssl'].transform(extra).astype(np.float32)] if arm == 'dual' else xs
        width = retained.shape[1]
        retained = (retained-m['expert_center'][:width])/m['expert_scale'][:width]
        fitted = state['matched_heads'][arm]
        pe = (retained@fitted['coef']+fitted['intercept']).astype(np.float32).astype(float)
        comp = discarded(raw, m['pca'])
        if arm == 'dual':
            comp = np.c_[comp, discarded(extra, m['pca_ssl'])]
        comp = (comp-stats['discarded_center'][:comp.shape[1]])/stats['discarded_scale'][:comp.shape[1]]
        common = np.ones((len(raw), 1))
    expert = np.c_[np.sqrt(.5)*retained/np.sqrt(192), np.sqrt(.5)*comp/np.sqrt(2560), common]
    return dict(phi=phi, mean=mean, conditions=conditions, prior=pb, expert_prior=pe, expert=expert)


def prepare(dataset):
    with (PREVIOUS/dataset/'prepared.pkl').open('rb') as stream:
        state = pickle.load(stream)
    path = ROOT/'backbone_ablation_20261010/targets'/dataset/state['task']
    with np.load(path/'coefficients.npz') as saved:
        state['matched_heads'] = {a: {k: saved[a+'_'+k] for k in ('coef', 'intercept')} for a in ARMS}
    if dataset == 'self24ports':
        np.testing.assert_array_equal(state['heads']['joint_base']['coef'][128:192], 0.)
        np.testing.assert_array_equal(state['matched_heads']['r50']['coef'][128:192], 0.)
    old = state['reference']
    conditions = old['components']['conditions']
    state['matched_reference'] = {}
    checks = {}
    with np.load(path/'predictions.npz') as saved:
        np.testing.assert_array_equal(saved['ids'][state['selected']], state['ids'])
        for arm in ARMS:
            c = components_from_features(state, old['raw'], old['c4'], old['extra'] if arm == 'dual' else None,
                                         conditions, state['angles'], arm)
            c['embedding'] = old['components']['embedding']
            for key in ('phi', 'mean', 'conditions', 'prior'):
                np.testing.assert_allclose(c[key], old['components'][key], atol=3e-5, rtol=1e-6)
            support = {k: v[:64] for k, v in c.items()}
            alpha = fit(support, state['labels'], state['eta'])
            predictions, scores = {}, {}
            for mode, suffix in [('late_softmax_fixed', 'late'), ('unified_fixed', 'unified')]:
                name = ('dual_refit' if arm == 'dual' else 'r50')+'_'+suffix
                expected = saved['predictions'][0, list(saved['modes']).index(name), state['selected']]
                scores[mode] = predict(c, support, alpha, state['eta'], mode)
                np.testing.assert_array_equal(scores[mode].argmax(1), expected)
                predictions[mode] = expected
            state['matched_reference'][arm] = dict(scores=scores, predictions=predictions)
            checks[arm] = dict(cached_predictions_equal=80, expert_dimensions=c['expert'].shape[1])
    # All raw references are only80 rows; avoid touching any full feature cache.
    folder = OUT/dataset
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'prepared.pkl').open('wb') as stream:
        pickle.dump(state, stream)
    write(folder/'PREPARED.json', dict(dataset=dataset, task=state['task'], checks=checks,
        source_heads=str(path/'coefficients.npz'), source_refit=False,
        same64_support_and16_queries_as_previous=True, target_labels_used=64))
    print('CPU MATCHED '+dataset+' '+json.dumps(checks), flush=True)


class CostPipeline(Pipeline):
    def __init__(self, state):
        super().__init__(state)
        self.arm = 'dual'

    def activate(self, arm):
        import torch
        self.arm = arm
        device = 'cuda' if arm == 'dual' else 'cpu'
        self.ssl_prefix.to(device)
        self.ssl_tail.to(device)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if arm == 'r50' and any(p.is_cuda for m in (self.ssl_prefix, self.ssl_tail) for p in m.parameters()):
            raise AssertionError('R18 still resident on CUDA during single-backbone arm')

    def read(self, index):
        if self.arm == 'dual':
            return super().read(index)
        pair = np.array(self.pixels[self.state['rows'][index]], copy=True) if self.pixels is not None else read_pair(self.state['paths'][index])[0]
        descriptor = describe(pair)
        return pair, None, [descriptor[k] for k in KEYS]

    def components(self, indices, keep_features=False):
        import torch
        items = list(self.pool.map(self.read, indices))
        conditions = pixel_conditions(np.array([v[2] for v in items]), self.state, indices)
        pixels = torch.from_numpy(np.stack([v[0] for v in items])).cuda().float()
        x = (pixels/255.-.5)/.25
        orbit = []
        extra = None
        with torch.inference_mode():
            for rotation in range(4):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    maps = self.r50_prefix(torch.rot90(x, rotation, dims=(-2, -1)))
                    spatial = self.r50_tail(maps).flatten(2).transpose(1, 2)
                orbit.append(spatial.float().cpu().numpy().astype(np.float16).astype(np.float32).mean(1))
                if rotation == 0:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        embedding, _ = self.frequency(maps.half().float(), torch.as_tensor(conditions, device='cuda', dtype=torch.float32))
                    z = embedding.cpu().numpy().astype(float)
            if self.arm == 'dual':
                if self.state['dataset'] != 'oss2':
                    prefix = self.ssl_prefix(pixels[:, 1:2].repeat(1, 3, 1, 1)/255.)
                    extra = self.ssl_tail(prefix.half().float()).mean((2, 3)).cpu().numpy()
                else:
                    groups = {}
                    for j, (_, tensor, _) in enumerate(items):
                        groups.setdefault(tuple(tensor.shape), []).append(j)
                    extra = np.empty((len(items), 512), np.float32)
                    for group in groups.values():
                        inputs = torch.stack([items[j][1] for j in group]).cuda()
                        extra[group] = self.ssl_tail(self.ssl_prefix(inputs)).mean((2, 3)).cpu().numpy()
        raw, c4 = orbit[0], np.mean(orbit, axis=0)
        c = components_from_features(self.state, raw, c4, extra, conditions, self.state['angles'][indices], self.arm)
        c['embedding'] = z
        return c


def run(dataset):
    import torch
    with (OUT/dataset/'prepared.pkl').open('rb') as stream:
        state = pickle.load(stream)
    reference = state.pop('matched_reference')
    del state['reference']
    pipe = CostPipeline(state)
    supports, alphas, parity = {}, {}, {}
    for arm in ARMS:
        pipe.activate(arm)
        chunks = [pipe.components(np.arange(i, i+16)) for i in range(0, 80, 16)]
        c = {k: np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
        supports[arm] = {k: v[:64] for k, v in c.items()}
        alphas[arm] = fit(supports[arm], state['labels'], state['eta'])
        parity[arm] = {}
        for mode in ('late_softmax_fixed', 'unified_fixed'):
            score = predict(c, supports[arm], alphas[arm], state['eta'], mode)
            expected = reference[arm]['predictions'][mode]
            matches = score.argmax(1) == expected
            parity[arm][mode] = dict(support_equal=int(matches[:64].sum()), query_equal=int(matches[64:].sum()),
                query_count=16, score_max_abs=float(np.max(abs(score-reference[arm]['scores'][mode]))),
                changed_query_ids=state['ids'][64:][~matches[64:]].tolist())
        print('PIXEL MATCHED '+dataset+'/'+arm+' '+json.dumps(parity[arm]), flush=True)
    write(OUT/dataset/'PIXEL_REPLAY.json', parity)
    records = []
    for batch in (1, 16):
        indices = np.arange(64, 64+batch)
        for repeat in range(10):
            order = ARMS if repeat % 2 == 0 else ARMS[::-1]
            for arm in order:
                pipe.activate(arm)
                # Residency changes are benchmark orchestration, not query work.
                warm = pipe.components(indices)
                predict(warm, supports[arm], alphas[arm], state['eta'], 'late_softmax_fixed')
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                resident = torch.cuda.memory_allocated()/2**20
                start = time.perf_counter()
                current = pipe.components(indices)
                score = predict(current, supports[arm], alphas[arm], state['eta'], 'late_softmax_fixed')
                torch.cuda.synchronize()
                elapsed = 1000*(time.perf_counter()-start)
                matches = score.argmax(1) == reference[arm]['predictions']['late_softmax_fixed'][indices]
                records.append(dict(batch=batch, pair=repeat, arm=arm, order=list(order), batch_ms=elapsed,
                    cuda_resident_mib=resident, cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    process_rss_mib=psutil.Process().memory_info().rss/2**20,
                    reference_prediction_equal=int(matches.sum()), reference_count=batch))
        print('TIMED '+dataset+' batch'+str(batch), flush=True)
    summary = []
    for batch in (1, 16):
        times = {}
        for arm in ARMS:
            rows = [r for r in records if r['batch'] == batch and r['arm'] == arm]
            times[arm] = np.array([r['batch_ms'] for r in rows])
            summary.append(dict(batch=batch, arm=arm, median_ms=float(np.median(times[arm])),
                p25_ms=float(np.percentile(times[arm], 25)), p75_ms=float(np.percentile(times[arm], 75)),
                peak_allocated_mib=max(r['cuda_peak_allocated_mib'] for r in rows),
                peak_reserved_mib=max(r['cuda_peak_reserved_mib'] for r in rows),
                resident_mib=max(r['cuda_resident_mib'] for r in rows)))
        difference = times['dual']-times['r50']
        summary.append(dict(batch=batch, contrast='dual_minus_r50', paired_median_delta_ms=float(np.median(difference)),
            paired_p25_delta_ms=float(np.percentile(difference, 25)), paired_p75_delta_ms=float(np.percentile(difference, 75)),
            paired_median_percent=float(np.median(100*difference/times['r50'])),
            extra_ms_per_image=float(np.median(difference)/batch)))
    write(OUT/dataset/'MEASURED.json', dict(dataset=dataset, task=state['task'],
        device=torch.cuda.get_device_name(), torch_version=torch.__version__, summary=summary, records=records, parity=parity,
        scope='Actual model-input pixels to equal-probability two-head output; full C4/frequency base retained in both arms; matched saved source-refit expert heads',
        residency='Single arm moves R18 parameters/buffers toCPU and clearsCUDAcache before warmup/reset. R18 forward and OSS2 extraVH RGBread absent. Arm switch/modeltransfer not timed.',
        protocol='10 balanced AB/BA paired repetitions perB1/B16, one warm query after each residency transition, CUDA sync, same fixed query/support pixels',
        boundary='Fixed known cohort and warm filesystem; sourcefit/64supportsetup/CDFsetup/modelinitialization excluded. F reads true uint8inputs128, not original scene. Not raw-scene or strict-stream deployment.',
        numerical_boundary='Historicalraw feature mismatch retained from prior probe; no renewed precision search. Pixel predictions compared to matched backbone ablation, not old-source priors.',
        additional_neural_training=False, additional_source_fitting=False))
    pipe.close()
    del pipe, state
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'measure'), required=True)
    parser.add_argument('--datasets', nargs='+', default=['self24ports', 'f0921', 'oss2'])
    args = parser.parse_args()
    if args.stage == 'measure':
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    with threadpool_limits(limits=2):
        for dataset in args.datasets:
            prepare(dataset) if args.stage == 'prepare' else run(dataset)


if __name__ == '__main__':
    main()
