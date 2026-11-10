"""Cost-only measurements of the current staged operators; no fitting."""
from pathlib import Path
import argparse
import copy
import gc
import json
import os
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
sys.path.insert(0, str(PAPER / 'dataset_completion_20261110'))
import engine as e
import numpy as np
import pandas as pd

torch, task, raw = e.torch, e.task, e.old.raw
nn = torch.nn
DEVICE = task.DEVICE
STAGED = PAPER / 'staged_ablation_20261109'
PRETRAIN = json.loads((STAGED / 'protocol.json').read_text())['pretrain']


def save(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def resident_tensor(value, ids):
    if torch.is_tensor(value):
        return value[torch.as_tensor(ids, device=value.device)]
    return torch.as_tensor(np.array(value[ids], copy=True), device=DEVICE)


def prepare(rows):
    """Existing native160 preprocessing; no resizing, labels or new detection."""
    n = len(rows)
    images = np.zeros((n, 2, 1, 160, 160), np.float32)
    valid = np.zeros_like(images)
    av = np.zeros((n, 2), np.float32)
    slots = np.zeros((n, 2, 20, 10, 10), np.float32)
    local = np.zeros((n, 2, 20, 20, 20), np.float32)
    geometry = []
    for j, row in enumerate(rows):
        gs = []
        for p, view in enumerate(task.data.prepare(row)):
            good = view is not None and view['valid_target']
            if good:
                images[j, p, 0] = view['pixels'].astype(np.float32) / 255
                valid[j, p, 0] = view['footprint']
                av[j, p] = 1
                native = raw.tokens.slots(view['geometry'], view['footprint'])
                for dest, size in ((slots, 10), (local, 20)):
                    dest[j, p] = raw.grid_occupancy(torch.from_numpy(native).float()[None], (size, size))[0].numpy()
            gs.append(raw.build_masks(view['geometry'] if good else None,
                                     view['footprint'] if good else np.zeros((160, 160)), 'coarse_fixed'))
        geometry.append(gs)
    arrays = dict(images=images, valid=valid, available=av, slots=slots, local_slots=local)
    arrays.update({k: np.stack([[g[k] for g in gs] for gs in geometry]) for k in ('read', 'mass', 'write')})
    return {k: torch.as_tensor(v, device=DEVICE) for k, v in arrays.items()}


class ChipPath(nn.Module):
    """Restore the frozen prefix, then call the unmodified staged forward."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        # M3 descriptors used the original ImageNet layer2 output during training.
        # Only this conv differs from the live path; the preceding prefix is shared.
        block = net.base.backbone.model.layer2[3]
        self.anchor_conv = copy.deepcopy(block.conv3).requires_grad_(False)
        self.anchor_bn = copy.deepcopy(block.bn3).requires_grad_(False)

    def extract(self, data, descriptors):
        n = len(data['images'])
        bb = self.net.base.backbone.model
        f1 = self.net.base.prefix(data['images'], stop=1)
        z = f1.flatten(0, 1)
        for layer in list(bb.layer2.children())[:3]:
            z = layer(z)
        identity = z
        block = bb.layer2[3]
        x = block.relu(block.bn1(block.conv1(z)))
        x = block.relu(block.bn2(block.conv2(x)))
        a = {k: data[k] for k in ('valid', 'available', 'slots', 'mass', 'write')}
        if descriptors:
            f2 = block.relu(self.anchor_bn(self.anchor_conv(x)) + identity)
            a['desc'] = raw.pool_multiscale(f1.flatten(0, 1), f2, data['read'].flatten(0, 1),
                                            'coarse_fixed').reshape(n, 2, 20, 768)
        c = dict(early_x=x.reshape(n, 2, 128, 20, 20),
                 early_identity=identity.reshape(n, 2, 512, 20, 20), local_slots=data['local_slots'])
        return a, c

    def reference(self, data):
        a, _ = self.extract(data, True)
        return {k: a[k] for k in ('desc', 'mass', 'available')}

    def finish(self, a, c, query_rows, reference, reference_rows, m3_on=True):
        n = len(query_rows)
        table = dict(a)
        if self.net.use3 and m3_on:
            for key in ('desc', 'mass', 'available'):
                table[key] = torch.cat((a[key], reference[key]), 0)
        return self.net(table, c, query_rows + reference_rows, np.arange(n),
                        np.arange(n, n + len(reference_rows)), m3_on=m3_on)[0]

    def forward(self, data, query_rows, reference, reference_rows, m3_on=True):
        a, c = self.extract(data, self.net.use3 and m3_on)
        return self.finish(a, c, query_rows, reference, reference_rows, m3_on)


@torch.inference_mode()
def timed(fn, repetitions, rounds=3):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    values, gpu = [], []
    for _ in range(rounds):
        one, cuda = [], []
        for _ in range(repetitions):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            one.append((time.perf_counter() - tick) * 1000)
            cuda.append(start.elapsed_time(end))
        values.append(one)
        gpu.append(cuda)
    flat = np.array(values).flatten()
    return dict(median_ms=float(np.median(flat)), p90_ms=float(np.percentile(flat, 90)),
                mean_ms=float(flat.mean()), round_medians_ms=np.median(values, axis=1).tolist(),
                gpu_event_median_ms=float(np.median(gpu)), measured_calls=len(flat))


@torch.inference_mode()
def peak_memory(fn):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return dict(resident_allocated_mib=baseline / 2**20,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                incremental_peak_mib=(torch.cuda.max_memory_allocated() - baseline) / 2**20)


@torch.inference_mode()
def operation_counts(path, fn):
    used = set()
    handles = []
    for module in path.modules():
        def hook(m, args, result):
            used.update(id(p) for p in m.parameters(recurse=False))
        handles.append(module.register_forward_hook(hook))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                record_shapes=True, with_flops=True) as prof:
        fn()
        torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    # Explicit partial convention: normalization/reductions/softmax/Sinkhorn
    # elementwise costs are not silently reported as zero total arithmetic.
    covered = ('aten::conv2d', 'aten::mm', 'aten::bmm', 'aten::addmm')
    counts = {x.key: int(x.flops) for x in prof.key_averages() if x.key in covered}
    all_params = {id(p): p for p in path.parameters()}
    active = {i: all_params[i] for i in used}
    return dict(conv_matmul_gflops=sum(counts.values()) / 1e9,
                conv_matmul_gmacs=sum(counts.values()) / 2e9,
                flop_ops=counts, executed_parameters=sum(p.numel() for p in active.values()),
                inactive_parameters=[name for name, p in path.named_parameters() if id(p) not in used])


def training_counts(net):
    result, union = {}, set()
    for phase, name in ((0, 'carrier'), (-1, 'intervention'), (1, 'reference'), (2, 'feedback'), (3, 'readout')):
        e.configure(net, phase)
        params = {id(p): p for p in net.parameters() if p.requires_grad}
        result[name] = sum(p.numel() for p in params.values())
        union.update(params)
    sizes = {id(p): p.numel() for p in net.parameters()}
    result['unique_updated_parameters'] = sum(sizes[i] for i in union)
    net.requires_grad_(False).eval()
    return result


@torch.inference_mode()
def measure(arm, classes, references, qrows, rrows, qdata, rdata, repetitions):
    print('START', arm, classes, references, flush=True)
    net = e.Network(arm, 17, PRETRAIN, classes)
    path = ChipPath(net).eval()
    checkpoint = STAGED / f'{arm}_s17_lr{0.3 if arm == "se" else 1}' / 'readout/final.pt'
    restored = classes == 4
    if restored:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        net.load_state_dict(state, strict=True)
        del state
    counts = training_counts(net)
    # Single shared prefix plus a small frozen descriptor conv, not a second R50.
    path.eval().requires_grad_(False)
    ref = path.reference(rdata) if net.use3 else {}
    ref_bytes = sum(x.numel() * x.element_size() for x in ref.values())
    result = dict(arm=arm, classes=classes, references=references if net.use3 else 0,
                  checkpoint=str(checkpoint) if restored else None,
                  scope='trained OSS2 seed17' if restored else 'eight-class architecture only; no restored trained weights',
                  registered_parameters=sum(p.numel() for p in net.parameters()),
                  frozen_descriptor_conv_parameters=sum(p.numel() for p in path.anchor_conv.parameters()) if net.use3 else 0,
                  frozen_descriptor_bn_parameters=sum(p.numel() for p in path.anchor_bn.parameters()) if net.use3 else 0,
                  training=counts, reference_cache_mib=ref_bytes / 2**20)
    # Verify the online prefix against the original stored training descriptors.
    if restored:
        rows, groups = e.prev.records()
        lookup = {r['id']: i for i, r in enumerate(rows)}
        ids = np.array([lookup[r['id']] for r in qrows])
        rid = groups['references']
        a, c = e.old.load_arrays(rows)
        expected = net(a, c, rows, ids, rid)[0]
        actual = path(qdata, qrows, ref, rrows)
        error = float((expected - actual).abs().max())
        result['raw_cache_max_logit_error'] = error
        saved = np.load(checkpoint.parent.parent / 'evaluation.npz')
        ix = [int(np.flatnonzero(saved['ids'] == i)[0]) for i in ids]
        result['saved_prediction_max_logit_error'] = float(np.abs(actual.cpu().numpy() - saved['score'][ix]).max())
        assert error < 1e-4 and result['saved_prediction_max_logit_error'] < 1e-4, result
        del a, c, expected, actual, saved
    first = {k: v[:1] for k, v in qdata.items()}
    one = lambda: path(first, qrows[:1], ref, rrows)
    result.update(operation_counts(path, one))
    result['batch1'] = dict(**timed(one, repetitions), **peak_memory(one))
    result['batch8'] = dict(**timed(lambda: path(qdata, qrows, ref, rrows), max(3, repetitions // 3)),
                            **peak_memory(lambda: path(qdata, qrows, ref, rrows)))
    a, c = path.extract(first, net.use3)
    result['cached_suffix_batch1'] = timed(lambda: path.finish(a, c, qrows[:1], ref, rrows), max(3, repetitions // 3))
    if net.use3:
        result['reference_build'] = timed(lambda: path.reference(rdata), 3)
        result['same_weight_m3_off_batch1'] = timed(lambda: path(first, qrows[:1], ref, rrows, False), max(3, repetitions // 3))
    save(f'{classes}class_{arm}.json', result)
    print('DONE', arm, classes, result['batch1']['median_ms'], result['conv_matmul_gflops'], flush=True)
    del path, net, ref, a, c
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repetitions', type=int, default=15)
    parser.add_argument('--classes', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--arms', nargs='+', default=['plain', 'se', 'full', 'no_m1', 'no_m2', 'no_m3'])
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    task.current.setup(DEVICE)
    task.tensor = resident_tensor
    torch.set_num_threads(2)
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, device=str(DEVICE),
               gpu=torch.cuda.get_device_name(), gpu_total_mib=torch.cuda.get_device_properties(0).total_memory / 2**20,
               threads=torch.get_num_threads(), dtype='float32', batch_sizes=[1, 8], seed=17,
               cudnn_benchmark=torch.backends.cudnn.benchmark,
               cudnn_deterministic=torch.backends.cudnn.deterministic,
               cudnn_tf32=torch.backends.cudnn.allow_tf32, matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
               timing='GPU-resident preprocessed chips + OBB masks; cached source reference descriptors; synchronized host wall time',
               excluded='disk IO, CPU native160/mask preparation, detection, training and initial model loading',
               flops='conv2d/mm/bmm/addmm only, multiply-add=2; other ops excluded, not full arithmetic FLOPs')
    save('environment.json', env)
    outputs = []
    for classes in args.classes:
        if classes == 4:
            rows, groups = e.prev.records()
            rrows = [rows[i] for i in groups['references']]
            candidates = [rows[i] for i in groups['evaluation']]
        else:
            source = PAPER.parents[0] / 'scene_aware_classification_20261018/model_closeout_20261023/spatial_m12_full_20261029/cohorts.json'
            parts = json.loads(source.read_text(encoding='utf-8'))
            rrows, candidates = parts['references'], parts['expanded']
        # First eight legal dual-polarization evaluation inputs; no score selection.
        qrows = []
        for row in candidates:
            views = task.data.prepare(row)
            if all(v is not None and v['valid_target'] for v in views):
                qrows.append(row)
            if len(qrows) == 8:
                break
        assert len(qrows) == 8
        tick = time.perf_counter()
        qdata, rdata = prepare(qrows), prepare(rrows)
        prep_seconds = time.perf_counter() - tick
        save(f'inputs_{classes}class.json', dict(query_ids=[r['id'] for r in qrows],
             reference_ids=[r['id'] for r in rrows], n_references=len(rrows),
             reference_labels=[r['label'] for r in rrows], preparation_seconds=prep_seconds,
             reference_available=rdata['available'].cpu().tolist()))
        for arm in args.arms:
            if classes == 8 and arm not in ('plain', 'se', 'full'):
                continue
            outputs.append(measure(arm, classes, len(rrows), qrows, rrows, qdata, rdata, args.repetitions))
        del qdata, rdata
        gc.collect()
        torch.cuda.empty_cache()
    summary = []
    for r in outputs:
        summary.append(dict(classes=r['classes'], arm=r['arm'], references=r['references'],
            executed_parameters=r['executed_parameters'], registered_parameters=r['registered_parameters'],
            updated_parameters=r['training']['unique_updated_parameters'],
            conv_matmul_gflops=r['conv_matmul_gflops'], batch1_median_ms=r['batch1']['median_ms'],
            batch1_p90_ms=r['batch1']['p90_ms'], batch8_ms_per_chip=r['batch8']['median_ms'] / 8,
            batch8_chips_per_second=8000 / r['batch8']['median_ms'],
            peak_allocated_mib=r['batch1']['peak_allocated_mib'],
            incremental_peak_mib=r['batch1']['incremental_peak_mib'],
            reference_cache_mib=r['reference_cache_mib'],
            reference_build_median_ms=r.get('reference_build', {}).get('median_ms'),
            scope=r['scope']))
    pd.DataFrame(summary).to_csv(ROOT / 'summary.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
