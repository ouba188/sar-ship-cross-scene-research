"""Frozen native-resolution C4 ResNet50 target features; no classifier training."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet50

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / 'context_expansion_20261022'
CHECKPOINT = Path('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth')
BATCH = 32
ATOL = 5e-5
RTOL = 5e-5


def read_pixels(path):
    with Image.open(path) as im:
        if im.mode != 'L':
            raise ValueError(f'Expected native grayscale PNG: {path}, {im.mode}')
        return np.array(im, dtype=np.uint8, copy=True)


def tensor(images, mean, std):
    x = torch.from_numpy(np.stack(images)).to(device='cuda', dtype=torch.float32)
    x = x[:, None].expand(-1, 3, -1, -1) / 255.0
    return (x - mean) / std


@torch.inference_mode()
def encode(model, x):
    values3, values4 = [], []
    for rotation in range(4):
        z = x.rot90(rotation, (-2, -1)).contiguous()
        z = model.maxpool(model.relu(model.bn1(model.conv1(z))))
        z = model.layer2(model.layer1(z))
        z = model.layer3(z)
        values3.append(z.mean((-2, -1)))
        z = model.layer4(z)
        values4.append(z.mean((-2, -1)))
    return (torch.stack(values3).mean(0).cpu().numpy(),
            torch.stack(values4).mean(0).cpu().numpy())


@torch.inference_mode()
def single_reference(model, x):
    # Hooks on the ordinary full forward independently check stage traversal.
    values = {'layer3': [], 'layer4': []}
    handles = []
    for name in values:
        def capture(module, inputs, output, key=name):
            values[key].append(output.mean((-2, -1)).cpu().numpy())
        handles.append(getattr(model, name).register_forward_hook(capture))
    try:
        for rotation in range(4):
            model(x.rot90(rotation, (-2, -1)).contiguous())
    finally:
        for handle in handles:
            handle.remove()
    return {name: np.mean(np.stack(items), axis=0)[0] for name, items in values.items()}


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('ClearSAR CUDA device is required for this extraction')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    with (EXP / 'readout/targets.csv').open(encoding='utf-8-sig', newline='') as f:
        targets = list(csv.DictReader(f))
    ids = np.array([r['instance_id'] for r in targets])
    assert len(ids) == 5695 and len(set(ids)) == len(ids)
    buckets = defaultdict(list)
    shapes = np.empty((len(ids), 2, 2), dtype=np.int32)
    for i, row in enumerate(targets):
        for pol_index, pol in enumerate(('vv', 'vh')):
            path = EXP / 'readout/native_targets' / f'{ids[i]}_{pol}.png'
            with Image.open(path) as im:
                width, height = im.size
                assert im.mode == 'L', (str(path), im.mode)
            recorded = json.loads(row[f'target_window_{pol}'])
            assert [width, height] == recorded[2:], (str(path), recorded)
            shapes[i, pol_index] = height, width
            buckets[(height, width)].append((i, pol_index, path))

    model = resnet50(weights=None)
    model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=True), strict=True)
    model = model.float().cuda().eval().requires_grad_(False)
    assert not any(m.training for m in model.modules())
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in model.parameters())
    mean = torch.tensor([.485, .456, .406], device='cuda', dtype=torch.float32)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda', dtype=torch.float32)[None, :, None, None]
    features = {'layer3': np.empty((len(ids), 2048), dtype=np.float32),
                'layer4': np.empty((len(ids), 4096), dtype=np.float32)}
    seen = np.zeros((len(ids), 2), dtype=bool)
    ordered = sorted(buckets, key=lambda s: (-len(buckets[s]), s))
    candidates = [ordered[0], max(ordered, key=lambda s: s[0] * s[1])]
    rectangles = [s for s in ordered if s[0] != s[1]]
    if rectangles:
        candidates.append(rectangles[0])
    samples = []
    for shape in dict.fromkeys(candidates):
        # Prefer both polarizations where available in each selected shape.
        for pol in (0, 1):
            match = next((item for item in buckets[shape] if item[1] == pol), None)
            if match is not None and match not in samples:
                samples.append(match)
    batches, processed, max_batch = 0, 0, 0
    last_print = time.perf_counter()
    print(json.dumps(dict(status='extracting',targets=len(ids),polarization_images=2*len(ids),
                          shape_buckets=len(buckets),device=torch.cuda.get_device_name(0),
                          output=str(HERE/'deep_features.npz'))), flush=True)
    for shape in ordered:
        items = buckets[shape]
        # Cap only batch size for large native chips, never image dimensions.
        batch_size = min(BATCH, max(1, (BATCH * 128 * 128) // (shape[0] * shape[1])))
        max_batch = max(max_batch, batch_size)
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            x = tensor([read_pixels(path) for _, _, path in chunk], mean, std)
            z3, z4 = encode(model, x)
            for row, (i, pol, _) in enumerate(chunk):
                assert not seen[i, pol]
                features['layer3'][i, pol*1024:(pol+1)*1024] = z3[row]
                features['layer4'][i, pol*2048:(pol+1)*2048] = z4[row]
                seen[i, pol] = True
            batches += 1
            processed += len(chunk)
            if time.perf_counter() - last_print >= 20:
                print(json.dumps(dict(images_done=processed,images_total=2*len(ids),
                                      shape=shape,batch=len(chunk),seconds=time.perf_counter()-started)),flush=True)
                last_print = time.perf_counter()
    assert seen.all()
    assert all(np.isfinite(a).all() and np.all(np.linalg.norm(a, axis=1)>0) for a in features.values())
    replay = []
    for i, pol, path in samples:
        ref = single_reference(model, tensor([read_pixels(path)], mean, std))
        for name, width in [('layer3', 1024), ('layer4', 2048)]:
            actual = features[name][i, pol*width:(pol+1)*width]
            expected = ref[name]
            np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
            assert actual.argmax() == expected.argmax(), (ids[i], pol, name, 'top feature changed')
            replay.append(dict(instance_id=str(ids[i]),polarization=('vv','vh')[pol],layer=name,
                               shape=shapes[i,pol].tolist(),max_abs_error=float(np.max(np.abs(actual-expected))),
                               top_feature=int(actual.argmax()),single_top_feature=int(expected.argmax())))

    np.savez_compressed(HERE/'deep_features.npz', instance_ids=ids, **features)
    with np.load(HERE/'deep_features.npz') as saved:
        assert set(saved.files) == {'instance_ids', 'layer3', 'layer4'}
        assert np.array_equal(saved['instance_ids'], ids)
        for name, values in features.items():
            assert saved[name].dtype == np.float32 and np.array_equal(saved[name], values)
    summary = dict(status='complete',targets=len(ids),ships=len({r['mmsi'] for r in targets}),
                   passes=len({r['pass_group'] for r in targets}),ports=len({r['port'] for r in targets}),
                   checkpoint=str(CHECKPOINT),source_table=str(EXP/'readout/targets.csv'),
                   source_images=str(EXP/'readout/native_targets'),
                   output=str(HERE/'deep_features.npz'),local_cache_no_git=True,
                   keys={name:list(a.shape) for name,a in features.items()},instance_ids_shape=list(ids.shape),
                   dtype='float32',model_eval=True,frozen_parameters=True,autocast=False,tf32=False,
                   input='native grayscale uint8 /255, replicate RGB, ImageNet mean/std; no resize, interpolation or added input padding',
                   architecture_padding='original ResNet convolution/maxpool padding unchanged',
                   rotations_degrees=[0,90,180,270],rotation_operation='torch.rot90 exact pixel permutation',
                   pooling='spatial mean at layer3 and layer4 per rotation, then arithmetic mean over C4',
                   polarization_order=['VV','VH'],shape_buckets=len(buckets),max_batch=max_batch,batches=batches,
                   shape_counts=[dict(height=s[0],width=s[1],images=len(buckets[s])) for s in sorted(buckets)],
                   replay_selection='first VV/VH of most frequent, largest-area and first nonsquare shape when present',
                   replay_atol=ATOL,replay_rtol=RTOL,replay=replay,
                   replay_max_abs_error=max(r['max_abs_error'] for r in replay),
                   identical_top_feature=True,persisted_arrays_exact=True,all_finite=True,
                   torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),
                   gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-started,
                   scope='target extraction only; no neighbor features, fitting, depth selection or accuracy claim')
    (HERE/'extraction_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['status','targets','keys','replay_max_abs_error','identical_top_feature','seconds']},indent=2),flush=True)


if __name__ == '__main__':
    main()
