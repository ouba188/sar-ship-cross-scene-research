"""Label-blind matched image rotations on a bounded common-query sample."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.nn import functional as F

from extract_learned_frequency_maps import build_encoder
from learned_frequency_model import FrequencyResidual
from train_frequency_information import InformationResidual
from c4_condition_descriptors import load_manifest, read_pair, MANIFESTS

EXP = Path('E:/Hermes/paper_tracking/sensitivity/learned_frequency_residual_20261009')
INFO = Path('E:/Hermes/paper_tracking/sensitivity/frequency_information_content_20261009')
OUT = INFO/'rotation_probe'


def sample_rows(support, eval_masks, dataset):
    query = np.flatnonzero(eval_masks.all(0))
    rng = np.random.default_rng(20261009)
    chosen = np.sort(rng.choice(query, min(len(query), 512 if dataset == 'self24ports' else 2048), replace=False))
    rows = np.unique(np.r_[support.ravel(), chosen])
    return rows, chosen


def encode(folder, encoder, manifests, executor):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = OUT/info['dataset']/info['task']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'ENCODED.json').exists():
        return
    with np.load(folder/'target.npz') as data:
        rows, query = sample_rows(data['support'], data['eval_masks'], info['dataset'])
        manifest_rows = data['rows'][rows]
        conditions = data['conditions'][rows]
    frame, pixels = manifests[info['dataset']]
    models, originals = {}, {}
    models['signed'] = FrequencyResidual('selective').cuda().eval()
    models['signed'].load_state_dict(torch.load(folder/'selective.pt', map_location='cpu', weights_only=True)['state'])
    unsigned_folder = INFO/'retrained'/info['dataset']/info['task']
    models['unsigned'] = InformationResidual('unsigned_bands').cuda().eval()
    models['unsigned'].load_state_dict(torch.load(unsigned_folder/'unsigned_bands.pt', map_location='cpu', weights_only=True)['state'])
    for mode, path in [('signed', folder/'selective_embedding.npz'), ('unsigned', unsigned_folder/'unsigned_bands.npz')]:
        with np.load(path) as f:
            originals[mode] = f['embedding'][rows]
    cached_embeddings = {m: z.copy() for m, z in originals.items()}
    values = {m: np.empty((len(rows), 256), np.float32) for m in models}
    map_cache = np.load(EXP/'maps'/info['dataset']/'maps.npy', mmap_mode='r')
    def read(i):
        if pixels is not None:
            return np.array(pixels[i], copy=True)
        keys = ['image_relpath_vv', 'image_relpath_vh'] if info['dataset'] == 'self24ports' else ['chip_path_vv', 'chip_path_vh']
        return read_pair([frame[k].iat[i] for k in keys])[0]
    replay = {}
    with torch.inference_mode():
        for lo in range(0, len(rows), 128):
            rr = manifest_rows[lo:lo+128]
            pixels_batch = np.stack(list(executor.map(read, rr)))
            x = (torch.from_numpy(pixels_batch).cuda().float()/255.-.5)/.25
            c = torch.as_tensor(conditions[lo:lo+len(rr)], dtype=torch.float32, device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                original_map = encoder(x)
                for mode, model in models.items():
                    old, _ = model(original_map.float(), c)
                    originals[mode][lo:lo+len(rr)] = old.cpu().numpy()
                    if lo == 0:
                        cached_map = torch.from_numpy(np.asarray(map_cache[rr], dtype=np.float32)).cuda()
                        replayed, _ = model(cached_map, c)
                        error = float((old-replayed).abs().max())
                        replay[mode] = error
                        print(f'PRECISION {info["task"]} {mode}: map={(original_map.float()-cached_map).abs().max().item()} embedding={error}', flush=True)
                        np.testing.assert_allclose(old.cpu().numpy(), replayed.cpu().numpy(), atol=2e-5, rtol=2e-5)
                rotated_map = encoder(torch.rot90(x, 2, (-2, -1)))
                for mode, model in models.items():
                    z, _ = model(rotated_map.float(), c)
                    values[mode][lo:lo+len(rr)] = z.cpu().numpy()
    np.savez_compressed(out/'embeddings.npz', local_rows=rows, query=query, manifest_rows=manifest_rows,
        signed_original=originals['signed'], unsigned_original=originals['unsigned'],
        signed_rotated=values['signed'], unsigned_rotated=values['unsigned'],
        signed_cached=cached_embeddings['signed'], unsigned_cached=cached_embeddings['unsigned'])
    (out/'ENCODED.json').write_text(json.dumps(dict(dataset=info['dataset'], task=info['task'],
        sampled_query_n=len(query), support_union_n=len(rows)-len(query),
        selection='Fixed seed; intersection of five query masks; no labels or outcomes read',
        raw_vs_cached_map_same_batch_embedding_max_error=replay,
        batch_roundoff_max={m:float(np.abs(originals[m]-cached_embeddings[m]).max()) for m in models},
        precision='Recompute zero and180degree paths in identical batches; separately account for historical batch-size roundoff'), indent=2))
    print(f'ROTATION {info["task"]}: {len(query)} queries + {len(rows)-len(query)} supports', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    # Match the original R50 extractor rather than retuning BF16 convolution kernels.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = build_encoder(torch.device('cuda'))
    manifests = {d: (load_manifest(d), np.load(MANIFESTS[d].parent/'inputs128.npy', mmap_mode='r') if d == 'f0921' else None)
                 for d in ('self24ports', 'f0921', 'oss2')}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
            encode(marker.parent, encoder, manifests, executor)
