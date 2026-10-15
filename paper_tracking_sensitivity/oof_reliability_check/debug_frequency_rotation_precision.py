"""Locate raw-encoder versus cached-head numerical replay differences."""

import numpy as np
import torch
from frequency_rotation_probe import EXP, sample_rows
from extract_learned_frequency_maps import build_encoder
from learned_frequency_model import FrequencyResidual
from c4_condition_descriptors import MANIFESTS

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
folder = EXP/'tasks/f0921'
with np.load(folder/'target.npz') as data:
    rows, _ = sample_rows(data['support'], data['eval_masks'], 'f0921')
    rows = rows[:128]
    c = torch.as_tensor(data['conditions'][rows], dtype=torch.float32, device='cuda')
with np.load(folder/'selective_embedding.npz') as f:
    expected = f['embedding'][rows]
pixels = np.load(MANIFESTS['f0921'].parent/'inputs128.npy', mmap_mode='r')
maps = np.load(EXP/'maps/f0921/maps.npy', mmap_mode='r')
encoder = build_encoder(torch.device('cuda'))
model = FrequencyResidual('selective').cuda().eval()
model.load_state_dict(torch.load(folder/'selective.pt', map_location='cpu', weights_only=True)['state'])
x = (torch.from_numpy(np.array(pixels[rows])).cuda().float()/255.-.5)/.25
cached = torch.from_numpy(np.array(maps[rows], dtype=np.float32)).cuda()
with torch.inference_mode():
    for benchmark in (False, True):
        torch.backends.cudnn.benchmark = benchmark
        for batch in (64, 128):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                raw = encoder(x[:batch]).float()
                z, _ = model(raw, c[:batch])
                zc, _ = model(cached[:batch], c[:batch])
            print(dict(benchmark=benchmark, batch=batch,
                map_error=float((raw-cached[:batch]).abs().max()),
                raw_embedding_error=float(np.abs(z.cpu().numpy()-expected[:batch]).max()),
                cached_embedding_error=float(np.abs(zc.cpu().numpy()-expected[:batch]).max())), flush=True)
