"""Extract the actual paired-R50 / VH-SSL recipe for public OpenSARShip2."""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

from opensarship2_labels import map_opensarship2_labels, summarize_mapping

WS = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline')
SOURCE = WS / 'artifacts/eight_class_adaptive_20260916'
TARGET = Path('D:/Documents/project_root/project_data/public_opensarship2/opensarship2')
sys.path.insert(0, str(WS))
from extract_full_six_spatial_features import build_encoder, read_one


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def extract(paths, paired, ssl, device, batch_size=64):
    tf = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])

    def read_pair(pair):
        pixels = np.stack([read_one(Path(p)) for p in pair])
        with Image.open(pair[1]) as im:
            ssl_pixels = tf(im.convert('RGB'))
        return pixels, ssl_pixels

    base_features, ssl_features = [], []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for begin in range(0, len(paths), batch_size):
            batch = list(pool.map(read_pair, paths[begin:begin+batch_size]))
            x = torch.from_numpy(np.stack([p[0] for p in batch])).to(device).float()
            x = (x / 255. - .5) / .25
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                spatial = paired(x).flatten(2).transpose(1, 2)
            rounded = spatial.float().cpu().numpy().astype(np.float16)
            base_features.append(rounded.astype(np.float32).mean(1))
            # Original SSL cache was CPU float32. GPU float32 is replay-checked below.
            by_shape = {}
            for index, (_, pixels) in enumerate(batch):
                by_shape.setdefault(tuple(pixels.shape), []).append(index)
            extra = np.empty((len(batch), 512), dtype=np.float32)
            for indices in by_shape.values():
                z = torch.stack([batch[i][1] for i in indices]).to(device)
                extra[indices] = ssl(z).flatten(1).cpu().numpy()
            ssl_features.append(extra)
            if begin == 0 or (begin // batch_size) % 10 == 0 or begin+batch_size >= len(paths):
                print(f'EXTRACT {min(begin+batch_size, len(paths))}/{len(paths)} '
                      f'{time.monotonic()-started:.1f}s', flush=True)
    return np.concatenate(base_features), np.concatenate(ssl_features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'features.npz').exists():
        raise ValueError('target features already exist; do not overwrite')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    source = pd.read_csv(SOURCE/'manifest.csv', keep_default_na=False, low_memory=False)
    target = pd.read_csv(TARGET/'metadata/opensarship2_manifest_native.csv', keep_default_na=False)
    mapped = map_opensarship2_labels(target)
    mapped.insert(0, 'target_manifest_row', np.arange(len(mapped)))
    mapped.to_csv(args.out/'label_mapping.csv', index=False)
    write_json(args.out/'label_mapping_summary.json', summarize_mapping(mapped, source))
    selected = mapped.loc[mapped.label_included].copy()
    selected.to_csv(args.out/'target_manifest.csv', index=False)
    paired = build_encoder(device)
    checkpoint = WS/'e206_out/ssl_resnet18.pt'
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    backbone = resnet18(weights=None)
    missing, unexpected = backbone.load_state_dict(ck['backbone'], strict=False)
    if set(missing) != {'fc.weight', 'fc.bias'} or unexpected:
        raise ValueError(f'unexpected SSL checkpoint keys: {missing}, {unexpected}')
    ssl = torch.nn.Sequential(*list(backbone.children())[:-1]).to(device).eval()
    # These fixed rows span the source cache; no target outcomes select the replay.
    replay_rows = np.linspace(0, len(source)-1, 8, dtype=int)
    paths = list(source.iloc[replay_rows][['image_relpath_vv', 'image_relpath_vh']].itertuples(index=False, name=None))
    replay_base, replay_ssl = extract(paths, paired, ssl, device)
    old_base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')[replay_rows]
    with np.load(WS/'b2_ssl_feats_38091.npz') as d:
        old_ssl = d['F'][replay_rows]
    replay = dict(rows=replay_rows.tolist(), ssl_checkpoint=str(checkpoint), ssl_epoch=int(ck['epoch']))
    for name, new, old in [('paired_r50', replay_base, old_base), ('vh_ssl', replay_ssl, old_ssl)]:
        relative_l2 = np.linalg.norm(new-old, axis=1)/np.maximum(np.linalg.norm(old, axis=1), 1e-12)
        replay[name] = dict(max_abs_error=float(np.max(np.abs(new-old))),
                            relative_l2_by_row=relative_l2.tolist())
    write_json(args.out/'encoder_replay.json', replay)
    print('REPLAY '+json.dumps(replay), flush=True)
    if max(replay['paired_r50']['relative_l2_by_row']) > .02 or max(replay['vh_ssl']['relative_l2_by_row']) > 1e-4:
        raise ValueError('source encoder replay mismatch; inspect provenance before target extraction')
    paths = list(selected[['chip_path_vv', 'chip_path_vh']].itertuples(index=False, name=None))
    base, extra = extract(paths, paired, ssl, device)
    if not np.isfinite(base).all() or not np.isfinite(extra).all():
        raise ValueError('nonfinite target features')
    np.savez_compressed(args.out/'features.npz', sample_ids=selected.sample_id.to_numpy(dtype=str),
                        paired_r50=base, vh_ssl=extra)
    write_json(args.out/'feature_recipe.json', dict(
        base='VV/VH two-channel ImageNet ResNet50 V2, 2x2 pool, spatial fp16 then mean fp32',
        base_input='L bilinear 128x128, (x/255 - 0.5)/0.25, CUDA bfloat16',
        extra='VH RGB ResNet18 e206 SSL epoch40, average pool, float32',
        extra_input='Resize(128), ToTensor, no ImageNet normalization',
        ssl_pretraining='E:/F_0921_pool_v1, transductive-unlabeled provenance; not independent encoder',
        rows=len(selected), torch_version=torch.__version__,
        note='Corrects earlier VV-only base descriptions. No quality or interference filtering.'))
    print('FEATURES SAVED', flush=True)


if __name__ == '__main__':
    main()
